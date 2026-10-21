// Export UI behavior against a deterministic GPU job API.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const fs=require('node:fs');
const path=require('node:path');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.SAGO_CHROME || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[],requests=[];
  page.on('pageerror',error=>errors.push(error.message));
  const png='data:image/svg+xml;base64,'+Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"/>').toString('base64');
  const gif=Buffer.from('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7','base64');
  let selected=0,id=0,run=0,cancelled=false,batchMode=false,holdBatch=false,legacy=true;
  const jobs=new Map(),download='/downloads/'+'a'.repeat(32)+'.gif';
  const objects=[{id:'id-a',number:1,name:'Chair',hidden:true,count:3},{id:'id-b',number:2,name:'Table',hidden:false,count:4},{id:'id-c',number:3,name:'Lamp',hidden:false,count:5}];
  const snapshot=()=>({scene:'test',count:100,selected:batchMode?0:selected,objects:batchMode?objects:[],camera:{yaw:0,pitch:0,radius:3,fov:50,size:768,up:[0,0,1],target:[0,0,0]},cameras:[],image:png,canUndo:false,canRedo:false});
  await page.route('http://sago.test/**',async route=>{
   const req=route.request(),url=new URL(req.url());
   if(url.pathname.startsWith('/downloads/'))return route.fulfill({body:gif,headers:{'content-type':'image/gif','content-disposition':'attachment; filename="'+(url.searchParams.get('filename') || 'foreground.gif')+'"'}});
   if(url.pathname==='/api/config')return route.fulfill({json:{token:'test',scene_loaded:true,gif_export_version:legacy?undefined:3}});
   if(url.pathname.endsWith('/cancel')){cancelled=true;return route.fulfill({json:{ok:true}});}
   if(url.pathname==='/api/jobs' && req.method()==='POST'){
    const job=req.postDataJSON();requests.push(job);
    if(job.action==='gif'){run++;cancelled=false;}
    const single={download,saved_path:'/test/outputs/example.gif',frames:36,fps:20,size:256,duration_seconds:1.8,generation_seconds:1.25};
    const result=job.action==='gif'?(job.data.source==='all'?{source:'all',items:objects.map(obj=>{const filename=`ID_${obj.number}_${obj.name}.gif`;return {...single,...obj,filename,download:download+'?filename='+filename};})}:single):snapshot();
    const key=String(++id);jobs.set(key,{action:job.action,result,hold:run===1||holdBatch});return route.fulfill({json:{id:key}});
   }
   if(url.pathname.startsWith('/api/jobs/')){
    const job=jobs.get(url.pathname.split('/').at(-1));
    return route.fulfill({json:job.action==='gif' && job.hold?(cancelled?{status:'cancelled',error:'GIF 生成已取消'}:{status:'running',done:43,message:'GIF 渲染 4/36'}):{status:'done',result:job.result}});
   }
   const filename=url.pathname==='/'?'index.html':url.pathname.replace('/assets/','');
   return route.fulfill({body:fs.readFileSync(path.join(__dirname,'../frontend',filename)),contentType:filename.endsWith('.js')?'text/javascript':filename.endsWith('.css')?'text/css':'text/html'});
  });
  await page.goto('http://sago.test/');
  await page.locator('#status').filter({hasText:'Python backend connected'}).waitFor();
  assert(await page.locator('#export-gif').isDisabled());
  assert(await page.locator('#export-all-gif').isDisabled());
  assert.equal(await page.locator('#gif-source').count(),0);
  selected=10;await page.reload();await page.locator('#status').filter({hasText:'Python backend connected'}).waitFor();
  assert(await page.locator('#export-gif').isEnabled());
  await page.locator('#export-gif').click();
  await page.locator('#status').filter({hasText:'The running backend is outdated'}).waitFor();
  assert.equal(requests.filter(r=>r.action==='gif').length,0);
  legacy=false;await page.reload();await page.locator('#status').filter({hasText:'Python backend connected'}).waitFor();
  await page.locator('.gif-options summary').click();
  assert.equal(await page.locator('#gif-elevation').inputValue(),'60');
  await page.locator('#gif-elevation').fill('25');
  await page.locator('#gif-size').selectOption('256');await page.locator('#gif-frames').selectOption('36');
  await page.locator('#export-gif').click();
  await page.locator('#busy-message').filter({hasText:'Rendering GIF 4/36'}).waitFor();
  assert.equal(await page.locator('#progress').getAttribute('max'),'100');
  assert.equal(await page.locator('#cancel').innerText(),'Cancel generation');
  await page.locator('#cancel').click();await page.locator('#status').filter({hasText:'GIF generation cancelled'}).waitFor();
  assert(!(await page.locator('#gif-dialog').evaluate(e=>e.open)));
  assert((await page.locator('#stats').innerText()).includes('10 / 100'));
  await page.locator('#export-gif').click();await page.locator('#gif-dialog').waitFor({state:'visible'});
  await page.locator('#gif-preview').evaluate(image=>image.decode());
  assert((await page.locator('#gif-info').innerText()).includes('36 frames'));
  assert.equal(await page.locator('#gif-saved-path').innerText(),'Saved to: /test/outputs/example.gif');
  const sent=requests.filter(r=>r.action==='gif');assert.equal(sent.length,2);
  assert.deepEqual(sent[1].data,{source:'current',size:256,frames_per_orbit:36,elevation_degrees:25,fps:20});
  const downloading=page.waitForEvent('download');await page.locator('#gif-download').click();
  const file=await downloading;assert(file.suggestedFilename().endsWith('.gif'));
  await page.locator('#gif-close').click();await page.locator('#language-toggle').click();
  await page.locator('#gif-open').click();assert((await page.locator('#gif-info').innerText()).includes('36 帧'));
  assert.equal(await page.locator('#gif-title').innerText(),'前景 GIF');
  await page.locator('#gif-close').click();await page.locator('#language-toggle').click();
  batchMode=true;await page.reload();await page.locator('#status').filter({hasText:'Python backend connected'}).waitFor();
  assert(await page.locator('#export-gif').isDisabled(),'Current foreground still requires a selection');
  assert(await page.locator('#export-all-gif').isEnabled(),'All objects do not require current foreground');
  await page.locator('#multi-none').click();
  await page.waitForFunction(()=>!document.getElementById('export-all-gif').disabled);
  assert.equal(await page.locator('#foreground-list input:checked').count(),0);
  await page.locator('#export-all-gif').click();await page.locator('#gif-dialog').waitFor({state:'visible'});
  const allData=requests.filter(r=>r.action==='gif').at(-1).data;
  assert.equal(allData.source,'all');
  assert.equal(allData.elevation_degrees,60);
  assert(!('ids' in allData),'All objects are determined by the backend');
  assert.equal(await page.locator('#gif-item option').count(),3);
  assert.equal(await page.locator('#gif-item option').first().innerText(),'ID 1 · Chair');
  await page.locator('#gif-item').selectOption('1');
  assert((await page.locator('#gif-preview').getAttribute('src')).includes('ID_2_Table.gif'));
  assert.equal(await page.locator('#gif-download').getAttribute('download'),'ID_2_Table.gif');
  const batchDownload=page.waitForEvent('download');await page.locator('#gif-download').click();
  assert.equal((await batchDownload).suggestedFilename(),'ID_2_Table.gif');
  await page.locator('#gif-close').click();holdBatch=true;
  await page.locator('#export-all-gif').click();await page.locator('#cancel').waitFor({state:'visible'});
  await page.locator('#cancel').click();await page.locator('#status').filter({hasText:'GIF generation cancelled'}).waitFor();
  await page.locator('#gif-open').click();assert.equal(await page.locator('#gif-item option').count(),3);
  assert.deepEqual(errors,[]);
  console.log('PASS: foreground guard, GIF settings/job, progress, cancel, preserved selection, preview, download, reopen and bilingual UI, all-object batch without a selection or checked visualization IDs, hidden IDs, independent named downloads, and preserved previous batch after cancel.');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
