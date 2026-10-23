// Mesh export browser workflow with deterministic job responses.
const {chromium}=require('@playwright/test');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.SAGO_CHROME||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1600,height:1100}}),calls=[],errors=[],jobs=new Map();let serial=0;const polls=new Map();
 page.on('pageerror',e=>errors.push(e.message));
 const state={scene:'test.ply',knowledge_path:'/tmp/SAGO_knowledge.json',count:100,selected:0,objects:[{id:'duck',number:3,name:'Duck <b>test</b>',hidden:true,count:20},{id:'cat',number:4,name:'Cat',hidden:false,count:20}],cameras:[],frame_id:'f',canUndo:false,canRedo:false,next_object_id:1,ids_dirty:true,render_mode:'rgb',background_color:[0,0,0],visualization:{},camera:{yaw:0,pitch:0,radius:2,target:[0,0,0],up:[0,-1,0],fov:50,size:768,aspect:1.5},image:'data:image/svg+xml;base64,'+Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="768" height="512"/>').toString('base64')};
 await page.route('http://sago.test/**',async route=>{
  const req=route.request(),url=new URL(req.url()),json=value=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
  if(url.pathname==='/api/config')return json({token:'test',scene_loaded:true,gif_export_version:3});
  if(url.pathname==='/api/jobs'&&req.method()==='POST'){
   const {action,data}=req.postDataJSON();calls.push({action,data});let result=structuredClone(state);
   if(action==='mesh')result={items:[{number:3,name:'Duck',filename:'ID_3_Duck.obj',download:'/downloads/mesh-'+('a'.repeat(32))+'/ID_3/ID_3_Duck.obj',saved_path:'/tmp/outputs/ID_3_Duck.obj',vertices:100,faces:200}],generation_seconds:3};
   if(action==='rag'){
    if(data.operation==='test')result={connected:true,model:'test-model'};
    if(data.operation==='build')result={indexed:1,saved_path:'/tmp/SAGO_spatial.json',knowledge_path:state.knowledge_path};
    if(data.operation==='ask')result={answer:'The chair [ID 2] is closest. <b>plain text</b>',model:'test-model',referenced_ids:[2],retrieved_ids:[2],seconds:.2};
   }
   const id=String(++serial);jobs.set(id,result);return json({id});
  }
  if(url.pathname.startsWith('/api/jobs/')){const id=url.pathname.split('/').at(-1);const n=(polls.get(id)||0)+1;polls.set(id,n);if(jobs.get(id)?.items && n<4)return json({status:'running',done:35,message:'Training mesh'});return json({status:'done',result:jobs.get(id)});}
  const file=url.pathname==='/'?path.join(__dirname,'../frontend/index.html'):path.join(__dirname,'../frontend',url.pathname.replace('/assets/',''));
  if(!fs.existsSync(file))return route.fulfill({status:404});
  return route.fulfill({contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(file)});
 });
 const idle=()=>page.waitForFunction(()=>document.getElementById('busy-overlay').hidden&&document.getElementById('scene-info').textContent.includes('100'));
 try{
  await page.goto('http://sago.test/');await idle();
  await page.locator('#export-mesh').click();await page.locator('#mesh-dialog').waitFor({state:'visible'});
  assert.equal(await page.locator('#mesh-objects input').count(),2);assert.equal(await page.locator('#mesh-objects b').count(),0);
  await page.locator('#mesh-none').click();await page.locator('#mesh-objects input[value="duck"]').check();
  await page.locator('#mesh-dialog details summary').click();
  await page.locator('#mesh-root').fill('/custom/NeuS2');await page.locator('#mesh-python').fill('/custom/env/bin/python');
  assert(!(await page.locator('#mesh-dialog').textContent()).includes('NeuS2'));
  await page.locator('#mesh-parallel').fill('2');
  await page.locator('#mesh-steps').fill('500');await page.locator('#mesh-views').fill('8');
  await page.locator('#mesh-start').click();
  await page.locator('#cancel').waitFor({state:'visible'});assert.equal(await page.locator('#progress').getAttribute('max'),'100');
  await page.locator('#mesh-result-dialog').waitFor({state:'visible'});
  const call=calls.find(c=>c.action==='mesh');assert.deepEqual(call.data.ids,['duck']);assert.equal(call.data.neus2_root,'/custom/NeuS2');assert.equal(call.data.steps,500);assert.equal(call.data.parallel_jobs,2);assert.equal(call.data.views_per_orbit,8);
  assert((await page.locator('#mesh-results').textContent()).includes('100 vertices'));assert((await page.locator('#mesh-results a').getAttribute('href')).endsWith('.obj'));
  await page.locator('#mesh-result-close').click();await page.locator('#language-toggle').click();
  assert.equal(await page.locator('#export-mesh').textContent(),'导出物体 Mesh');
  await page.locator('#export-mesh').click();assert.equal(await page.locator('#mesh-start').textContent(),'开始导出 Mesh');await page.locator('#mesh-close').click();
  assert.deepEqual(errors,[]);console.log('Mesh UI: hidden-ID selection, settings, progress/cancel control, OBJ links, escaping and bilingual controls passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
