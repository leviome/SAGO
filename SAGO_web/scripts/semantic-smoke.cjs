// Browser UI test; no running backend or scene changes.
const {chromium}=require('@playwright/test');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.SAGO_CHROME||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1600,height:1100}});
 const calls=[],errors=[],jobs=new Map();let serial=0;
 page.on('pageerror',e=>errors.push(e.message));
 const state={scene:'test.ply',knowledge_path:'/tmp/SAGO_knowledge.json',count:100,selected:0,objects:[],cameras:[],frame_id:'f',canUndo:false,canRedo:false,next_object_id:1,ids_dirty:true,render_mode:'rgb',background_color:[0,0,0],visualization:{},camera:{yaw:0,pitch:0,radius:2,target:[0,0,0],up:[0,-1,0],fov:50,size:768,aspect:1.5},image:'data:image/svg+xml;base64,'+Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="768" height="512"/>').toString('base64')};
 await page.route('http://sago.test/**',async route=>{
  const req=route.request(),url=new URL(req.url()),json=value=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
  if(url.pathname==='/api/config')return json({token:'test',scene_loaded:true,gif_export_version:3});
  if(url.pathname==='/api/jobs'&&req.method()==='POST'){
   const {action,data}=req.postDataJSON();calls.push({action,data});let result=structuredClone(state);
   if(action==='semantic'){
    if(data.operation==='build'){state.ids_dirty=false;result={indexed:1,updated:1,saved_path:'/tmp/SAGO_knowledge.json'};}
    else if(data.operation==='clear'){state.semantic_match=null;result=structuredClone(state);}
    else{state.semantic_match='chair';result={...structuredClone(state),results:[{id:'chair',number:2,name:'Chair <b>plain text</b>',score:.345}],stale_ids:[3]};}
   }
   const id=String(++serial);jobs.set(id,result);return json({id});
  }
  if(url.pathname.startsWith('/api/jobs/'))return json({status:'done',result:jobs.get(url.pathname.split('/').at(-1))});
  const file=url.pathname==='/'?path.join(__dirname,'../frontend/index.html'):path.join(__dirname,'../frontend',url.pathname.replace('/assets/',''));
  if(!fs.existsSync(file))return route.fulfill({status:404});
  return route.fulfill({contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(file)});
 });
 const idle=()=>page.waitForFunction(()=>document.getElementById('busy-overlay').hidden&&document.getElementById('scene-info').textContent.includes('100'));
 try{
  await page.goto('http://sago.test/');await idle();assert(await page.locator('#semantic-build').isDisabled());
  state.objects=[{id:'chair',number:2,name:'Chair',hidden:true,count:20}];await page.reload();await idle();
  assert(await page.locator('#semantic-build').isEnabled());assert(await page.locator('#semantic-search').isDisabled());
  await page.locator('.gif-options > summary').click();await page.locator('#gif-elevation').fill('35');await page.locator('#semantic-build').click();
  await page.waitForFunction(()=>document.getElementById('semantic-status').textContent.includes('Indexed 1'));
  assert.equal(calls.at(-1).action,'semantic');assert.equal(calls.at(-1).data.elevation_degrees,35);assert.equal(calls.at(-1).data.size,512);
  await page.locator('#semantic-query').fill('a chair');await page.locator('#semantic-query').press('Enter');
  await page.waitForFunction(()=>document.getElementById('semantic-results').textContent.includes('ID 2'));
  assert.equal(calls.at(-1).data.text,'a chair');assert.equal(await page.locator('#semantic-results b').count(),0);
  assert((await page.locator('#semantic-status').textContent()).includes('3'));
  assert(await page.locator('#show-bbox').isChecked());assert(await page.locator('#semantic-clear').isEnabled());
  assert(await page.locator('#display button[data-mode="all"]').evaluate(el=>el.classList.contains('active')));
  await page.locator('#language-toggle').click();assert.equal(await page.locator('#semantic-build').textContent(),'建立 / 更新语义知识库');
  assert((await page.locator('#semantic-status').textContent()).includes('需要更新'));
  await page.locator('#language-toggle').click();assert.equal(await page.locator('#semantic-search').textContent(),'Find object IDs');
  await page.locator('#semantic-clear').click();await page.waitForFunction(()=>document.getElementById('semantic-clear').disabled);
  assert.equal(calls.at(-1).data.operation,'clear');
  assert.deepEqual(errors,[]);console.log('Semantic UI build, settings, Enter search, ranking, escaping and bilingual checks passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
