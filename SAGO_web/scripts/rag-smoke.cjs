// GaussianRAG browser contract test; API jobs are mocked, no provider credentials needed.
const {chromium}=require('@playwright/test');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.SAGO_CHROME||'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1600,height:1100}}),calls=[],errors=[],jobs=new Map();let serial=0;
 page.on('pageerror',e=>errors.push(e.message));
 const state={scene:'test.ply',knowledge_path:'/tmp/SAGO_knowledge.json',count:100,selected:0,objects:[],cameras:[],frame_id:'f',canUndo:false,canRedo:false,next_object_id:1,ids_dirty:true,render_mode:'rgb',background_color:[0,0,0],visualization:{},camera:{yaw:0,pitch:0,radius:2,target:[0,0,0],up:[0,-1,0],fov:50,size:768,aspect:1.5},image:'data:image/svg+xml;base64,'+Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="768" height="512"/>').toString('base64')};
 await page.route('http://sago.test/**',async route=>{
  const req=route.request(),url=new URL(req.url()),json=value=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
  if(url.pathname==='/api/config')return json({token:'test',scene_loaded:true,gif_export_version:3});
  if(url.pathname==='/api/jobs'&&req.method()==='POST'){
   const {action,data}=req.postDataJSON();calls.push({action,data});let result=structuredClone(state);
   if(action==='rag'){
    if(data.operation==='test')result={connected:true,model:'test-model'};
    if(data.operation==='build')result={indexed:1,saved_path:'/tmp/SAGO_spatial.json',knowledge_path:state.knowledge_path};
    if(data.operation==='ask')result={answer:'The chair [ID 2] is closest. <b>plain text</b>',model:'test-model',referenced_ids:[2],retrieved_ids:[2],seconds:.2};
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
  await page.goto('http://sago.test/');await idle();assert(await page.locator('#rag-build').isDisabled());
  await page.locator('details[data-panel="rag"] .control-body > details > summary').click();
  await page.locator('#rag-base').fill('https://example.invalid/v1');await page.locator('#rag-model').fill('test-model');await page.locator('#rag-key').fill('session-secret');
  assert.equal(await page.locator('#rag-key').getAttribute('type'),'password');
  await page.locator('#rag-test').click();await page.waitForFunction(()=>document.getElementById('rag-status').textContent.includes('API connected'));
  state.objects=[{id:'chair',number:2,name:'Chair',hidden:true,count:20}];await page.reload();await idle();
  assert.equal(await page.locator('#rag-key').inputValue(),'');
  await page.locator('#rag-describe').check();await page.locator('#rag-build').click();
  await page.waitForFunction(()=>document.getElementById('rag-status').textContent.includes('Spatial knowledge saved'));
  assert(calls.at(-1).data.describe);assert.equal(calls.at(-1).data.elevation_degrees,60);
  await page.locator('#rag-question').fill('Which object is nearest to the table?');await page.locator('#rag-question').press('Enter');
  await page.waitForFunction(()=>document.getElementById('rag-answer').textContent.includes('[ID 2]'));
  assert.equal(await page.locator('#rag-answer b').count(),0);assert.equal(calls.at(-1).data.operation,'ask');
  await page.locator('#language-toggle').click();assert.equal(await page.locator('#rag-ask').textContent(),'询问场景');
  assert((await page.locator('#rag-status').textContent()).includes('引用 ID'));
  const saved=await page.evaluate(()=>JSON.stringify(localStorage));assert(!saved.includes('session-secret'));
  assert.deepEqual(errors,[]);console.log('GaussianRAG UI: settings, no key persistence, build, Enter question, answer, references, bilingual and escaping passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
