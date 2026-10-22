// Run with @playwright/test available in NODE_PATH; all API traffic is mocked.
const {chromium}=require('@playwright/test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const frontend=path.resolve(__dirname,'../frontend');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.SAGO_CHROME || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1600,height:1000}});
 const errors=[],calls=[],jobs=new Map();let serial=0;
 page.on('pageerror',e=>errors.push(e.message));
 const svg='<svg xmlns="http://www.w3.org/2000/svg" width="768" height="512"><rect width="768" height="512" fill="#302030"/></svg>';
 const state={scene:'test.ply',count:100,selected:0,objects:[],cameras:[],frame_id:'f',canUndo:false,canRedo:false,next_object_id:1,render_mode:'rgb',background_color:[0,0,0],visualization:{},camera:{yaw:0,pitch:0,radius:2,target:[0,0,0],up:[0,-1,0],fov:50,size:768,aspect:1.5},image:'data:image/svg+xml;base64,'+Buffer.from(svg).toString('base64')};
 await page.route('http://sago.test/**',async route=>{
  const request=route.request(),url=new URL(request.url());
  const json=value=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
  if(url.pathname==='/api/config') return json({token:'test',scene_loaded:true,gif_export_version:3,nativePicker:true});
  if(url.pathname==='/api/jobs' && request.method()==='POST') {
   const {action,data}=request.postDataJSON();calls.push({action,data});
   if(data.background_color) state.background_color=data.background_color;
   let result=structuredClone(state);
   if(action==='objects' && data.action==='complete') {state.selected+=5;state.canUndo=true;result={...structuredClone(state),completion:{added:5,selected:state.selected}};}
   if(action==='capture') result={...result,capture:'cap',capture_mode:'all'};
   if(action==='load_model') result={model:'sam2',model_load_seconds:.01};
   if(action==='preview') result={masks:[],scores:[],timing:{}};
   const id=String(++serial);jobs.set(id,result);return json({id});
  }
  if(url.pathname.startsWith('/api/jobs/')) {
   // Slow frames exercise panning and key release while a request is in flight.
   await new Promise(r=>setTimeout(r,60));
   return json({status:'done',result:jobs.get(url.pathname.split('/').at(-1))});
  }
  const file=url.pathname==='/'?path.join(frontend,'index.html'):path.join(frontend,url.pathname.replace('/assets/',''));
  if(!fs.existsSync(file)) return route.fulfill({status:404,body:'Missing test asset'});
  return route.fulfill({contentType:file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html',body:fs.readFileSync(file)});
 });
 try {
  await page.goto('http://sago.test/');await page.locator('#canvas:not([hidden])').waitFor();
  const idle=()=>page.waitForFunction(()=>!document.getElementById('background-flip').disabled);
  await idle();assert(await page.locator('#complete-cube').isDisabled());assert(await page.locator('#complete-cylinder').isDisabled());
  await page.locator('#background-flip').click();await idle();
  assert.equal(await page.locator('#background-r').inputValue(),'1');
  await page.locator('#background-flip').click();await idle();
  assert.equal(await page.locator('#background-r').inputValue(),'0');
  assert.equal(await page.locator('#background-flip').innerText(),'Black / White');
  const countMove=()=>calls.filter(c=>c.action==='view' && c.data.move).length;
  await page.locator('#canvas').focus();
  for(const [key,axis,sign] of [['w',1,1],['s',1,-1],['a',0,-1],['d',0,1]]) {
   const count=countMove();await page.keyboard.down(key);await page.waitForTimeout(200);await page.keyboard.up(key);
   await idle();assert(countMove()>count,`No movement for ${key}`);
   const move=calls.filter(c=>c.data.move).at(-1).data.move;assert(move[axis]*sign>0);
   const released=countMove();await page.waitForTimeout(150);assert.equal(countMove(),released);
  }
  const beforeText=countMove();await page.locator('#object-name').fill('wasd');
  await page.locator('#object-name').press('w');await page.waitForTimeout(150);assert.equal(countMove(),beforeText);
  await page.locator('#canvas').focus();
  await page.keyboard.down('w');await page.waitForTimeout(100);
  await page.evaluate(()=>window.dispatchEvent(new Event('blur')));await page.keyboard.up('w');await idle();
  const afterBlur=countMove();await page.waitForTimeout(150);assert.equal(countMove(),afterBlur);
  const rect=await page.locator('#canvas').boundingBox(),x=rect.x+rect.width*.5,y=rect.y+rect.height*.5;
  await page.mouse.move(x,y);await page.mouse.down({button:'right'});
  await page.mouse.move(x+70,y+30,{steps:8});await page.waitForTimeout(180);
  assert(calls.some(c=>c.action==='view' && c.data.pan),'Right drag must pan before mouse release');
  await page.mouse.up({button:'right'});await idle();
  const pan=calls.filter(c=>c.data.pan).map(c=>c.data.pan).reduce((a,b)=>a.map((x,i)=>x+b[i]),[0,0]);
  assert(pan[0]>0 && pan[1]>0);
  state.selected=20;await page.reload();await idle();
  for(const shape of ['cube','cylinder']) {
   await page.locator(`#complete-${shape}`).click();await idle();
   assert.equal(calls.filter(c=>c.action==='objects' && c.data.action==='complete').at(-1).data.shape,shape);
   assert(await page.locator('#display button[data-mode="foreground"]').evaluate(el=>el.classList.contains('active')));
  }
  await page.locator('#prompt-mode').check();
  await page.waitForFunction(()=>document.getElementById('model-load-dialog').open);
  await page.waitForFunction(()=>!document.getElementById('model-load-dialog').open);
  assert(await page.locator('#background-flip').isDisabled());
  assert(await page.locator('#complete-cube').isDisabled());assert(await page.locator('#complete-cylinder').isDisabled());
  const beforePrompt=countMove();await page.locator('#canvas').focus();await page.keyboard.press('w');await page.waitForTimeout(100);assert.equal(countMove(),beforePrompt);
  await page.mouse.click(x,y,{button:'left'});await page.waitForTimeout(300);
  await page.mouse.click(x,y,{button:'right'});await page.waitForTimeout(300);
  const prompt=calls.filter(c=>c.action==='preview').at(-1);assert(prompt && prompt.data.background.length===1);
  await page.locator('#language-toggle').click();assert.equal(await page.locator('#background-flip').innerText(),'黑 / 白');assert.equal(await page.locator('#complete-cylinder').innerText(),'圆柱体补全');
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,requests:calls.length,checks:['cube/cylinder completion and foreground display','completion locks','black/white','W/S/A/D and release','input exclusion','blur stops motion','live right drag','prompt locking','right-click background prompt','bilingual labels']}));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
