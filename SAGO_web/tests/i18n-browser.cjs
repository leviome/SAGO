const {chromium}=require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 try {
  const page=await browser.newPage({viewport:{width:1600,height:1000}});
  const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));
  const image='data:image/svg+xml;base64,'+Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="90"><rect width="160" height="90" fill="#314"/></svg>').toString('base64');
  let result,jobId=0;
  const snapshot=()=>({scene:'scene.ply',count:1000,selected:123,render_mode:'rgb',objects:[{id:'chair',number:1,name:'我的椅子',count:20,hidden:false}],camera:{fov:60,size:768,up:[0,-1,0],target:[0,0,0]},cameras:[],image,frame_id:'frame-'+jobId,canUndo:true,canRedo:false,ids_dirty:true,knowledge_path:'/tmp/场景/SAGO_knowledge.json',timing:{model:'sam2',refine_seconds:.3,total_seconds:.5}});
  await page.route('http://i18n.test/**',async route=>{
   const req=route.request(),url=new URL(req.url());
   if(url.pathname==='/api/config')return route.fulfill({json:{token:'test',nativePicker:false,scene_loaded:true}});
   if(url.pathname==='/api/jobs' && req.method()==='POST') {
    const job=req.postDataJSON();requests.push(job);result=snapshot();
    if(job.data.action==='preview_correction')result={...result,preview_id:'pending',preview_count:2,target_id:99};
    return route.fulfill({json:{id:String(++jobId)}});
   }
   if(url.pathname.startsWith('/api/jobs/'))return route.fulfill({json:{status:'done',result}});
   const filename=url.pathname==='/'?'index.html':url.pathname.replace('/assets/','');
   return route.fulfill({body:fs.readFileSync(path.join(__dirname,'../frontend',filename)),contentType:filename.endsWith('.js')?'text/javascript':filename.endsWith('.css')?'text/css':'text/html'});
  });
  await page.goto('http://i18n.test/');
  await page.locator('#status').filter({hasText:'Python backend connected'}).waitFor();
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  assert.equal(await page.locator('#open-scene').innerText(),'Open scene…');
  assert.equal(await page.locator('#text-prompt').getAttribute('placeholder'),'For example: bear, red chair');
  assert((await page.locator('#timing').innerText()).includes('Segmentation total: 0.500 s'));
  const unlocalized=await page.evaluate(()=>{
   const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT),texts=[];
   let node;while(node=walker.nextNode()) {
    if(['SCRIPT','STYLE'].includes(node.parentElement.tagName))continue;
    const value=node.nodeValue.replaceAll('我的椅子','').replaceAll('中文','');
    if(/[\u3400-\u9fff]/u.test(value))texts.push(value);
   }return texts;
  });
  assert.deepEqual(unlocalized,[],'English UI has untranslated strings');
  await page.locator('#object-name').fill('自定义名称');
  await page.locator('.manual-path summary').click();
  await page.locator('#path').fill('/tmp/我的场景/point_cloud.ply');
  const stats=await page.locator('#stats').innerText();
  await page.locator('#language-toggle').click();
  assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
  assert.equal(await page.locator('#open-scene').innerText(),'打开场景…');
  assert.equal(await page.locator('#object-name').inputValue(),'自定义名称');
  assert.equal(await page.locator('#path').inputValue(),'/tmp/我的场景/point_cloud.ply');
  assert((await page.locator('#timing').innerText()).includes('分割合计：0.500 秒'));
  assert.equal(await page.locator('#timing span').count(),1);
  assert.equal(await page.locator('#timing strong').count(),1);
  await page.reload();
  await page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
  assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
  await page.locator('#language-toggle').click();
  assert.equal(await page.locator('#stats').innerText(),stats);
  assert.equal(await page.locator('#objects .object-name-edit').innerText(),'我的椅子');
  await page.locator('#background-box').click();
  assert.equal(await page.locator('#correction-title').innerText(),'Correct Gaussian IDs');
  assert.deepEqual(await page.locator('#correction-mode option').allTextContents(),['Ellipse correction','Point correction']);
  await page.locator('#correction-id').fill('99');await page.locator('#correction-id').press('Enter');
  const rect=await page.locator('#canvas').boundingBox();
  await page.mouse.click(rect.x+rect.width*.5,rect.y+rect.height*.5);
  await page.locator('#status').filter({hasText:'Highlighted 2 visible Gaussians'}).waitFor();
  const before=requests.length;
  await page.locator('#language-toggle').click();
  assert.equal(await page.locator('#correction-confirm').innerText(),'确定修正（2）');
  assert((await page.locator('#status').innerText()).includes('已高亮 2 个可见高斯'));
  assert.equal(requests.length,before);
  assert(await page.locator('#prompt-mode').isDisabled());
  await page.locator('#language-toggle').click();
  assert.equal(await page.locator('#correction-confirm').innerText(),'Confirm correction (2)');
  assert.equal(await page.locator('#stats').innerText(),stats);
  await page.keyboard.press('Escape');
  await page.locator('#status').filter({hasText:'Exited correction mode'}).waitFor();
  await page.reload();
  await page.locator('#status').filter({hasText:'Python backend connected'}).waitFor();
  assert.equal(await page.locator('html').getAttribute('lang'),'en');
  await page.locator('#help').click();
  assert.equal(await page.locator('#guide h2').innerText(),'SAGO user guide');
  await page.locator('#close-help').click();
  assert.deepEqual(errors,[]);
  console.log('PASS: English default, static/dynamic translations, persistent toggle, timing children, user data, correction preview and guide.');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
