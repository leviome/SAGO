const {chromium}=require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1920,height:1000}}), errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const automatic=()=>page.locator('#preview-status[data-state="done"]').waitFor({timeout:240000});
 const ready=async()=>{
  await page.locator('#busy-overlay').waitFor({state:'hidden',timeout:240000});
  if(await page.locator('#status').evaluate(el=>el.classList.contains('error')))throw Error(await page.locator('#status').innerText());
 };
 const dismissLoad=async()=>{
  await page.locator('#model-load-dialog').waitFor();
  await page.locator('#model-load-spinner').waitFor({state:'hidden',timeout:240000});
  if(!(await page.locator('#model-load-time').innerText()).includes('模型加载用时'))throw Error('Missing load time');
  await page.locator('#model-load-dialog').waitFor({state:'hidden',timeout:10000});
 };
 try {
  await page.addInitScript(()=>localStorage.setItem('sago-language','zh'));
    await page.goto('http://127.0.0.1:8080');
  await page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
  if(await page.locator('#seg-model').inputValue()!=='sam2')throw Error('Default SAM2 missing');
  await page.locator('.manual-path summary').click();
  await page.locator('#path').fill('/home/pc/gaussian-splatting/output/for_seg1');
  await page.locator('#load').click(); await ready();
  await page.locator('#prompt-mode').check(); await dismissLoad(); await ready();
  if(await page.locator('#text-prompt').isDisabled())throw Error('SAM2 text disabled');
  await page.locator('#text-prompt').fill('yellow rubber duck');
  await dismissLoad();
  await page.locator('#candidates button').first().waitFor({timeout:240000}); await ready();
  const count=await page.locator('#candidates button').count();
  if(count<4 || count%4)throw Error('Missing per-target SAM2 levels');
  if(!(await page.locator('#candidates button').first().innerText()).includes('目标 1'))throw Error('No target labels');
  if((await page.locator('#timing').innerText()).includes('模型加载'))throw Error('Mixed timing');
  // Further prompts reuse the detector without another loading popup.
  await page.locator('#text-prompt').fill('yellow duck'); await automatic(); await ready();
  if(await page.locator('#model-load-dialog').isVisible())throw Error('Detector unnecessarily reloaded');
  fs.mkdirSync('test-results',{recursive:true});
  await page.screenshot({path:'test-results/sam2-text.png'});
  await page.locator('#drone').click(); await ready();
  if(await page.locator('.drone-thumb').count()!==6)throw Error('Missing six-view refinement');
  await page.screenshot({path:'test-results/sam2-text-3d.png'});
  if(errors.length)throw Error(errors.join('\n'));
  console.log(`SAM2 TEXT: ${count} CANDIDATES, DETECTOR LOAD DIALOG/REUSE, SIX VIEWS OK`);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
