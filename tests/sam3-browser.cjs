const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const fs = require('fs');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1920,height:1000}});
 const errors=[]; page.on('pageerror',e=>errors.push(e.message));
 const routes=[];
 page.on('response',async response=>{
  if(!response.url().includes('/api/jobs/'))return;
  try{const job=await response.json();if(job.status==='done' && job.result?.route?.length===6 && job.result.tracking_stats)routes.push(job.result.tracking_stats);}catch{}
 });
 const automatic=()=>page.locator('#preview-status[data-state="done"]').waitFor({timeout:240000});
 const ready=()=>page.locator('#busy-overlay').waitFor({state:'hidden',timeout:240000});
 const success=async()=>{await ready(); await page.locator('#model-load-dialog').waitFor({state:'hidden',timeout:240000}); const s=await page.locator('#status').innerText();if(await page.locator('#status').evaluate(el=>el.classList.contains('error')))throw Error(s);};
 try{
  await page.addInitScript(()=>localStorage.setItem('sago-language','zh'));
    await page.goto('http://127.0.0.1:8080');
  await page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
  if(await page.locator('#seg-model').inputValue()!=='sam2')throw Error('Default model should be SAM2');
  await page.locator('#seg-model').selectOption('sam3'); await success();
  await page.locator('.manual-path summary').click();
  await page.locator('#path').fill('/home/pc/gaussian-splatting/output/for_seg1');
  await page.locator('#load').click(); await success();
  await page.keyboard.press('m'); await success();
  await page.locator('#text-prompt').fill('yellow rubber duck');
  // M typed into the prompt must not toggle capture mode.
  await page.locator('#text-prompt').press('m');
  if(!await page.locator('#prompt-mode').isChecked())throw Error('M shortcut fired in text input');
  await page.locator('#text-prompt').fill('yellow rubber duck');
  await automatic(); await success();
  if(await page.locator('#candidates button').count()<1)throw Error('No SAM3 semantic instances');
  if(!(await page.locator('#timing').innerText()).includes('SAM3'))throw Error('Wrong model timing');
  fs.mkdirSync('test-results',{recursive:true});
  await page.screenshot({path:'test-results/sam3-text.png'});
  const downloadP=page.waitForEvent('download'); await page.locator('#export-2d').click();
  await(await downloadP).saveAs('test-results/sam3-text-mask.png'); await success();
  await page.locator('#text-prompt').fill('blue toy');
  if(!await page.locator('#drone').isDisabled())throw Error('Stale text candidate still applicable');
  await page.locator('#text-prompt').fill('yellow rubber duck');
  await automatic(); await success();
  await page.locator('#drone').click(); await success();
  if(await page.locator('.drone-thumb').count()!==6)throw Error('Missing SAM3 six-view previews');
  await page.screenshot({path:'test-results/sam3-3d.png'});
  const selectedBefore=await page.locator('#stats').innerText();
  await page.locator('#display button[data-mode="all"]').click(); await success();
  await page.keyboard.press('m'); await success();
  await automatic(); await success();
  await page.locator('#drone').click();
  await page.waitForFunction(()=>Number(document.getElementById('progress').value)>=1,{},{timeout:120000});
  await page.locator('#cancel').click(); await ready();
  if(!(await page.locator('#status').innerText()).includes('取消'))throw Error('SAM3 cancellation missing');
  if(await page.locator('#stats').innerText()!==selectedBefore)throw Error('Cancelled SAM3 route changed selection');
  await page.locator('#prompt-mode').uncheck(); await success();
  await page.locator('#display button[data-mode="all"]').click(); await success();
  await page.keyboard.press('m'); await success();
  await page.locator('#seg-model').selectOption('sam3.1'); await success();
  if(!await page.locator('#apply').isDisabled())throw Error('Stale model candidate still applicable');
  await automatic(); await success();
  if(await page.locator('#candidates button').count()<1)throw Error('No SAM3.1 instances');
  await page.screenshot({path:'test-results/sam31-text.png'});
  await page.locator('#drone').click(); await success();
  if(await page.locator('.drone-thumb').count()!==6)throw Error('Missing SAM3.1 six-view previews');
  await page.locator('#display button[data-mode="all"]').click(); await success();
  await page.keyboard.press('m'); await success();
  await page.locator('#seg-model').selectOption('sam2'); await success();
  if(await page.locator('#text-prompt-group').isHidden())throw Error('SAM2 text prompt hidden');
  await page.locator('#text-prompt').fill('');
  await page.locator('#box-tool').click();
  const r=await page.locator('#canvas').boundingBox();
  await page.mouse.move(r.x+r.width*.25,r.y+r.height*.2);await page.mouse.down();
  await page.mouse.move(r.x+r.width*.75,r.y+r.height*.8);await page.mouse.up();
  await automatic();await success();
  if(await page.locator('#candidates button').count()!==4)throw Error('SAM2 fallback failed');
  if(routes.length!==2)throw Error(`Expected two completed semantic routes, got ${routes.length}`);
  for(const route of routes){
   if(route.seed_runs!==1 || route.new_frames!==6 || route.image_encodes!==7)throw Error(`Unexpected incremental work: ${JSON.stringify(route)}`);
  }
  if(errors.length)throw Error(errors.join('\n'));
  console.log('SAM3 + SAM3.1 INCREMENTAL SIX VIEWS (7 IMAGE ENCODES), CANCEL ROLLBACK, TEXT, SAM2 SWITCH OK');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
