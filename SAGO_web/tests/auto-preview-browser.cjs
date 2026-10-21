const {chromium}=require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage({viewport:{width:1920,height:1000}}),errors=[],previews=[];
 let holdNext=false,held=false;
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',req=>{if(req.url().endsWith('/api/jobs') && req.method()==='POST'){const j=req.postDataJSON();if(j.action==='preview'){previews.push(j.data);console.log('preview',j.data.model,j.data.foreground.length,j.data.background.length,j.data.box.length,j.data.text);}}});
 await page.route('**/api/jobs/*',async route=>{
  const response=await route.fetch(),j=await response.json();
  if(holdNext && j.status==='done' && j.result?.masks){holdNext=false;held=true;await new Promise(r=>setTimeout(r,1000));}
  await route.fulfill({response});
 });
 const ready=()=>page.locator('#busy-overlay').waitFor({state:'hidden',timeout:240000});
 const automatic=async()=>{await page.waitForFunction(()=>['done','error'].includes(document.getElementById('preview-status').dataset.state),{},{timeout:240000});assert.equal(await page.locator('#preview-status').getAttribute('data-state'),'done',await page.locator('#status').innerText());await page.locator('#preview-status[data-state="done"]').waitFor({timeout:240000});assert(!(await page.locator('#status').evaluate(e=>e.classList.contains('error'))),await page.locator('#status').innerText());};
 const waitHeld=async()=>{const end=Date.now()+240000;while(!held){if(Date.now()>end)throw Error('No delayed preview');await new Promise(r=>setTimeout(r,20));}};
 try{
  await page.addInitScript(()=>localStorage.setItem('sago-language','zh'));
    await page.goto(process.env.SAGO_TEST_URL || 'http://127.0.0.1:8080');await page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
  assert.equal(await page.locator('#preview').count(),0);
  await page.locator('.manual-path summary').click();await page.locator('#path').fill('/home/pc/gaussian-splatting/output/for_seg1');await page.locator('#load').click();await ready();
  await page.locator('#prompt-mode').check();await page.locator('#model-load-dialog').waitFor({state:'hidden',timeout:240000});await page.locator('#status').filter({hasText:'已进入分割提示'}).waitFor({timeout:240000});
  const canvas=page.locator('#canvas'),r=await canvas.boundingBox();
  const click=(x,y,button='left')=>page.mouse.click(r.x+r.width*x,r.y+r.height*y,{button});
  holdNext=true;await click(.25,.52);await waitHeld();
  assert(await page.locator('#text-prompt').isEnabled());assert(await page.locator('#clear-prompt').isEnabled());assert(await page.locator('#busy-overlay').isHidden());
  await click(.26,.53);await click(.7,.2,'right');assert(await page.locator('#drone').isDisabled());
  await automatic();assert.equal(previews.length,2);assert.equal(previews.at(-1).foreground.length,2);assert.equal(previews.at(-1).background.length,1);assert.equal(await page.locator('#candidates button').count(),4);
  held=false;holdNext=true;await click(.24,.51);await waitHeld();await page.locator('#clear-prompt').click();await page.waitForTimeout(1300);
  assert.equal(await page.locator('#candidates button').count(),0);assert(await page.locator('#drone').isDisabled());assert.equal(await page.locator('#preview-status').getAttribute('data-state'),'idle');
  await page.locator('#box-tool').click();await page.mouse.move(r.x+r.width*.18,r.y+r.height*.36);await page.mouse.down();await page.mouse.move(r.x+r.width*.35,r.y+r.height*.7);await page.mouse.up();await automatic();assert.equal(previews.at(-1).box.length,2);
  await page.locator('#clear-prompt').click();const beforeText=previews.length;
  await page.locator('#text-prompt').pressSequentially('yellow rubber duck',{delay:15});await automatic();assert.equal(previews.length,beforeText+1);assert(await page.locator('#candidates button').count()>=4);
  // IME composition must not submit partially composed text.
  const beforeIme=previews.length;await page.locator('#text-prompt').dispatchEvent('compositionstart');await page.locator('#text-prompt').fill('yellow duck');await page.waitForTimeout(550);assert.equal(previews.length,beforeIme);
  await page.locator('#text-prompt').dispatchEvent('compositionend');await automatic();assert.equal(previews.at(-1).text,'yellow duck');
  for(const model of ['sam3','sam3.1']){
   await page.locator('#seg-model').selectOption(model);await automatic();assert.equal(previews.at(-1).model,model);
   await page.locator('#point-tool').click();await click(.25,.52);await automatic();assert.equal(previews.at(-1).foreground.length,1);
   await page.locator('#clear-prompt').click();
   await page.locator('#text-prompt').fill('yellow rubber duck');await automatic();
  }
  // Exiting prompt mode while inference is in flight must discard the response.
  held=false;holdNext=true;await page.locator('#text-prompt').fill('yellow duck');await waitHeld();await page.locator('#prompt-mode').uncheck();await page.waitForTimeout(1300);
  assert.equal(await page.locator('#candidates button').count(),0);assert(!(await page.locator('#prompt-mode').isChecked()));assert(await page.locator('#drone').isDisabled());
  await page.locator('#prompt-mode').check();await automatic();
  require('node:fs').mkdirSync('test-results',{recursive:true});await page.screenshot({path:'test-results/auto-preview.png'});
  assert.deepEqual(errors,[]);console.log('AUTO PREVIEW: all 3 models, continued points/background/box/text, debounce/IME, stale clear/exit and reentry OK');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
