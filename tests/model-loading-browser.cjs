const {chromium}=require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
(async()=>{
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
 const page=await browser.newPage(); const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 try {
  await page.addInitScript(()=>localStorage.setItem('sago-language','zh'));
    await page.goto('http://127.0.0.1:8080');
  await page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
  for(const model of ['sam3','sam3.1','sam2']) {
   await page.locator('#seg-model').selectOption(model);
   await page.locator('#model-load-dialog').waitFor();
   await page.locator('#model-load-spinner').waitFor();
   await page.keyboard.press('Escape');
   if(!await page.locator('#model-load-dialog').isVisible())throw Error('Loading dialog dismissed early');
   await page.locator('#model-load-spinner').waitFor({state:'hidden',timeout:240000});
   const message=await page.locator('#model-load-message').innerText();
   const time=await page.locator('#model-load-time').innerText();
   if(message!=='模型加载完成' || !/模型加载用时：\d+\.\d{3} 秒/.test(time))throw Error(`${model}: ${message}: ${time}`);
   console.log(model,time);
   if((await page.locator('#timing').innerText()).includes('模型加载'))throw Error('Load time mixed with inference');
   await page.locator('#model-load-dialog').waitFor({state:'hidden',timeout:10000});
   if(await page.locator('#seg-model').isDisabled())throw Error('Controls not restored');
  }
  if(errors.length)throw Error(errors.join('\n'));
  console.log('MODEL SWITCH DIALOGS AND SEPARATE TIMING OK');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
