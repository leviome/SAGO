const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const fs = require('fs');
(async()=>{
    const browser=await chromium.launch({executablePath:process.env.SAGO_CHROME || '/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    try {
        const page=await browser.newPage({viewport:{width:1500,height:1000}});
        const errors=[];page.on('pageerror',e=>errors.push(e.message));
        let native=false;
        await page.route('**/api/config',async route=>{const response=await route.fetch();const data=await response.json();await route.fulfill({json:{...data,nativePicker:native}});});
        const ready=()=>page.locator('#busy-overlay').waitFor({state:'hidden',timeout:120000});
        const connected=()=>page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
        const n=600;
        const props=['x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];
        const header=Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${n}\n${props.map(x=>'property float '+x).join('\n')}\nend_header\n`);
        const body=Buffer.alloc(n*props.length*4);
        for(let i=0;i<n;i++){
            const angle=i*2.399963, y=1-2*i/n, r=Math.sqrt(1-y*y);
            const values=[r*Math.cos(angle),y,r*Math.sin(angle),3,1,-.6,-.6,-3,-3,-3,1,0,0,0];
            values.forEach((v,j)=>body.writeFloatLE(v,(i*props.length+j)*4));
        }
        const buffer=Buffer.concat([header,body]);
        await page.addInitScript(()=>localStorage.setItem('sago-language','zh'));
    await page.goto('http://127.0.0.1:8080');await connected();
        if(await page.locator('.manual-path').evaluate(el=>el.open))throw new Error('Manual path should be collapsed');
        const choose=page.waitForEvent('filechooser');await page.locator('#open-scene').click();
        await(await choose).setFiles({name:'test-sphere.ply',mimeType:'application/octet-stream',buffer});
        await page.waitForFunction(()=>document.getElementById('scene-info').textContent.includes('600'));
        await ready();
        if(await page.locator('.recent-open').count()!==1)throw new Error('Successful upload missing from recents');
        const originalPath=await page.locator('#path').inputValue();
        await page.reload();await connected();
        if(await page.locator('.recent-open').count()!==1)throw new Error('Recent scenes did not persist');
        await page.locator('.recent-open').first().click();await ready();
        if(!(await page.locator('#scene-info').innerText()).includes('600'))throw new Error('Recent scene did not reopen');
        const transfer=await page.evaluateHandle(bytes=>{const d=new DataTransfer();d.items.add(new File([new Uint8Array(bytes)],'test-sphere.ply',{type:'application/octet-stream'}));return d;},Array.from(buffer));
        await page.locator('#viewport').dispatchEvent('dragenter',{dataTransfer:transfer});
        if(await page.locator('#drop-hint').isHidden())throw new Error('Drop hint missing');
        const uploaded=page.waitForResponse(r=>r.url().endsWith('/api/scenes/upload'));
        await page.locator('#viewport').dispatchEvent('drop',{dataTransfer:transfer});
        await uploaded;await ready();
        if(await page.locator('#path').inputValue()!==originalPath)throw new Error('Identical drop was not reused');
        if(await page.locator('.recent-open').count()!==1)throw new Error('Duplicate recent record');
        // Exercise native-picker wiring without opening an unattended desktop dialog.
        native=true;
        await page.route('**/api/jobs',async route=>{
            const data=route.request().postDataJSON();
            if(data.action==='pick')await route.fulfill({status:202,json:{id:'picker-test'}});else await route.continue();
        });
        await page.route('**/api/jobs/picker-test',route=>route.fulfill({json:{status:'done',result:{cancelled:true}}}));
        await page.reload();await connected();
        await page.locator('#open-scene').click();await ready();
        if(!(await page.locator('#status').innerText()).includes('已取消选择'))throw new Error('Native cancel handling failed');
        if(await page.locator('.recent-open').count()!==1)throw new Error('Cancel changed recents');
        await page.locator('.recent-open').click();await ready();
        fs.mkdirSync('test-results',{recursive:true});await page.screenshot({path:'test-results/scene-inputs.png'});
        await page.locator('.recent-remove').click();
        if(await page.locator('.recent-open').count()!==0)throw new Error('Remove recent failed');
        if(errors.length)throw new Error(errors.join('\n'));
        console.log('PASS: browser picker upload, real CUDA scene load, recent persistence/reopen/removal, drag/drop, deduplication, native picker cancellation; no page errors.');
    } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
