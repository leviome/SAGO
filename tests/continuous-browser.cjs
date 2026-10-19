// UI workflow against deterministic API responses; engine mask/history tests are in test_continuous.py.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || '@playwright/test');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
(async () => {
    const browser = await chromium.launch({executablePath:'/usr/bin/google-chrome',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    try {
        const page = await browser.newPage({viewport:{width:1600,height:1000},deviceScaleFactor:2});
        const errors=[]; page.on('pageerror',e=>errors.push(e.message));
        const image='data:image/svg+xml;base64,'+Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="white"/></svg>').toString('base64');
        let objects=[],selected=0,jobId=0,result,legacyError=false,rejectLegacyOnce=true,dirty=false,saved=false,restored=false;
        const submissions=[];
        let renderMode='rgb',pending=null;
        const snapshot=()=>({scene:'test',count:100,selected,render_mode:renderMode,objects:objects.map(o=>({...o})),camera:{fov:60,size:768,up:[0,-1,0],target:[0,0,0]},cameras:[],image,canUndo:false,canRedo:false,ids_dirty:dirty,ids_restored:restored,knowledge_path:'/test/SAGO_knowledge.json'});
        await page.route('http://sago.test/**',async route=>{
            const req=route.request(), url=new URL(req.url());
            let data;
            if(url.pathname==='/api/config')data={token:'test',nativePicker:false,scene_loaded:saved};
            else if(url.pathname==='/api/jobs' && req.method()==='POST') {
                const job=req.postDataJSON(); submissions.push(job);
                const {action,data:input}=job;
                if(input.render_mode)renderMode=input.render_mode;
                if(action==='load' || action==='capture')renderMode='rgb';
                if(action==='objects' && input.action==='add_id' && rejectLegacyOnce) {
                    rejectLegacyOnce=false; legacyError=true;
                    return route.fulfill({json:{id:String(++jobId)}});
                }
                if(action==='load_model')result={model_load_seconds:0,reused:true};
                else if(action==='capture')result={...snapshot(),capture:'capture-'+jobId};
                else if(action==='preview')result={masks:[image],model:'sam2',scores:[1]};
                else {
                    if(action==='apply')selected=10;
                    if(action==='load' && saved)restored=true;
                    if(action==='objects' && input.action==='save_ids') { saved=true; dirty=false; }
                    if(action==='objects' && input.action==='rename') {
                        objects.find(o=>o.id===input.id).name=input.name.trim();dirty=true;
                    }
                    if(action==='objects' && input.action==='renumber') {
                        objects.sort((a,b)=>a.number-b.number).forEach((o,i)=>o.number=i+1); dirty=true;
                    }
                    if(action==='objects' && input.action==='merge') {
                        const chosen=objects.filter(o=>input.ids.includes(o.id)).sort((a,b)=>a.number-b.number);
                        chosen[0].count+=chosen[1].count; chosen[0].hidden=false;
                        objects=objects.filter(o=>o.id!==chosen[1].id);
                        selected=chosen[0].count; dirty=true;
                    }
                    if(action==='objects' && input.action==='add_id') {
                        const number=objects.length+1;
                        objects.push({id:'id-'+number,number,name:input.name||'物体 '+number,hidden:true,count:selected});
                        selected=0;
                        dirty=true;
                    }
                    result=snapshot();
                    if(action==='frame' && input.mode==='all')result.frame_id='frame-'+jobId;
                    if(action==='objects' && input.action==='preview_correction') {
                        const picks=input.pixel ? new Set(pending?.picks || []) : new Set(['box1','box2','box3']);
                        if(input.pixel)picks.add(input.pixel.map(Math.floor).join(','));
                        pending={...input,picks,preview_id:'preview-'+jobId};
                        result={...snapshot(),frame_id:'frame-'+jobId,preview_id:pending.preview_id,preview_count:picks.size,target_id:input.target_id};
                    }
                    if(action==='objects' && input.action==='cancel_correction') {
                        pending=null;result={...snapshot(),frame_id:'frame-'+jobId};
                    }
                    if(action==='objects' && input.action==='confirm_correction') {
                        assert.equal(input.preview_id,pending.preview_id);
                        input.target_id=pending.target_id;
                        pending=null;
                        objects[0].count-=3;dirty=true;
                        if(input.target_id>0) {
                            const target=objects.find(o=>o.number===input.target_id);
                            if(target)target.count+=3;
                            else objects.push({id:'id-'+input.target_id,number:input.target_id,name:'物体 '+input.target_id,hidden:false,count:3});
                        }
                        result={...snapshot(),frame_id:'frame-'+jobId,corrected_count:3,target_id:input.target_id};
                    }
                }
                data={id:String(++jobId)};
            } else if(url.pathname.startsWith('/api/jobs/')) {
                data=legacyError?{status:'error',error:'未知物体操作'}:{status:'done',result};
                legacyError=false;
            }
            else {
                const filename=url.pathname==='/'?'index.html':url.pathname.replace('/assets/','');
                const file=path.join(__dirname,'../frontend',filename);
                return route.fulfill({body:fs.readFileSync(file),contentType:filename.endsWith('.js')?'text/javascript':filename.endsWith('.css')?'text/css':'text/html'});
            }
            return route.fulfill({json:data});
        });
        await page.goto('http://sago.test/');
        await page.locator('#status').filter({hasText:'Python backend connected'}).waitFor();
        assert.equal(await page.locator('html').getAttribute('lang'),'en');
        assert.equal(await page.locator('#open-scene').innerText(),'Open scene…');
        await page.locator('#language-toggle').click();
        assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
        await page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
        const scenePanel=page.locator('details[data-panel="scene"]');
        await scenePanel.locator(':scope > summary').click();
        assert(!(await scenePanel.evaluate(el=>el.open)));
        assert(!(await page.locator('#open-scene').isVisible()));
        await page.reload();
        await page.locator('#status').filter({hasText:'后端已连接'}).waitFor();
        assert(!(await scenePanel.evaluate(el=>el.open)));
        await scenePanel.locator(':scope > summary').focus();
        await page.keyboard.press('Enter');
        assert(await page.locator('#open-scene').isVisible());
        for (const key of ['camera','ground','objects','prompts','selection','visualization','timing','export']) {
            const panel=page.locator(`details[data-panel="${key}"]`);
            await panel.locator(':scope > summary').click();
            assert(!(await panel.evaluate(el=>el.open)));
            await panel.locator(':scope > summary').click();
            assert(await panel.evaluate(el=>el.open));
        }
        assert(await page.locator('#add-id').isDisabled());
        assert(await page.locator('#merge-ids').isDisabled());
        await page.locator('.manual-path summary').click();
        await page.locator('#path').fill('/test/scene.ply');
        await page.locator('#load').click();
        await page.locator('#prompt-mode').waitFor({state:'visible'});
        await page.waitForFunction(()=>!document.getElementById('prompt-mode').disabled);
        assert.deepEqual(await page.locator('#render-mode button').allTextContents(),['RGB','Splats','Points','Depth']);
        for(const style of ['splats','points','depth','rgb','points']) {
            await page.locator(`#render-mode button[data-render="${style}"]`).click();
            await page.waitForFunction(()=>!document.getElementById('prompt-mode').disabled);
            assert.equal(await page.locator(`#render-mode button[data-render="${style}"]`).getAttribute('aria-pressed'),'true');
            assert.equal(submissions.filter(j=>j.action==='frame').at(-1).data.render_mode,style);
        }
        await page.locator('#prompt-mode').check();
        await page.locator('#status').filter({hasText:'已进入分割提示'}).waitFor();
        assert.equal(await page.locator('#render-mode button[data-render="rgb"]').getAttribute('aria-pressed'),'true');
        assert(await page.locator('#render-mode button[data-render="splats"]').isDisabled());
        for(let n=1;n<=2;n++) {
            assert(await page.locator('#add-id').isDisabled());
            await page.locator('#canvas').click();
            await page.locator('#preview-status[data-state="done"]').waitFor();
            await page.locator('#point-tool').focus();
            await page.keyboard.press('p');
            await page.locator('#status').filter({hasText:'细化完成'}).waitFor();
            assert(await page.locator('#add-id').isEnabled());
            await page.locator('#object-name').fill('Target '+n);
            // Resetting the operation and text is important before the next object.
            await page.locator('#operation').selectOption('add');
            if(n===1) {
                await page.locator('#add-id').click();
                await page.locator('#status').filter({hasText:'请重启 Python 后端并刷新网页'}).waitFor();
                assert.equal(await page.locator('#objects .object').count(),0);
                assert(await page.locator('#add-id').isEnabled());
            }
            await page.locator('#add-id').click();
            await page.locator('#status').filter({hasText:`已增加 ID ${n}`}).waitFor();
            assert.equal(await page.locator('#objects .object').count(),n);
            assert.equal(await page.locator('#objects .object').last().locator('small').innerText(),`ID ${n} · 10 高斯`);
            assert.equal(await page.locator('#objects .object').last().getByRole('button',{name:'显示',exact:true}).count(),1);
            assert(await page.locator('#prompt-mode').isChecked());
            assert(await page.locator('#add-id').isDisabled());
            assert.equal(await page.locator('#operation').inputValue(),'set');
            assert.equal(await page.locator('#object-name').inputValue(),'');
            assert.equal(await page.locator('#text-prompt').inputValue(),'');
            assert.equal(await page.locator('#candidates button').count(),0);
        }
        assert.equal(submissions.filter(j=>j.action==='objects' && j.data.action==='add_id').length,3);
        assert((await page.locator('#ids-status').innerText()).includes('未保存'));
        await page.getByRole('button',{name:'修改 ID 1 名称',exact:true}).click();
        assert.equal(await page.locator('#rename-name').inputValue(),'Target 1');
        await page.locator('#rename-name').fill('不保存');
        await page.locator('#rename-cancel').click();
        assert.equal(objects[0].name,'Target 1');
        await page.getByRole('button',{name:'修改 ID 1 名称',exact:true}).click();
        await page.locator('#rename-name').fill('红色椅子');
        await page.locator('#rename-name').press('Enter');
        await page.locator('#status').filter({hasText:'物体名称已修改'}).waitFor();
        assert.equal(await page.locator('#objects .object-name-edit').first().innerText(),'红色椅子');
        assert((await page.locator('#merge-first').textContent()).includes('红色椅子'));
        assert((await page.locator('#foreground-list').textContent()).includes('红色椅子'));
        await page.locator('#save-ids').click();
        await page.locator('#status').filter({hasText:'已保存 2 个物体 ID'}).waitFor();
        assert(await page.locator('#prompt-mode').isChecked());
        assert(!(await page.locator('#ids-status').innerText()).includes('未保存'));
        await page.reload();
        await page.locator('#status').filter({hasText:'已恢复当前场景与 2 个物体 ID'}).waitFor();
        assert.equal(await page.locator('#objects .object').count(),2);
        assert(await page.locator('.manual-path').evaluate(el=>el.open));
        await page.locator('#load').click();
        await page.locator('#status').filter({hasText:'恢复 2 个物体 ID'}).waitFor();
        assert.equal(await page.locator('#objects .object').count(),2);
        assert((await page.locator('#ids-status').innerText()).includes('恢复 ID'));
        await page.locator('#display button[data-mode="all"]').click();
        await page.waitForFunction(()=>document.getElementById('busy-overlay').hidden);
        let latest=()=>submissions.filter(j=>j.action==='frame').at(-1).data;
        assert.equal(latest().mode,'all');
        assert.deepEqual(latest().foreground_ids,['id-1','id-2']);
        assert.deepEqual(await page.locator('#display button').allTextContents(),['全部','此轮前景','此轮背景']);
        assert.equal(latest().show_background,true);
        assert.equal(latest().show_foreground,true);
        assert.equal(latest().id_colors,false);
        const ready=()=>page.waitForFunction(()=>!document.getElementById('show-background').disabled);
        await page.locator('#show-background').uncheck();await ready();
        assert.equal(latest().show_background,false);
        assert.equal(latest().show_foreground,true);
        await page.locator('#show-foreground').uncheck();await ready();
        assert.equal(latest().show_foreground,false);
        await page.locator('#show-background').check();await ready();
        assert.equal(latest().show_background,true);
        assert.equal(latest().show_foreground,false);
        await page.locator('#show-foreground').check();await ready();
        await page.locator('#color-ids').check();await ready();
        assert.equal(latest().show_background,true);
        assert.equal(latest().show_foreground,true);
        assert.equal(latest().id_colors,true);
        await page.locator('#display button[data-mode="foreground"]').click();await ready();
        assert.equal(latest().mode,'foreground');
        await page.locator('#display button[data-mode="background"]').click();await ready();
        assert.equal(latest().mode,'background');
        await page.locator('#display button[data-mode="all"]').click();await ready();

        await page.getByRole('checkbox',{name:'前景 ID 2',exact:true}).uncheck();
        await page.locator('#status').filter({hasText:'· 1 个物体'}).waitFor();
        assert.deepEqual(latest().foreground_ids,['id-1']);
        await page.locator('#color-ids').uncheck();
        await page.locator('#status').filter({hasText:'原始颜色'}).waitFor();
        assert.equal(latest().id_colors,false);
        await page.locator('#multi-none').click();
        await page.locator('#status').filter({hasText:'· 0 个物体'}).waitFor();
        assert.deepEqual(latest().foreground_ids,[]);
        await page.locator('#multi-all').click();
        await page.locator('#status').filter({hasText:'· 2 个物体'}).waitFor();
        assert.deepEqual(latest().foreground_ids,['id-1','id-2']);
        await page.locator('.merge-panel summary').click();
        await page.locator('#merge-first').selectOption('id-2');
        await page.locator('#merge-second').selectOption('id-2');
        assert(await page.locator('#merge-ids').isDisabled());
        await page.locator('#merge-second').selectOption('id-1');
        assert(await page.locator('#merge-ids').isEnabled());
        await page.locator('#merge-ids').click();
        await page.locator('#status').filter({hasText:'ID 1、2 已合并为 ID 1'}).waitFor();
        assert.equal(await page.locator('#objects .object').count(),1);
        assert.equal(await page.locator('#objects .object small').innerText(),'ID 1 · 20 高斯');
        assert.equal(await page.locator('#objects .object').getByRole('button',{name:'隐藏',exact:true}).count(),1);
        assert(await page.locator('#merge-ids').isDisabled());
        assert(!(await page.locator('#prompt-mode').isChecked()));
        assert((await page.locator('#ids-status').innerText()).includes('未保存'));
        await page.locator('#save-ids').click();
        await page.locator('#status').filter({hasText:'已保存 1 个物体 ID'}).waitFor();
        objects[0].number=7; // Sparse-ID fixture for the one-click renumber UI.
        await page.reload();
        await page.locator('#status').filter({hasText:'已恢复当前场景与 1 个物体 ID'}).waitFor();
        await page.locator('#renumber-ids').click();
        await page.locator('#status').filter({hasText:'连续编号为 1–1'}).waitFor().catch(async error=>{console.error('Renumber status:',await page.locator('#status').innerText(),'errors:',errors,'recent:',submissions.slice(-5));throw error;});
        assert.equal(await page.locator('#objects .object small').innerText(),'ID 1 · 20 高斯');
        assert((await page.locator('#ids-status').innerText()).includes('未保存'));
        await page.locator('#display button[data-mode="all"]').click();
        await page.waitForFunction(()=>!document.getElementById('background-box').disabled);
        await page.locator('#background-box').click();
        assert.equal(await page.locator('#correction-id').inputValue(),'0');
        assert.equal(await page.locator('#correction-mode').inputValue(),'ellipse');
        assert.equal(await page.locator('#background-box').getAttribute('aria-pressed'),'false');
        await page.locator('#correction-cancel').click();
        await page.locator('#background-box').click();
        await page.locator('#correction-id').press('Enter');
        assert.equal(await page.locator('#background-box').getAttribute('aria-pressed'),'true');
        assert(await page.locator('#prompt-mode').isDisabled());
        const viewJobs=submissions.filter(j=>j.action==='view').length;
        const drawRect=await page.locator('#canvas').boundingBox();
        await page.mouse.click(drawRect.x+drawRect.width*.3,drawRect.y+drawRect.height*.3);
        await page.locator('#status').filter({hasText:'已高亮 1 个可见高斯'}).waitFor();
        assert.equal(await page.locator('#objects .object small').innerText(),'ID 1 · 20 高斯');
        await page.mouse.click(drawRect.x+drawRect.width*.4,drawRect.y+drawRect.height*.4);
        await page.locator('#status').filter({hasText:'已高亮 2 个可见高斯'}).waitFor();
        await page.mouse.click(drawRect.x+drawRect.width*.4,drawRect.y+drawRect.height*.4);
        await page.locator('#status').filter({hasText:'已高亮 2 个可见高斯'}).waitFor();
        await page.waitForFunction(()=>!document.getElementById('correction-discard').disabled);
        const clicks=submissions.filter(j=>j.data.action==='preview_correction' && j.data.pixel);
        assert.equal(clicks.length,3);
        // A square render is stretched into a rectangular CSS viewport at DPR 2.
        // Picking must still use image pixels, independently on each axis.
        for (const [i,expected] of [[0,30],[1,40],[2,40]]) {
            assert(Math.abs(clicks[i].data.pixel[0]-expected)<.15);
            assert(Math.abs(clicks[i].data.pixel[1]-expected)<.15);
        }
        assert(clicks.every(j=>!j.data.rectangle));
        assert.equal(await page.locator('#objects .object small').innerText(),'ID 1 · 20 高斯');
        await page.locator('#correction-discard').click();
        await page.locator('#status').filter({hasText:'已取消选中'}).waitFor();

        await page.mouse.move(drawRect.x+drawRect.width*.7,drawRect.y+drawRect.height*.7);
        await page.mouse.wheel(0,100);
        await page.mouse.down();
        await page.mouse.move(drawRect.x+drawRect.width*.2,drawRect.y+drawRect.height*.2);
        await page.mouse.up();
        await page.locator('#status').filter({hasText:'已高亮 3 个可见高斯'}).waitFor();
        assert.equal(await page.locator('#objects .object small').innerText(),'ID 1 · 20 高斯');
        assert.equal(submissions.filter(j=>j.data.action==='confirm_correction').length,0);
        const callsBeforeSwitch=submissions.length;
        await page.locator('#language-toggle').click();
        assert.equal(await page.locator('html').getAttribute('lang'),'en');
        assert.equal(await page.locator('#correction-confirm').innerText(),'Confirm correction (3)');
        assert((await page.locator('#status').innerText()).includes('Highlighted 3 visible Gaussians'));
        assert(await page.locator('#prompt-mode').isDisabled());
        assert.equal(submissions.length,callsBeforeSwitch);
        await page.locator('#language-toggle').click();
        assert.equal(await page.locator('#correction-confirm').innerText(),'确定修正（3）');

        await page.locator('#correction-discard').click();
        await page.locator('#status').filter({hasText:'已取消选中'}).waitFor();
        assert.equal(await page.locator('#objects .object small').innerText(),'ID 1 · 20 高斯');
        assert(await page.locator('#correction-confirm').isDisabled());
        await page.mouse.move(drawRect.x+drawRect.width*.7,drawRect.y+drawRect.height*.7);
        await page.mouse.down();
        await page.mouse.move(drawRect.x+drawRect.width*.2,drawRect.y+drawRect.height*.2);
        await page.mouse.up();
        await page.locator('#status').filter({hasText:'已高亮 3 个可见高斯'}).waitFor();
        await page.locator('#correction-confirm').click();
        await page.locator('#status').filter({hasText:'已将 3 个可见高斯修正为 ID 0'}).waitFor();
        assert.equal(submissions.filter(j=>j.action==='view').length,viewJobs);
        const correction=submissions.filter(j=>j.action==='objects' && j.data.action==='preview_correction').at(-1).data;
        assert.equal(correction.target_id,0);
        assert.equal(correction.correction_mode,'ellipse');
        assert(correction.frame_id.startsWith('frame-'));
        assert(correction.rectangle[0][0]>correction.rectangle[1][0]);
        for (const [i,expected] of [[0,70],[1,20]]) {
            assert(Math.abs(correction.rectangle[i][0]-expected)<.15);
            assert(Math.abs(correction.rectangle[i][1]-expected)<.15);
        }
        assert.equal(await page.locator('#background-box').getAttribute('aria-pressed'),'true');
        assert.equal(await page.locator('#objects .object small').innerText(),'ID 1 · 17 高斯');
        await page.keyboard.press('Escape');
        await page.waitForFunction(()=>document.getElementById('background-box').getAttribute('aria-pressed')==='false');
        await page.locator('#background-box').click();
        assert.equal(await page.locator('#correction-id').inputValue(),'0');
        assert.equal(await page.locator('#correction-mode').inputValue(),'ellipse');
        await page.locator('#correction-mode').selectOption('point');
        await page.locator('#correction-id').fill('7');
        await page.locator('#correction-id').press('Enter');
        await page.mouse.move(drawRect.x+drawRect.width*.7,drawRect.y+drawRect.height*.7);
        await page.mouse.down();
        await page.mouse.move(drawRect.x+drawRect.width*.2,drawRect.y+drawRect.height*.2);
        await page.mouse.up();
        await page.locator('#status').filter({hasText:'已高亮 3 个可见高斯'}).waitFor();
        assert.equal(await page.locator('#objects .object').count(),1);
        await page.locator('#correction-confirm').click();
        await page.locator('#status').filter({hasText:'已将 3 个可见高斯修正为 ID 7'}).waitFor();
        assert.equal(submissions.filter(j=>j.action==='objects' && j.data.action==='preview_correction').at(-1).data.correction_mode,'point');
        assert.equal(await page.locator('#objects .object').count(),2);
        assert.equal(await page.locator('#objects .object-name-edit').last().textContent(),'物体 7');
        await page.keyboard.press('Escape');
        await page.waitForFunction(()=>document.getElementById('background-box').getAttribute('aria-pressed')==='false');
        await page.locator('#save-ids').click();
        await page.locator('#status').filter({hasText:'已保存 2 个物体 ID'}).waitFor();
        assert.deepEqual(errors,[]);
        console.log('PASS: P segmentation, two consecutive IDs, hidden object controls, fresh prompts, cleared names, replacement mode, button guards; no page errors.');
    } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
