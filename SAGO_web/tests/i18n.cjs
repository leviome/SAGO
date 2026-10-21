const assert=require('assert/strict');
const fs=require('fs');
const uri=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
(async()=>{
 const catalogSource=fs.readFileSync('frontend/translations.js','utf8');
 const catalogURL=uri(catalogSource);
 const catalog=await import(catalogURL);
 const source=fs.readFileSync('frontend/i18n.js','utf8').replace("'/assets/translations.js'",JSON.stringify(catalogURL));
 const {t,backendText,getLanguage,setLanguage}=await import(uri(source));
 assert.equal(getLanguage(),'en');
 assert.equal(t('修正'),'Correct');
 assert.equal(t('确定修正（{0}）',32),'Confirm correction (32)');
 assert.equal(backendText('画面已变化，请重新进入修正'),'The view changed. Enter correction mode again.');
 assert.equal(backendText('完成 3/6，剩余 120 个高斯'),'Completed 3/6; 120 Gaussians remaining');
 assert.equal(backendText('无法读取地面知识 /tmp/场景：permission denied'),'Cannot read ground knowledge /tmp/场景: permission denied');
 assert.equal(backendText('Target custom name'),'Target custom name');
 const placeholders=text=>[...new Set(text.match(/\{\d+\}/g)||[])].sort();
 for(const [key,value] of Object.entries(catalog.messages)) {
  assert(!/[\u3400-\u9fff]/u.test(value),'Untranslated: '+key);
  assert.deepEqual(placeholders(key),placeholders(value),'Placeholders: '+key);
 }
 const html=fs.readFileSync('frontend/index.html','utf8');
 for(const [,key] of html.matchAll(/data-i18n(?:-[\w-]+)?="([^"]+)"/g)) {
  assert(catalog.messages[key.replaceAll('&amp;','&').replaceAll('&quot;','"')]!=null,'Missing HTML translation: '+key);
 }
 global.document={documentElement:{},querySelectorAll:()=>[],getElementById:()=>({setAttribute(){}})};
 setLanguage('zh');
 assert.equal(t('修正'),'修正');
 assert.equal(backendText('The current selection is empty'),'当前选区为空');
 assert.equal(backendText('Completed 3/6; 120 Gaussians remaining'),'完成 3/6，剩余 120 个高斯');
 setLanguage('en');
 console.log('PASS: English default, UI catalog coverage, backend messages, placeholders, and untouched user data.');
})().catch(error=>{console.error(error);process.exitCode=1});
