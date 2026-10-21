import { t, localized, messageText, backendText, initLanguage } from '/assets/i18n.js';
import { drawVisualization } from '/assets/visualization.js';
import { LatestPreview } from '/assets/auto-preview.js';
const $ = id => document.getElementById(id);
// Native details remain keyboard accessible and keep controls mounted when folded.
const panels = [...document.querySelectorAll('aside details[data-panel]')];
try {
    const saved = JSON.parse(localStorage.getItem('sago-panels') || '{}');
    for (const panel of panels) {
        if (typeof saved?.[panel.dataset.panel] === 'boolean') panel.open = saved[panel.dataset.panel];
    }
} catch {}
for (const panel of panels) panel.addEventListener('toggle', () => {
    try {
        localStorage.setItem('sago-panels', JSON.stringify(Object.fromEntries(panels.map(p => [p.dataset.panel, p.open]))));
    } catch {}
});
const canvas = $('canvas'), ctx = canvas.getContext('2d');
let gifExportVersion = 0;
let token, state = null, capture = null, busy = false, currentJob = null;
let mode = 'all', tool = 'point', level = 0, masks = [], fg = [], bg = [], box = [];
let renderMode = 'rgb';
let visualization = null;
let renameObjectId = null;
const foregroundIds = new Set(), knownObjects = new Set();
let baseImage = null, maskImage = null, drag = null, wheelTimer;
let resizeTimer;
const viewportAspect = () => { const r=$('viewport').getBoundingClientRect(); return Math.max(.2,Math.min(5,r.width/Math.max(1,r.height))); };
let samSeconds = null;
let groundSession = null;
let backgroundBox = false, backgroundRect = [];
let correctionId = 0;
let correctionMode = 'ellipse';
let correctionPreview = null;
let gifItems = [];
let loadedModel = null;
let groundingLoaded = false;
let previewBusy = false, composingPrompt = false;
const autoPreview = new LatestPreview({
    read: promptSnapshot,
    ready: () => !busy && !drag && !composingPrompt && !$('model-load-dialog').open,
    run: automaticPreview,
    apply: showCandidates,
    onError: error => { $('preview-status').dataset.state='error'; localized($('preview-status'), "textContent", () => t("分割失败，请调整提示重试")); status(() => error.message || String(error), true); }
});
let timingModel = 'SAM2';
const selectedModel = () => $('seg-model').value;
let nativePicker = false;
let recentScenes = [];
try {
    const saved = JSON.parse(localStorage.getItem('sago-recent-scenes') || '[]');
    if (Array.isArray(saved)) recentScenes = saved.filter(x=>x && typeof x.path==='string' && typeof x.name==='string').slice(0, 8);
} catch {}
function showTiming(timing) {
    if (!timing) return;
    timingModel = (timing.model || selectedModel()).toUpperCase();
    if (timing.sam_seconds != null) {
        samSeconds=timing.sam_seconds;
        localized($('timing'), "textContent", () => t("{0} 分割：{1} 秒", timingModel, samSeconds.toFixed(3)));
    } else if (timing.refine_seconds != null) {
        localized($('timing'), "textContent", () => t("{0}：{1} 秒", timingModel, (samSeconds || 0).toFixed(3)));
        const line=document.createElement('span'); localized(line, "textContent", () => t("三维筛选 / 细化：{0} 秒", timing.refine_seconds.toFixed(3)));
        const total=document.createElement('strong'); localized(total, "textContent", () => t("分割合计：{0} 秒", timing.total_seconds.toFixed(3)));
        $('timing').append(line,total);
    }
}
const loadImage = src => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src; });
function status(message, error = false) { localized($('status'), 'textContent', () => backendText(typeof message==='function' ? message() : message)); $('status').classList.toggle('error', error); }
async function json(url, options) {
    const response = await fetch(url, { ...options, headers: { 'Content-Type': 'application/json', 'X-SAGO-Token': token || '', ...options?.headers } });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}
function controls() {
    document.querySelectorAll('button,input,select').forEach(el => { if (!['cancel', 'help', 'close-help', 'close-model-load','language-toggle'].includes(el.id)) el.disabled = busy; });
    if (busy) {
        if (previewBusy) {
            for (const id of ['prompt-mode','text-prompt','clear-prompt','point-tool','box-tool']) $(id).disabled = !capture;
            $('prompt-mode').checked=!!capture;
            $('view-hint').hidden=!capture;
            canvas.classList.toggle('frozen',!!capture);
            localized($('view-label'), "textContent", () => capture ? t("提示画面已固定") : t("透视视图"));
        }
        return;
    }
    const hasScene = !!state?.count;
    for(const button of document.querySelectorAll('#render-mode button')) {
        const active=button.dataset.render===renderMode;
        button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));
        button.disabled=!hasScene || !!capture || !!groundSession || backgroundBox;
    }
    for (const id of ['background-color','background-r','background-g','background-b']) $(id).disabled = !hasScene || !!capture || !!groundSession || backgroundBox;
    $('adversarial-background').disabled = !hasScene || !!groundSession || backgroundBox;
    $('save-ids').disabled = !hasScene;
    const hasObjects=hasScene && state.objects.length>0;
    const canCorrect=hasScene && ($('show-background').checked || ($('show-foreground').checked && state.objects.some(o=>foregroundIds.has(o.id) && !o.hidden)));
    if (mode!=='all' || !canCorrect || capture || groundSession) { backgroundBox=false; backgroundRect=[]; }
    $('background-box').hidden = mode!=='all';
    $('background-box').disabled = !canCorrect || !!capture || !!groundSession || !state.frame_id;
    localized($('background-box'), "textContent", () => backgroundBox ? t("退出修正") : t("修正"));
    $('background-box').setAttribute('aria-pressed',String(backgroundBox));
    $('correction-confirm').hidden = $('correction-discard').hidden = !backgroundBox;
    $('correction-confirm').disabled = !correctionPreview?.preview_count;
    $('correction-discard').disabled = !correctionPreview;
    localized($('correction-confirm'), "textContent", () => correctionPreview?.preview_count ? t("确定修正（{0}）", correctionPreview.preview_count.toLocaleString()) : t("确定修正"));
    $('renumber-ids').disabled = !hasObjects;
    for (const id of ['show-background','show-foreground']) $(id).disabled = !hasScene || !!capture || !!groundSession;
    for (const id of ['multi-all','multi-none','color-ids']) $(id).disabled = !hasObjects || !!capture;
    document.querySelectorAll('#foreground-list input').forEach(input=>input.disabled = !hasObjects || !!capture);
    for (const id of ['merge-first','merge-second']) $(id).disabled = !hasScene || state.objects.length < 2;
    $('merge-ids').disabled = !hasScene || state.objects.length < 2 || !$('merge-first').value || !$('merge-second').value || $('merge-first').value === $('merge-second').value;
    localized($('ids-status'), "textContent", () => !hasScene ? t("0 为背景，正整数为物体 ID") : state.ids_dirty ? t("ID 分割结果有未保存的修改") : state.ids_restored ? t("已从 SAGO_knowledge.json 恢复 ID") : t("保存到当前场景的 SAGO_knowledge.json"));
    $('ids-status').title = state?.knowledge_path || '';
    for (const id of ['prompt-mode','camera-apply','save-object','export-image','clear']) $(id).disabled = !hasScene;
    for (const id of ['clear-prompt','point-tool','box-tool']) $(id).disabled = !capture;
    for (const id of ['drone','export-2d']) $(id).disabled = !capture || !masks.length;
    for (const id of ['save-object','add-id','export-ply','export-mask']) $(id).disabled = !state?.selected;
    $('export-gif').disabled = !hasScene || !state?.selected;
    $('export-all-gif').disabled = !hasScene || !state.objects.length;
    $('add-id').disabled ||= !!capture;
    $('undo').disabled = !state?.canUndo; $('redo').disabled = !state?.canRedo;
    $('prompt-mode').checked = !!capture;
    $('text-prompt-group').hidden = false;
    $('text-prompt').disabled = !capture;
    $('view-hint').hidden = !capture && !groundSession && !backgroundBox;
    localized($('view-hint'), "textContent", () => backgroundBox ? t("{0} → ID {1} · 点击累加 / 拖框替换 · 仅可见高斯 · 高亮后点击确定修正 · Esc 退出", correctionMode==='ellipse'?t("椭圆修正"):t("点修正"), correctionId) : groundSession ? t("地面标定 · 左键依次选择三个地面点") : t("左键前景 · 右键背景 · 框选拖动"));
    canvas.classList.toggle('frozen', !!capture || !!groundSession || backgroundBox);
    localized($('view-label'), "textContent", () => backgroundBox ? `${correctionMode==='ellipse'?t("椭圆修正"):t("点修正")} → ID ${correctionId}` : groundSession ? t("地面标定画面已固定") : capture ? t("提示画面已固定") : t("透视视图"));
    $('ground-start').disabled = !hasScene;
    $('ground-start').hidden = !!groundSession;
    $('ground-edit').hidden = !groundSession;
    const plane = groundSession?.plane || (!groundSession && state?.ground);
    $('ground-flip').disabled = !plane || !!capture;
    $('ground-apply').disabled = !groundSession?.plane;
    $('ground-undo').disabled = !groundSession?.points.length;
    $('ground-reset').disabled = !groundSession?.points.length;
    localized($('ground-info'), "textContent", () => groundSession ? t("已选 {0}/3 点{1}", groundSession.points.length, plane ? t(" · 检查方向后应用地面") : t(" · 请选择分散且不共线的地面点")) : plane ? (state?.ground_restored ? t("已从 SAGO_knowledge.json 读取地面与向上方向") : t("已保存地面 · 相机与三维巡航使用地面向上方向")) : backendText(state?.knowledge_warning) || t("在同一平面上选三个分散且不共线的点。"));
    $('ground-info').title = state?.knowledge_path || '';
    $('ground-details').hidden = !plane;
    if (plane) {
        const n=plane.normal.map(x=>x.toFixed(5));
        localized($('ground-equation'), "textContent", () => t("向上：[{0}]；地面：{1}x + {2}y + {3}z + ({4}) = 0", n.join(', '), n[0], n[1], n[2], plane.offset.toFixed(5)));
        $('ground-coordinates').replaceChildren(...plane.points.map((p,i)=> {const row=document.createElement('div'); row.textContent=`P${i+1}：${p.map(x=>x.toFixed(4)).join(', ')}`;return row;}));
    }
    if (groundSession) {
        const allowed=['language-toggle','ground-undo','ground-reset','ground-apply','ground-cancel','ground-flip','help','close-help'];
        document.querySelectorAll('button,input,select').forEach(el=> { if (!allowed.includes(el.id)) el.disabled=true; });
    }
    if (backgroundBox) {
        const allowed=['language-toggle','background-box','correction-confirm','correction-discard','help','close-help'];
        document.querySelectorAll('button,input,select').forEach(el=> { if (!allowed.includes(el.id)) el.disabled=true; });
    }
}
function clearPrompts() { fg = []; bg = []; box = []; invalidateCandidates(); }
function release() { capture = null; groundSession = null; drag=null; clearPrompts(); controls(); }
async function present(result) {
    if (result.camera) {
        correctionPreview = result.preview_id ? {preview_id:result.preview_id,preview_count:result.preview_count} : null;
        state = result;
        if(result.render_mode)renderMode=result.render_mode;
        if(result.background_color) setBackgroundColor(result.background_color);
        if(result.display) {
            $('show-background').checked=result.display.show_background;
            $('show-foreground').checked=result.display.show_foreground;
            $('color-ids').checked=result.display.id_colors;
            if(result.display.foreground_ids!==null) {
                foregroundIds.clear();
                result.display.foreground_ids.forEach(id=>foregroundIds.add(id));
                result.objects.forEach(obj=>knownObjects.add(obj.id));
            }
        }
        const v = state.camera;
        $('preset-label').hidden = !state.cameras?.length;
        if ($('camera-preset').options.length !== state.cameras?.length) {
            $('camera-preset').replaceChildren();
            (state.cameras || []).forEach((name,i)=> { const o=document.createElement('option'); o.value=i; o.textContent=`${i} · ${name}`; $('camera-preset').append(o); });
        }
        $('fov').value = v.fov; $('size').value = v.size;
        ['x','y','z'].forEach((axis, i) => { $(`up-${axis}`).value = v.up[i]; $(`target-${axis}`).value = Number(v.target[i].toFixed(4)); });
        localized($('scene-info'), "textContent", () => t("{0} · {1} 高斯", state.scene, state.count.toLocaleString()));
        localized($('stats'), "textContent", () => t("{0} / {1} 已选", state.selected.toLocaleString(), state.count.toLocaleString()));
        renderObjects();
    }
    if (result.image) {
        visualization = result.visualization || null;
        baseImage = await loadImage(result.image);
        canvas.width = baseImage.width; canvas.height = baseImage.height;
        $('empty').hidden = true; canvas.hidden = false; maskImage = null;
        draw();
    }
    controls();
}
let droneViews = [];
function clearDroneViews() {
    droneViews=[]; $('drone-views').replaceChildren(); $('drone-gallery-section').hidden=true;
    $('drone-view-count').textContent='0/6'; $('drone-gallery-status').textContent='';
    $('drone-live').hidden=true; $('drone-live').removeAttribute('src');
}
function showDroneViews(views) {
    if (!views?.length || views.length===droneViews.length) return;
    droneViews=views; $('drone-gallery-section').hidden=false;
    $('drone-view-count').textContent=`${views.length}/6`; $('drone-views').replaceChildren();
    for (const view of views) {
        const button=document.createElement('button');button.className='drone-thumb';button.disabled=busy;
        const image=document.createElement('img');image.src=view.overlay;localized(image, "alt", () => t("V-Drone {0} {1}° 分割图", view.drone, view.yaw));
        const name=document.createElement('span');name.textContent=`D${view.drone} · ${view.yaw>0?'+':''}${view.yaw}°`;
        const info=document.createElement('small');localized(info, "textContent", () => view.lost?t("目标丢失 / mask 为空"):t("{0} 高斯 · {1}", view.count.toLocaleString(), view.changed?t("已收紧"):t("选区未变")));
        button.append(image,name,info);
        button.onclick=()=>{
            $('drone-view-title').textContent=`V-Drone ${view.drone} · ${view.yaw>0?'+':''}${view.yaw}°`;
            $('drone-render').src=view.render; $('drone-overlay').src=view.overlay;
            $('drone-render-download').href=view.render; $('drone-overlay-download').href=view.overlay; $('drone-mask-download').href=view.mask;
            $('drone-view-info').textContent=info.textContent;
            $('drone-view-dialog').showModal();
        };
        $('drone-views').append(button);
    }
    $('drone-live').src=views.at(-1).overlay; $('drone-live').hidden=!busy;
}
$('close-drone-view').onclick=()=> $('drone-view-dialog').close();
async function job(action, data = {}, interactive = false) {
    if (busy) throw new Error(t("请等待当前任务完成"));
    if (['load','view','frame','capture'].includes(action) || (action==='ground' && data.operation==='start')) data={...data,aspect:viewportAspect()};
    if (state && ['frame','view','export'].includes(action)) {
        data={...data,render_mode:renderMode,id_colors:$('color-ids').checked,show_background:$('show-background').checked,show_foreground:$('show-foreground').checked};
        data.foreground_ids=[...foregroundIds];
    }
    data.background_color = ['r','g','b'].map(c=>Number($(`background-${c}`).value));
    const started=performance.now();
    $('busy-elapsed').textContent='';
    const elapsed=setInterval(()=> { localized($('busy-elapsed'), "textContent", () => t("已用时 {0} 秒", ((performance.now()-started)/1000).toFixed(1))); },100);
    busy = true; previewBusy=interactive; controls(); $('busy-overlay').hidden = interactive;
    $('drone-live').hidden=true;
    if (action==='apply' && data.drone) { clearDroneViews(); $('drone-gallery-section').hidden=false; localized($('drone-gallery-status'), "textContent", () => t("正在生成各视角…")); }
    $('cancel').hidden = !(action === 'apply' && data.drone) && action!=='gif';
    localized($('cancel'), 'textContent', () => action==='gif' ? t("取消生成") : t("取消巡航"));
    $('progress').max = action==='gif' ? 100 : 6;
    $('progress').value = 0;
    localized($('busy-message'), "textContent", () => action === 'pick' ? t("请在系统文件选择框中选择场景…") : action === 'preview' ? t("{0} 编码与分割…", selectedModel().toUpperCase()) : t("Python 后端处理中…"));
    try {
        const submitted = await json('/api/jobs', { method: 'POST', body: JSON.stringify({ action, data }) });
        currentJob = submitted.id;
        for (;;) {
            const result = await json(`/api/jobs/${currentJob}`);
            localized($('busy-message'), "textContent", () => backendText(result.message) || t("处理中…"));
            $('progress').value = result.done || 0;
            if (result.views) showDroneViews(result.views);
            if (result.status === 'done') {
                if (action==='apply' && data.drone) { showDroneViews(result.result.route); localized($('drone-gallery-status'), "textContent", () => t("巡航完成 · 点击缩略图查看渲染与分割对照")); }
                if (!interactive) { showTiming(result.result.timing); status(() => t("完成")); }
                return result.result;
            }
            if (result.status === 'error' || result.status === 'cancelled') {
                if (action==='apply' && data.drone) localized($('drone-gallery-status'), "textContent", () => t("本次未提交选区 · 保留已完成视角供查看"));
                if (action==='objects' && ['add_id','save_ids','merge','renumber','background_box','correct_box'].includes(data.action) && result.error==='未知物体操作') {
                    const label={add_id:t("增加ID"),save_ids:t("保存分割结果"),merge:t("合并ID"),renumber:t("ID连续编号"),background_box:t("修正"),correct_box:t("修正")}[data.action];
                    throw new Error(t("当前运行的后端尚不支持“{0}”，请重启 Python 后端并刷新网页", label));
                }
                throw new Error(result.error);
            }
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    } finally {
        clearInterval(elapsed);
        currentJob = null; busy = false; previewBusy=false; $('busy-overlay').hidden = true; $('drone-live').hidden=true; controls(); autoPreview.resume();
    }
}
const run = fn => async (...args) => { try { await fn(...args); } catch (error) { status(() => error.message || String(error), true); } };
function bind(id, fn) { $(id).onclick = run(fn); }
async function loadModel(text = false) {
    const model=selectedModel(), dialog=$('model-load-dialog');
    localized($('model-load-title'), "textContent", () => text ? t("SAM2 文本定位模型加载") : t("{0} 模型加载", model.toUpperCase()));
    localized($('model-load-message'), "textContent", () => t("正在加载，请稍候…"));
    $('model-load-spinner').hidden=false;
    localized($('model-load-time'), "textContent", () => t("已用时 0.0 秒"));
    $('close-model-load').hidden=true;
    if (!dialog.open) dialog.showModal();
    const started=performance.now();
    const timer=setInterval(()=> { localized($('model-load-time'), "textContent", () => t("已用时 {0} 秒", ((performance.now()-started)/1000).toFixed(1))); },100);
    try {
        const result=await job('load_model', {model, text});
        clearInterval(timer);
        if (loadedModel !== model) groundingLoaded=false;
        loadedModel=model;
        if (text) groundingLoaded=true;
        localized($('model-load-message'), "textContent", () => result.reused ? t("模型已加载，可直接使用") : t("模型加载完成"));
        localized($('model-load-time'), "textContent", () => t("模型加载用时：{0} 秒{1}", result.model_load_seconds.toFixed(3), result.reused ? t("（复用已加载模型）") : ''));
        $('model-load-spinner').hidden=true;
        await new Promise(resolve => setTimeout(resolve, 1500));
        dialog.close();
    } catch (error) {
        clearInterval(timer);
        loadedModel=null; groundingLoaded=false;
        localized($('model-load-message'), "textContent", () => t("模型加载失败"));
        localized($('model-load-time'),'textContent',()=>backendText(error.message || String(error)));
        throw error;
    } finally {
        clearInterval(timer);
        $('model-load-spinner').hidden=true;
        $('close-model-load').hidden=!dialog.open;
        autoPreview.resume();
    }
}
bind('close-model-load', () => $('model-load-dialog').close());
$('model-load-dialog').addEventListener('cancel', event => { if (busy) event.preventDefault(); });
function draw() {
    if (!baseImage) return;
    canvas.style.objectFit = capture || groundSession ? 'contain' : 'fill';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(maskImage || baseImage, 0, 0);
    if (groundSession) {
        const points=groundSession.pixels;
        ctx.lineWidth=Math.max(2,canvas.width/350);ctx.strokeStyle='#ffc86b';
        if(points.length>1) {
            ctx.beginPath();ctx.moveTo(...points[0]);points.slice(1).forEach(p=>ctx.lineTo(...p));
            if(points.length===3){ctx.closePath();ctx.fillStyle='#ffc86b25';ctx.fill();}ctx.stroke();
        }
        points.forEach((p,i)=> {ctx.beginPath();ctx.arc(...p,6,0,Math.PI*2);ctx.fillStyle='#ffc86b';ctx.fill();ctx.strokeStyle='#111820';ctx.stroke();ctx.font=`bold ${Math.max(14,canvas.width/55)}px sans-serif`;ctx.fillText(String(i+1),p[0]+10,p[1]-10);});
        if (groundSession.arrow) {
            const [a,b]=groundSession.arrow, angle=Math.atan2(b[1]-a[1],b[0]-a[0]), head=Math.max(8,canvas.width/70);
            ctx.strokeStyle='#ff4058';ctx.fillStyle='#ff4058';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
            ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-head*Math.cos(angle-.45),b[1]-head*Math.sin(angle-.45));ctx.lineTo(b[0]-head*Math.cos(angle+.45),b[1]-head*Math.sin(angle+.45));ctx.closePath();ctx.fill();ctx.fillText(t("向上"),b[0]+8,b[1]-8);
        }
        return;
    }
    drawVisualization(ctx, visualization, canvas.width, canvas.height, $('show-bbox').checked, $('show-route').checked);
    if (backgroundRect.length===2) {
        ctx.save();ctx.strokeStyle='#ffc857';ctx.fillStyle='#ffc85722';ctx.lineWidth=Math.max(2,canvas.width/350);ctx.setLineDash([7,4]);
        const [a,b]=backgroundRect;ctx.fillRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);ctx.strokeRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);ctx.restore();
    }
    if (!capture) return;
    ctx.lineWidth = Math.max(2, canvas.width / 350);
    for (const [points, color] of [[fg, '#68f4b3'], [bg, '#ff7777']]) {
        for (const [x,y] of points) { ctx.beginPath(); ctx.arc(x,y,5,0,Math.PI*2); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = '#101820'; ctx.stroke(); }
    }
    if (box.length === 2) { ctx.strokeStyle = '#ff4058'; ctx.setLineDash([6,4]); ctx.strokeRect(box[0][0],box[0][1],box[1][0]-box[0][0],box[1][1]-box[0][1]); ctx.setLineDash([]); }
}
for (const id of ['show-bbox','show-route']) $(id).onchange=draw;
function renderObjects() {
    $('object-count').textContent = state.objects.length;
    $('objects').replaceChildren();
    const existing=new Set(state.objects.map(o=>o.id));
    for (const id of foregroundIds) if (!existing.has(id)) foregroundIds.delete(id);
    for (const id of knownObjects) if (!existing.has(id)) knownObjects.delete(id);
    $('foreground-list').replaceChildren();
    for (const obj of state.objects) {
        if (!knownObjects.has(obj.id)) { knownObjects.add(obj.id); foregroundIds.add(obj.id); }
        const label=document.createElement('label'); label.className='foreground-item';
        const input=document.createElement('input'); input.type='checkbox'; input.checked=foregroundIds.has(obj.id);
        input.dataset.id=obj.id; localized(input, "aria-label", () => t("前景 ID {0}", obj.number), true);
        input.onchange=run(async()=>{ if(input.checked)foregroundIds.add(obj.id);else foregroundIds.delete(obj.id); await showMultiForeground(); });
        const swatch=document.createElement('i'); swatch.className='id-swatch'; swatch.style.backgroundColor=obj.color || '#ff4058';
        const text=document.createElement('span');text.textContent=`ID ${obj.number} · ${obj.name}`;text.title=text.textContent;
        label.append(input,swatch,text);$('foreground-list').append(label);
    }
    for (const obj of state.objects) {
        const row = document.createElement('div'); row.className = 'object';
        const name = document.createElement('div'); name.className = 'name';
        const edit = document.createElement('button'); edit.className='object-name-edit'; edit.textContent=obj.name;
        localized(edit, "title", () => t("修改物体名称")); localized(edit, "aria-label", () => t("修改 ID {0} 名称", obj.number), true);
        edit.onclick=()=>{
            renameObjectId=obj.id; $('rename-name').value=obj.name;
            localized($('rename-title'), "textContent", () => t("修改 ID {0} 名称", obj.number));
            $('rename-dialog').showModal(); $('rename-name').focus(); $('rename-name').select();
        };
        name.append(edit);
        const count = document.createElement('small'); localized(count, "textContent", () => t("ID {0} · {1} 高斯", obj.number, obj.count.toLocaleString())); name.append(count); row.append(name);
        for (const [action, label, title] of [['select',t("选择"),t("设为当前选区")],['hide', obj.hidden ? t("显示") : t("隐藏"),t("切换物体显示")],['delete',t("删除"),t("删除物体，并将对应高斯归为背景（ID 0）")]]) {
            const button = document.createElement('button'); button.textContent = label; button.title = title;
            button.onclick = run(async () => { const result = await job('objects', { action, id: obj.id }); release(); mode = 'all'; updateTabs(); await present(result); if(action==='delete') status(() => t("物体已删除，对应高斯已归为 ID 0；可撤销，保存分割结果后写入场景知识文件")); });
            row.append(button);
        }
        $('objects').append(row);
    }
    for (const [index,id] of ['merge-first','merge-second'].entries()) {
        const select=$(id), previous=select.value;
        select.replaceChildren();
        for (const obj of state.objects) {
            const option=document.createElement('option'); option.value=obj.id;
            option.textContent=`ID ${obj.number} · ${obj.name}`; select.append(option);
        }
        if (state.objects.some(o=>o.id===previous)) select.value=previous;
        else select.selectedIndex=Math.min(index,state.objects.length-1);
    }
}
for (const id of ['merge-first','merge-second']) $(id).onchange=controls;
$('rename-cancel').onclick=()=> $('rename-dialog').close();
$('rename-dialog').addEventListener('cancel',event=>{if(busy)event.preventDefault();});
$('rename-form').onsubmit=run(async event=>{
    event.preventDefault();
    const result=await job('objects',{action:'rename',id:renameObjectId,name:$('rename-name').value});
    // Renaming updates metadata without replacing the current view or prompts.
    if(state.frame_id)result.frame_id=state.frame_id;
    await present(result); $('rename-dialog').close();
    status(() => t("物体名称已修改，点击“保存分割结果”写入场景知识文件"));
});
function updateTabs() { document.querySelectorAll('#display button').forEach(el => el.classList.toggle('active', el.dataset.mode === mode)); }
function storeRecent() {
    try { localStorage.setItem('sago-recent-scenes', JSON.stringify(recentScenes)); } catch {}
    renderRecent();
}
function renderRecent() {
    $('recent-scenes').replaceChildren();
    if (!recentScenes.length) {
        const empty=document.createElement('p'); empty.className='subtle'; localized(empty, "textContent", () => t("打开过的场景会显示在这里")); $('recent-scenes').append(empty);
    }
    for (const entry of recentScenes) {
        const row=document.createElement('div'); row.className='recent-scene';
        const open=document.createElement('button'); open.className='recent-open'; open.title=entry.path; open.disabled=busy;
        const name=document.createElement('span'); name.textContent=entry.name;
        const path=document.createElement('small'); path.textContent=entry.path; open.append(name,path);
        open.onclick=run(()=>loadScene(entry.path,entry.name));
        const remove=document.createElement('button'); remove.className='recent-remove'; remove.textContent='×'; localized(remove, "title", () => t("移除此记录")); remove.disabled=busy;
        remove.onclick=()=>{recentScenes=recentScenes.filter(x=>x.path!==entry.path);storeRecent();};
        row.append(open,remove); $('recent-scenes').append(row);
    }
}
async function loadScene(path, name) {
    path=path.trim();
    if (!path) throw new Error(t("请输入本机 PLY 路径"));
    const result=await job('load',{path});
    foregroundIds.clear(); knownObjects.clear();
    clearDroneViews();
    release(); mode='all'; updateTabs(); await present(result);
    $('path').value=path;
    recentScenes=[{path,name:name || path.split(/[\\/]/).filter(Boolean).pop() || path},...recentScenes.filter(x=>x.path!==path)].slice(0,8);
    storeRecent();
    try { localStorage.setItem('sago-scene',path); } catch {}
    if (result.knowledge_warning) status(() => result.knowledge_warning, true);
    else if (result.ids_restored) status(() => t("场景已加载 · 已从 SAGO_knowledge.json 恢复 {0} 个物体 ID{1}", result.objects.length, result.ground_restored ? t("及地面") : ''));
    else if (result.ground_restored) status(() => t("场景已加载 · 已从 SAGO_knowledge.json 恢复地面与向上方向"));
}
bind('load',()=>loadScene($('path').value));
bind('clear-recent',()=>{recentScenes=[];storeRecent();});
bind('open-scene',async()=>{
    if (!nativePicker) { $('scene-file').click(); return; }
    try {
        const result=await job('pick',{initial:$('path').value});
        if (!result.cancelled) await loadScene(result.path);
        else status(() => t("已取消选择，当前场景保持不变"));
    } catch (error) {
        $('upload-scene').hidden=false;
        throw error;
    }
});
bind('upload-scene',()=> $('scene-file').click());
async function uploadScene(file) {
    if (busy) throw new Error(t("请等待当前任务完成"));
    if (!file || !file.name.toLowerCase().endsWith('.ply')) throw new Error(t("请拖入或选择一个 PLY 文件"));
    if (file.size>2*1024**3-1024*1024) throw new Error(t("文件过大，请使用系统文件选择或输入本机路径（上传上限约 2 GB）"));
    busy=true; controls(); $('busy-overlay').hidden=false; $('cancel').hidden=true;
    localized($('busy-message'), "textContent", () => t("正在读取 {0}…", file.name)); $('busy-elapsed').textContent=''; $('progress').value=0;
    let uploaded;
    try {
        uploaded=await new Promise((resolve,reject)=>{
            const xhr=new XMLHttpRequest(); xhr.open('POST','/api/scenes/upload'); xhr.setRequestHeader('X-SAGO-Token',token);
            xhr.upload.onprogress=event=>{if(event.lengthComputable){const percent=Math.round(event.loaded/event.total*100);localized($('busy-message'), "textContent", () => t("接收 {0} · {1}%", file.name, percent));$('progress').value=percent/100*6;}};
            xhr.onerror=()=>reject(new Error(t("文件传输失败，请检查本机服务后重试")));
            xhr.onload=()=>{let result;try{result=JSON.parse(xhr.responseText);}catch{reject(new Error(t("文件传输失败")));return;} if(xhr.status>=200 && xhr.status<300) resolve(result);else reject(new Error(result.error || t("文件传输失败")));};
            const body=new FormData();body.append('file',file);xhr.send(body);
        });
    } finally { busy=false; $('busy-overlay').hidden=true; controls(); }
    await loadScene(uploaded.path,uploaded.name);
}
$('scene-file').onchange=run(async()=>{const file=$('scene-file').files[0];$('scene-file').value='';if(file)await uploadScene(file);});
let dragDepth=0;
const hasFiles=event=>Array.from(event.dataTransfer?.types || []).includes('Files');
window.addEventListener('dragover',event=>{if(hasFiles(event))event.preventDefault();});
window.addEventListener('drop',event=>{if(hasFiles(event))event.preventDefault();});
$('viewport').addEventListener('dragenter',event=>{if(!hasFiles(event))return;event.preventDefault();if(!busy){dragDepth++;$('drop-hint').hidden=false;}});
$('viewport').addEventListener('dragleave',()=>{if(--dragDepth<=0){dragDepth=0;$('drop-hint').hidden=true;}});
$('viewport').addEventListener('dragover',event=>{if(hasFiles(event)){event.preventDefault();event.dataTransfer.dropEffect=busy?'none':'copy';}});
$('viewport').addEventListener('drop',run(async event=>{
    event.preventDefault();dragDepth=0;$('drop-hint').hidden=true;
    const files=event.dataTransfer?.files;
    if (!files?.length) return;
    if (files.length!==1) throw new Error(t("请每次拖入一个 PLY 场景"));
    await uploadScene(files[0]);
}));
async function groundOperation(operation, extra = {}) {
    const result=await job('ground', {operation, capture:groundSession?.id, ...extra});
    if (result.ground_capture) {
        groundSession={id:result.ground_capture, points:result.ground_points, pixels:result.ground_pixels, plane:result.ground_draft, arrow:result.ground_arrow};
    } else groundSession=null;
    await present(result); draw(); controls();
    status(() => groundSession ? (groundSession.plane ? t("三点已就绪，请检查向上方向并点击“应用地面”") : t("请选择第 {0} 个地面点", groundSession.points.length+1)) : operation==='cancel' ? t("已取消本次地面标定") : t("已更新地面向上方向并保存到 SAGO_knowledge.json"));
}
bind('ground-start', async () => {
    release(); clearDroneViews(); mode='all'; updateTabs();
    await groundOperation('start');
});
for (const operation of ['undo','reset','apply','cancel','flip']) bind(`ground-${operation}`, () => groundOperation(operation));
async function enterPromptMode() {
    const result = await job('capture');
    clearPrompts(); capture = result.capture; mode = result.capture_mode || (state?.selected ? 'foreground' : 'all'); updateTabs();
    await present(result);
    if (loadedModel !== selectedModel()) await loadModel();
    status(() => t("已进入分割提示：添加文本、前景点或框后自动分割"));
    schedulePreview(0);
}
async function changePromptMode() {
    if ($('prompt-mode').checked) {
        await enterPromptMode();
    } else {
        release(); if (!busy) await present(await job('frame', { mode }));
    }
}
$('prompt-mode').onchange = run(changePromptMode);
bind('clear-prompt', () => { $('text-prompt').value=''; clearPrompts(); });
function invalidateCandidates() {
    autoPreview.cancel();
    masks=[]; maskImage=null; level=0;
    $('candidates').replaceChildren();
    $('preview-status').dataset.state='idle';
    localized($('preview-status'), "textContent", () => t("添加提示后自动分割"));
    controls(); draw();
}
function promptSnapshot() {
    if (!capture || composingPrompt) return null;
    const text=$('text-prompt').value.trim();
    if (!text && !fg.length && box.length!==2) return null;
    return {capture, model:selectedModel(), text, foreground:structuredClone(fg), background:structuredClone(bg), box:structuredClone(box)};
}
function schedulePreview(delay=160) {
    invalidateCandidates();
    if (!promptSnapshot()) return;
    $('preview-status').dataset.state='waiting';
    localized($('preview-status'), "textContent", () => t("等待更新二维分割…"));
    autoPreview.request(delay);
}
async function automaticPreview(input,current) {
    if (loadedModel !== input.model || (input.model==='sam2' && input.text && !groundingLoaded)) {
        await loadModel(input.model==='sam2' && !!input.text);
    }
    if (!current()) return null;
    $('preview-status').dataset.state='running';
    localized($('preview-status'), "textContent", () => t("正在更新二维分割…"));
    return job('preview',input,true);
}
async function showCandidates(result,current) {
    const images=result.masks || [];
    if (!images.length) throw new Error(t("未找到匹配物体，请调整提示"));
    const first=await loadImage(images[0]);
    if (!current()) return;
    masks=images; level=0; maskImage=first; $('candidates').replaceChildren();
    masks.forEach((src, i) => {
        const button = document.createElement('button'); const image = document.createElement('img'); image.src = src; localized(image, "alt", () => t("候选 {0}", i));
        button.append(image, messageText(() => result.model==='sam2' ? `${backendText(result.candidate_labels?.[i]) || `Level ${i}`}${result.scores?.[i]!=null ? ` · ${(result.scores[i]*100).toFixed(0)}%` : ''}` : t("实例 {0}{1}", i+1, result.scores?.[i]!=null ? ` · ${(result.scores[i]*100).toFixed(0)}%` : ''))); button.classList.toggle('active', i===0);
        button.onclick = run(async () => {
            if (!current() || busy) return;
            level=i;
            Array.from($('candidates').children).forEach((el,j)=>el.classList.toggle('active',j===i));
            const chosen=await loadImage(src);
            if (!current() || busy || level!==i) return;
            maskImage=chosen; draw();
        });
        $('candidates').append(button);
    });
    showTiming(result.timing); draw(); controls();
    $('preview-status').dataset.state='done';
    localized($('preview-status'), "textContent", () => t("二维分割已更新 · {0} 个候选", masks.length));
    status(() => t("二维分割已更新，可继续添加提示或选择候选轮廓"));
}
$('seg-model').onchange = run(async () => {
    invalidateCandidates(); clearDroneViews(); samSeconds=null; localized($('timing'), "textContent", () => t("尚未分割"));
    await loadModel(); status(() => t("已切换为 {0}", selectedModel().toUpperCase())); schedulePreview(0);
});
$('text-prompt').oninput = event => { if (event.isComposing || composingPrompt) invalidateCandidates(); else schedulePreview(450); };
$('text-prompt').addEventListener('compositionstart',()=> { composingPrompt=true; invalidateCandidates(); });
$('text-prompt').addEventListener('compositionend',()=> { composingPrompt=false; schedulePreview(450); });
for (const name of ['point','box']) bind(`${name}-tool`, () => { tool = name; $('point-tool').classList.toggle('active', tool==='point'); $('box-tool').classList.toggle('active', tool==='box'); });
bind('drone', async () => {
    const result = await job('apply', { capture, model:selectedModel(), level, padding: Number($('padding').value), operation: $('operation').value, adversarial_background:$('adversarial-background').checked, drone:true });
    release(); mode = 'foreground'; updateTabs(); await present(result);
    status(() => t("六视角细化完成，选中 {0} 个高斯", result.selected.toLocaleString()));
});
bind('cancel', async () => { if (currentJob) { await json(`/api/jobs/${currentJob}/cancel`, { method: 'POST', body: '{}' }); localized($('busy-message'), "textContent", () => t("等待当前 GPU 步骤结束后取消…")); } });
bind('add-id', async () => {
    const result = await job('objects', { action:'add_id', name:$('object-name').value });
    release(); clearDroneViews(); mode='all'; updateTabs(); await present(result);
    $('object-name').value=''; $('text-prompt').value=''; $('operation').value='set';
    $('prompt-mode').checked=true;
    await changePromptMode();
    status(() => t("已增加 ID {0} 并隐藏物体，请继续添加分割提示", result.objects.at(-1).number));
});
bind('save-ids', async () => {
    const result = await job('objects', { action:'save_ids' });
    state = { ...state, ...result }; controls();
    status(() => t("已保存 {0} 个物体 ID · {1} 维整数向量 → {2}", result.objects.length, result.count.toLocaleString(), result.knowledge_path));
});
bind('merge-ids', async () => {
    const ids=[$('merge-first').value,$('merge-second').value];
    const numbers=state.objects.filter(o=>ids.includes(o.id)).map(o=>o.number).sort((a,b)=>a-b);
    const result=await job('objects', {action:'merge',ids});
    release(); clearDroneViews(); mode='foreground'; updateTabs(); await present(result);
    status(() => t("ID {0} 已合并为 ID {1} · 点击“保存分割结果”保存", numbers.join('、'), numbers[0]));
});
bind('renumber-ids', async () => {
    const result=await job('objects', {action:'renumber'});
    release(); mode='all'; updateTabs(); await present(result);
    await present(await job('frame',{mode}));
    status(() => t("已按原 ID 排序并连续编号为 1–{0} · 点击“保存分割结果”保存", result.objects.length));
});
async function showMultiForeground() {
    if (capture || groundSession) throw new Error(t("请先退出分割提示或地面标定，再查看多物体前景"));
    mode='all'; updateTabs(); await present(await job('frame',{mode}));
    status(() => t("全部 · 背景{0} · 物体前景{1} · {2} 个物体{3}", $('show-background').checked?t("开"):t("关"), $('show-foreground').checked?t("开"):t("关"), foregroundIds.size, $('color-ids').checked ? t(" · 按 ID 着色") : t(" · 原始颜色")));
}
bind('multi-all', async()=>{for(const obj of state.objects)foregroundIds.add(obj.id);await showMultiForeground();});
bind('multi-none', async()=>{foregroundIds.clear();await showMultiForeground();});
for(const id of ['show-background','show-foreground','color-ids']) $(id).onchange=run(showMultiForeground);
bind('background-box', async()=>{
    if (mode!=='all') return;
    if(!backgroundBox) {
        $('correction-id').value='0';
        $('correction-mode').value='ellipse';
        localized($('correction-existing'), "textContent", () => t("已有 ID：{0}", state.objects.map(o=>`${o.number}（${o.name}）`).join('、') || t("无")));
        $('correction-dialog').showModal(); $('correction-id').focus(); $('correction-id').select();
        return;
    }
    backgroundBox=false; correctionPreview=null; backgroundRect=[]; drag=null;
    clearTimeout(wheelTimer);clearTimeout(resizeTimer);controls();draw();
    await present(await job('objects',{action:'cancel_correction'}));
    status(() => t("已退出修正，未确认的选中已取消"));
});
bind('correction-confirm', async()=>{
    if (!backgroundBox || !correctionPreview?.preview_count) return;
    const result=await job('objects',{action:'confirm_correction',preview_id:correctionPreview.preview_id});
    await present(result);
    status(() => t("已将 {0} 个可见高斯修正为 ID {1} · 支持撤销，点击“保存分割结果”保存", result.corrected_count.toLocaleString(), result.target_id));
});
bind('correction-discard', async()=>{
    if (!backgroundBox) return;
    await present(await job('objects',{action:'cancel_correction'}));
    status(() => t("已取消选中，ID 未修改，可重新框选"));
});
$('correction-cancel').onclick=()=> $('correction-dialog').close();
$('correction-form').onsubmit=run(async event=>{
    event.preventDefault();
    const target=Number($('correction-id').value);
    if(!Number.isInteger(target) || target<0 || target>2147483647) throw new Error(t("请输入有效的非负整数 ID"));
    correctionId=target; correctionMode=$('correction-mode').value; correctionPreview=null; backgroundBox=true; backgroundRect=[]; drag=null;
    $('correction-dialog').close(); clearTimeout(wheelTimer);clearTimeout(resizeTimer);controls();draw();
    status(() => t("{0}：连续点击累加选中，或拖框替换选中，高亮后点击“确定修正”归入 ID {1} · Esc 退出", correctionMode==='ellipse'?t("椭圆修正"):t("点修正"), correctionId));
});
for (const action of ['undo','redo','clear','save']) bind(action === 'save' ? 'save-object' : action, async () => {
    const keepMulti=mode==='all' && ['undo','redo'].includes(action);
    const result = await job('objects', { action, name: $('object-name').value }); release(); backgroundRect=[]; mode='all'; updateTabs(); await present(result);
    if (keepMulti) await present(await job('frame',{mode}));
});
function showGif(index) {
    const item=gifItems[index];
    if (!item) return;
    $('gif-preview').src=item.download;
    $('gif-download').href=item.download;
    $('gif-download').download=item.filename || '';
    $('gif-saved-path').hidden=!item.saved_path;
    localized($('gif-saved-path'), 'textContent', () => item.saved_path ? t("已保存到：{0}", item.saved_path) : '');
    localized($('gif-info'), 'textContent', () => t("{0} 帧 · {1} FPS · {2} × {2} · 时长 {3} 秒 · 生成用时 {4} 秒", item.frames, item.fps, item.size, item.duration_seconds.toFixed(1), item.generation_seconds.toFixed(2)));
}
$('gif-item').onchange=()=>showGif(Number($('gif-item').value));
async function exportGif(source) {
    if (gifExportVersion < 3) throw new Error(t("当前后端未更新，GIF 尚未生成。请先保存分割结果，再重启 Python 后端并刷新网页"));
    const elevation=Number($('gif-elevation').value);
    if (!$('gif-elevation').value.trim() || !Number.isFinite(elevation) || elevation < -89 || elevation > 89) throw new Error(t("GIF 俯仰角应在 -89° 到 89° 之间"));
    const data={source,size:Number($('gif-size').value), frames_per_orbit:Number($('gif-frames').value), elevation_degrees:elevation, fps:20};
    const result = await job('gif', data);
    if (source==='all' && result.source!=='all') throw new Error(t("当前后端不支持批量 GIF，请重启 Python 后端并刷新网页"));
    gifItems=result.items || [result];
    $('gif-item').replaceChildren(...gifItems.map((item,index)=>{
        const option=document.createElement('option');option.value=index;
        if(item.number!=null)option.textContent=`ID ${item.number} · ${item.name}`;
        else localized(option, 'textContent', () => t("当前前景"));
        return option;
    }));
    $('gif-item-label').hidden=source!=='all';
    $('gif-item').value='0';showGif(0);
    $('gif-open').hidden=false;
    $('gif-dialog').showModal();
    status(() => source==='all' ? t("已生成 {0} 个物体 GIF", gifItems.length) : t("前景 GIF 已生成"));
}
bind('export-gif', () => exportGif('current'));
bind('export-all-gif', () => exportGif('all'));
bind('gif-open', () => $('gif-dialog').showModal());
bind('gif-close', () => $('gif-dialog').close());
for (const [id, kind] of [['export-ply','ply'],['export-mask','mask'],['export-image','image']]) bind(id, async () => {
    const result = await job('export', { kind, mode }); const a = document.createElement('a'); a.href=result.download; a.download=''; a.click();
});
bind('export-2d', async () => {
    // Export the binary mask from the backend, not a thresholded UI overlay.
    const response = await job('export', { kind: 'mask2d', capture, level });
    const a=document.createElement('a'); a.href=response.download; a.download=''; a.click();
});
for (const button of document.querySelectorAll('#display button')) button.onclick = run(async () => {
    if (!state?.count) return;
    if (capture) throw new Error(t("请先取消“进入分割提示”，再切换三维显示"));
    mode = button.dataset.mode; updateTabs(); await present(await job('frame', { mode }));
});
for(const button of document.querySelectorAll('#render-mode button')) button.onclick=run(async()=>{
    if(!state?.count || capture || groundSession || backgroundBox) return;
    const previous=renderMode;
    renderMode=button.dataset.render;
    try {
        await present(await job('frame',{mode}));
        status(() => t("渲染模式：{0}", button.textContent));
    } catch(error) {renderMode=previous;controls();throw error;}
});
$('camera-preset').onchange = run(async () => {
    if (capture) throw new Error(t("请先取消“进入分割提示”，再切换相机"));
    await present(await job('view',{preset:Number($('camera-preset').value),mode}));
});
bind('camera-apply', async () => {
    if (capture) throw new Error(t("请先取消“进入分割提示”，再调整相机"));
    await present(await job('view', { fov:Number($('fov').value), size:Number($('size').value), up:['x','y','z'].map(a=>Number($(`up-${a}`).value)), target:['x','y','z'].map(a=>Number($(`target-${a}`).value)), mode }));
});
const pixel = event => { const r=canvas.getBoundingClientRect();
    if (capture || groundSession) { const scale=Math.min(r.width/canvas.width,r.height/canvas.height); const w=canvas.width*scale,h=canvas.height*scale; r.x+=(r.width-w)/2; r.y+=(r.height-h)/2; r.width=w; r.height=h; }
    if (groundSession && (event.clientX<r.x || event.clientX>=r.x+r.width || event.clientY<r.y || event.clientY>=r.y+r.height)) return null;
    return [Math.max(0,Math.min(canvas.width-1,(event.clientX-r.x)/r.width*canvas.width)), Math.max(0,Math.min(canvas.height-1,(event.clientY-r.y)/r.height*canvas.height))]; };
canvas.oncontextmenu = event => event.preventDefault();
canvas.onpointerdown = event => {
    if ((busy && !(previewBusy && capture)) || !state) return;
    const start= pixel(event);
    if (!start || ((groundSession || backgroundBox) && event.button!==0)) return;
    canvas.setPointerCapture(event.pointerId);
    drag = { start, last:pixel(event), button:event.button, shift:event.shiftKey, camera:structuredClone(state.camera), frameId:state.frame_id };
    if (backgroundBox) { backgroundRect=[start,start];draw();return; }
    if (capture && tool==='box') { invalidateCandidates(); box=[drag.start,drag.start]; draw(); }
};
canvas.onpointermove = event => { if (!drag) return; drag.last=pixel(event); if (backgroundBox) {backgroundRect=[drag.start,drag.last];draw();} else if (capture && tool==='box') { box=[drag.start, drag.last]; draw(); } };
canvas.onpointerup = run(async event => {
    if (!drag) return;
    const d=drag; drag=null;
    if (busy && !(previewBusy && capture)) return;
    const end=pixel(event); if (!end) return;
    const dx=end[0]-d.start[0], dy=end[1]-d.start[1];
    if (backgroundBox) {
        backgroundRect=[]; draw();
        const click=Math.abs(dx)<3 && Math.abs(dy)<3;
        if (!click && (Math.abs(dx)<2 || Math.abs(dy)<2)) return;
        const selection=click ? {pixel:end} : {rectangle:[d.start,end]};
        const result=await job('objects',{action:'preview_correction',target_id:correctionId,correction_mode:correctionMode,frame_id:d.frameId,...selection});
        await present(result);
        status(() => result.preview_count ? t("已高亮 {0} 个可见高斯 · 点击“确定修正”归入 ID {1}，继续点击可累加，拖框可替换选中", result.preview_count.toLocaleString(), result.target_id) : t("没有选中需要修改 ID 的可见高斯，请点击高斯或拖框重选"));
        return;
    }
    if (groundSession) {
        if (groundSession.points.length<3 && Math.abs(dx)+Math.abs(dy)<6) await groundOperation('pick',{pixel:end});
        return;
    }
    if (capture) {
        if (tool==='point') (d.button===2 ? bg : fg).push(end);
        else box = Math.abs(dx)>2 && Math.abs(dy)>2 ? [[Math.min(d.start[0],end[0]),Math.min(d.start[1],end[1])],[Math.max(d.start[0],end[0]),Math.max(d.start[1],end[1])]] : [];
        schedulePreview(); return;
    }
    if (Math.abs(dx)+Math.abs(dy)<2) return;
    const v=d.camera;
    if (d.shift || d.button===2) {
        // Camera screen-plane translation based on the current yaw and up basis.
        const up=v.up, length=Math.hypot(...up), n=up.map(x=>x/length);
        let ref=Math.abs(n[2])<.9 ? [0,0,1] : [1,0,0];
        const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
        let right=cross(n,ref); const norm=Math.hypot(...right); right=right.map(x=>x/norm);
        const forward=cross(right,n), angle=v.yaw*Math.PI/180;
        right=right.map((x,i)=>x*Math.cos(angle)+forward[i]*Math.sin(angle));
        v.target=v.target.map((x,i)=>x+(-dx*right[i]+dy*n[i])*v.radius/canvas.width);
    } else { v.yaw+=dx*.25; v.pitch=Math.max(-85,Math.min(85,v.pitch+dy*.25)); }
    await present(await job('view',{ ...v,mode }));
});
canvas.onpointercancel = () => { drag=null; if(backgroundBox){backgroundRect=[];draw();} else if (capture && tool==='box') {box=[]; schedulePreview();} else autoPreview.resume(); };
canvas.onwheel = event => {
    event.preventDefault();
    if (busy || capture || groundSession || backgroundBox || !state) return;
    clearTimeout(wheelTimer); state.camera.radius=Math.max(.001,Math.min(1e7,state.camera.radius*Math.exp(event.deltaY*.001)));
    wheelTimer=setTimeout(run(async()=>{ if (!busy && !capture && !groundSession) await present(await job('view',{radius:state.camera.radius,mode})); }),150);
};
$('help').onclick = () => $('guide').showModal(); $('close-help').onclick=()=> $('guide').close();
window.addEventListener('keydown', run(async event => {
    const active = document.activeElement;
    if (document.querySelector('dialog[open]')) return;
    if (event.isComposing || active.isContentEditable || active.matches('textarea,select,input:not([type=checkbox])')) return;
    if (event.key==='Escape' && backgroundBox && !busy) {event.preventDefault();$('background-box').click();return;}
    if (event.key.toLowerCase()==='p' && !event.ctrlKey && !event.metaKey && !event.altKey && !event.repeat) {
        if (!busy && !$('drone').disabled && !document.querySelector('dialog[open]')) {
            event.preventDefault(); $('drone').click();
        }
        return;
    }
    if (event.key.toLowerCase()==='m' && !event.ctrlKey && !event.metaKey && !event.altKey && !event.repeat) {
        if ((!busy || previewBusy) && !$('prompt-mode').disabled && !$('guide').open) {
            event.preventDefault(); $('prompt-mode').click();
        }
        return;
    }
    if (event.key==='Escape' && groundSession && !busy) { event.preventDefault(); await groundOperation('cancel'); return; }
    if (event.key==='Escape' && previewBusy) { event.preventDefault(); release(); status(() => t("已退出分割提示")); return; }
    if (event.key==='Escape' && currentJob) { $('cancel').click(); return; }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase()==='z' && !busy) { event.preventDefault(); $(event.shiftKey?'redo':'undo').click(); }
}));
// Keep the actual render aspect in sync; frozen prompts stay mapped to their captured image.
new ResizeObserver(()=> {
    clearTimeout(resizeTimer);
    resizeTimer=setTimeout(run(async()=> {
        if (!state || busy || capture || groundSession || backgroundBox || drag) return;
        await present(await job('frame',{mode}));
    }),250);
}).observe($('viewport'));
$('path').value = localStorage.getItem('sago-scene') || '';
initLanguage(() => { if (state) renderObjects(); renderRecent(); controls(); draw(); });
renderRecent();
controls();
run(async () => {
    const config=await json('/api/config'); gifExportVersion=config.gif_export_version || 0; token=config.token; nativePicker=!!config.nativePicker;
    if (config.scene_loaded) {
        await present(await job('frame', {mode}));
        status(() => t("Python 后端已连接 · 已恢复当前场景与 {0} 个物体 ID", state.objects.length));
    } else status(() => t("Python 后端已连接 · 加载场景开始"));
})();

function setBackgroundColor(rgb) {
    ['r','g','b'].forEach((c,i)=>{$(`background-${c}`).value=rgb[i];});
    $('background-color').value='#'+rgb.map(c=>Math.round(c*255).toString(16).padStart(2,'0')).join('');
}
$('background-color').onchange=run(async()=>{
    const hex=$('background-color').value;
    setBackgroundColor([1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255));
    await present(await job('frame',{mode}));
});
for(const channel of ['r','g','b']) $(`background-${channel}`).onchange=run(async()=>{
    const rgb=['r','g','b'].map(c=>Number($(`background-${c}`).value));
    if(rgb.some(c=>!Number.isFinite(c)||c<0||c>1)) {
        setBackgroundColor(state?.background_color || [0,0,0]);
        return;
    }
    setBackgroundColor(rgb);
    await present(await job('frame',{mode}));
});
