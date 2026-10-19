import { messages, backendKeys } from '/assets/translations.js';

let language = 'en';
try { if (localStorage.getItem('sago-language') === 'zh') language = 'zh'; } catch {}
const bindings = new Map();
let onChange = () => {};
const format = (text, values) => text.replace(/\{(\d+)\}/g, (match, i) => values[i] == null ? match : String(values[i]));
export const getLanguage = () => language;
export function t(source, ...values) {
    return format(language === 'en' ? (messages[source] ?? source) : source, values);
}
const escape = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const reverseMessages = new Map(Object.entries(messages).map(([source,english])=>[english,source]));
const backendPatterns = backendKeys.filter(key => /\{\d+\}/.test(key)).flatMap(key => [false,true].map(reverse => {
    const slots=[];
    const parts=(reverse ? messages[key] : key).split(/(\{\d+\})/g);
    const pattern=parts.map(part=>{
        if (/^\{\d+\}$/.test(part)) {slots.push(Number(part.slice(1,-1)));return '([\\s\\S]*?)';}
        return escape(part);
    }).join('');
    return { key, slots, reverse, pattern:new RegExp('^'+pattern+'$') };
}));
// Only translate backend-owned messages. Paths, object names and user prompts stay intact.
export function backendText(value) {
    const text=String(value ?? '');
    if (language === 'zh') {
        if (reverseMessages.has(text)) return reverseMessages.get(text);
        if (/[\u3400-\u9fff]/u.test(text)) return text;
    } else {
        if (messages[text] != null) return t(text);
        if (!/[\u3400-\u9fff]/u.test(text)) return text;
    }
    for (const {key,slots,pattern,reverse} of backendPatterns) {
        if (reverse !== (language === 'zh')) continue;
        const match=pattern.exec(text);
        if (match) {
            const values=[];slots.forEach((slot,i)=>values[slot]=match[i+1]);
            return t(key,...values);
        }
    }
    return text; // Preserve technical diagnostics when no translation is known.
}
export function localized(element, property, read, attribute=false) {
    const value=String(read() ?? '');
    element.removeAttribute(attribute ? 'data-i18n-'+property : property === 'textContent' ? 'data-i18n' : 'data-i18n-'+property);
    let node=null;
    if (attribute) element.setAttribute(property,value);
    else if (property==='textContent') {
        node=document.createTextNode(value);element.replaceChildren(node);
    } else element[property]=value;
    if (!bindings.has(element)) bindings.set(element,new Map());
    bindings.get(element).set(property,{read,attribute,node});
    return value;
}
export function messageText(read) {
    const span=document.createElement('span');localized(span,'textContent',read);return span;
}
function apply() {
    document.documentElement.lang=language==='en'?'en':'zh-CN';
    document.querySelectorAll('[data-i18n]').forEach(el=>el.textContent=t(el.dataset.i18n));
    for (const attribute of ['title','placeholder','aria-label','alt']) {
        document.querySelectorAll('[data-i18n-'+attribute+']').forEach(el=>el.setAttribute(attribute,t(el.getAttribute('data-i18n-'+attribute))));
    }
    for (const [element,properties] of bindings) {
        if (!element.isConnected) {bindings.delete(element);continue;}
        for (const [property,binding] of properties) {
            const {read,attribute,node}=binding;
            if (property==='textContent' && node?.parentNode!==element) {properties.delete(property);continue;}
            const value=String(read() ?? '');
            if(attribute)element.setAttribute(property,value);
            else if(node)node.nodeValue=value;
            else element[property]=value;
        }
    }
    const button=document.getElementById('language-toggle');
    button.textContent=language==='en'?'中文':'English';
    button.setAttribute('aria-label',language==='en'?'Switch to Chinese':'切换为英语');
    button.title=language==='en'?'Switch to Chinese':'切换为英语';
}
export function setLanguage(next) {
    if (!['en','zh'].includes(next)) return;
    language=next;
    try {localStorage.setItem('sago-language',language);} catch {}
    apply();onChange();
}
export function initLanguage(callback) {
    onChange=callback;
    document.getElementById('language-toggle').onclick=()=>setLanguage(language==='en'?'zh':'en');
    apply();
    // Release bindings of removed candidates/objects after they have been mounted.
    new MutationObserver(records => {
        if (!records.some(record => record.removedNodes.length)) return;
        for (const element of bindings.keys()) if (!element.isConnected) bindings.delete(element);
    }).observe(document.body,{childList:true,subtree:true});
}
