const assert=require('node:assert/strict');
const fs=require('node:fs');
const tick=()=>new Promise(r=>setTimeout(r,15));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject};};
(async()=>{
const {LatestPreview}=await import('data:text/javascript;base64,'+Buffer.from(fs.readFileSync('frontend/auto-preview.js','utf8')).toString('base64'));
let input='a', ready=true, calls=[], shown=[], errors=[], inflight=[];
const queue=new LatestPreview({read:()=>input,ready:()=>ready,run:value=>{calls.push(value);const d=deferred();inflight.push(d);return d.promise;},apply:async(value,current)=>{await Promise.resolve();if(current())shown.push(value);},onError:e=>errors.push(e.message)});
queue.request(5);input='b';queue.request(5);await tick();assert.deepEqual(calls,['b']);
input='c';queue.request(0);input='d';queue.request(0);assert.deepEqual(calls,['b']);
inflight[0].resolve('old');await tick();assert.deepEqual(shown,[]);assert.deepEqual(calls,['b','d']);
inflight[1].resolve('new');await tick();assert.deepEqual(shown,['new']);
input='e';queue.request(0);queue.cancel();inflight[2].resolve('cancelled');await tick();assert.deepEqual(shown,['new']);
ready=false;input='f';queue.request(0);await tick();assert.equal(calls.length,3);ready=true;queue.resume();assert.equal(calls.at(-1),'f');
input='g';queue.request(0);inflight[3].reject(new Error('stale failure'));await tick();assert.deepEqual(errors,[]);assert.equal(calls.at(-1),'g');inflight[4].reject(new Error('current failure'));await tick();assert.deepEqual(errors,['current failure']);
queue.request(50);queue.cancel();await new Promise(r=>setTimeout(r,60));assert.equal(calls.length,5);
// A revision arriving during image decoding must also invalidate the result.
const decode=deferred();let valid=false;
const images=new LatestPreview({read:()=>1,ready:()=>true,run:async()=>1,apply:async(result,current)=>{await decode.promise;valid=current();},onError:e=>{throw e;}});
images.request(0);await tick();images.cancel();decode.resolve();await tick();assert.equal(valid,false);
console.log('Auto preview: debounce, newest snapshot, serialized jobs, stale success/error, exit/clear and delayed image decode OK');
})().catch(e=>{console.error(e);process.exit(1)});
