// Serialize inference and retain only the newest complete prompt snapshot.
export class LatestPreview {
    constructor({read, ready, run, apply, onError}) {
        Object.assign(this, {read, ready, run, apply, onError});
        this.version=0; this.pending=false; this.running=false;
    }
    cancel() {
        this.version++;
        this.pending=false;
        clearTimeout(this.timer);
    }
    request(delay=160) {
        this.version++;
        this.pending=true;
        this.due=performance.now()+delay;
        clearTimeout(this.timer);
        this.resume();
    }
    resume() {
        if (this.running || !this.pending || !this.ready()) return;
        clearTimeout(this.timer);
        const wait=this.due-performance.now();
        if (wait>0) { this.timer=setTimeout(()=>this.resume(),wait); return; }
        void this.execute();
    }
    async execute() {
        const version=this.version, input=this.read();
        this.pending=false;
        if (!input) return;
        this.running=true;
        const current=()=>version===this.version;
        try {
            const result=await this.run(input,current);
            if (result && current()) await this.apply(result,current);
        } catch(error) {
            if (current()) this.onError(error);
        } finally {
            this.running=false;
            this.resume();
        }
    }
}
