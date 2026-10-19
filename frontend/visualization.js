// Gaussian renderer uses row vectors. Like its rasterizer, do not cull at the far plane.
export function projectSegment(a, b, matrix, width, height) {
    const clip = p => Array.from({length:4}, (_,j) => [p[0],p[1],p[2],1].reduce((sum,v,i)=>sum+v*matrix[i][j],0));
    let u=clip(a), v=clip(b);
    if (![...u,...v].every(Number.isFinite)) return null;
    const planes=[p=>p[3]-1e-6,p=>p[2],p=>p[0]+p[3],p=>p[3]-p[0],p=>p[1]+p[3],p=>p[3]-p[1]];
    for (const plane of planes) {
        const x=plane(u), y=plane(v);
        if (x<0 && y<0) return null;
        if (x<0 || y<0) {
            const t=x/(x-y), cut=u.map((value,i)=>value+t*(v[i]-value));
            if (x<0) u=cut; else v=cut;
        }
    }
    return [u,v].map(p=>[(p[0]/p[3]+1)*width*.5-.5,(p[1]/p[3]+1)*height*.5-.5]);
}

export function frustumCorners(frustum) {
    const {position, right, down, forward, tan_half_fov: [tx,ty], depth}=frustum;
    return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([x,y])=>position.map((p,i)=>p+depth*(forward[i]+x*tx*right[i]+y*ty*down[i])));
}

export function drawVisualization(ctx, data, width, height, bbox, route) {
    if (!data?.projection || (!bbox && !route)) return;
    ctx.save();
    ctx.lineWidth=Math.max(1.5,width/500);
    ctx.lineJoin='round'; ctx.shadowColor='#000'; ctx.shadowBlur=3;
    ctx.font=`bold ${Math.max(12,width/65)}px sans-serif`;
    const line=(a,b,color,label) => {
        const points=projectSegment(a,b,data.projection,width,height);
        if (!points) return;
        const [u,v]=points;
        ctx.strokeStyle=color; ctx.fillStyle=color;
        ctx.beginPath();ctx.moveTo(...u);ctx.lineTo(...v);ctx.stroke();
        if (label) {
            // Label only an actual camera position, never a clipped endpoint.
            const point=projectSegment(b,b,data.projection,width,height);
            if (point) {
                if (Math.hypot(v[0]-u[0],v[1]-u[1])>1) {
                    const angle=Math.atan2(v[1]-u[1],v[0]-u[0]), head=7;
                    ctx.beginPath();ctx.moveTo(...v);ctx.lineTo(v[0]-head*Math.cos(angle-.5),v[1]-head*Math.sin(angle-.5));ctx.lineTo(v[0]-head*Math.cos(angle+.5),v[1]-head*Math.sin(angle+.5));ctx.closePath();ctx.fill();
                }
                ctx.fillText(label,v[0]+7,v[1]-7);
            }
        }
    };
    const drawFrustum=(frustum,color,label) => {
        if (!frustum) return;
        const corners=frustumCorners(frustum);
        for (let i=0;i<4;i++) {
            line(frustum.position,corners[i],color);
            line(corners[i],corners[(i+1)%4],color);
        }
        if (label) line(frustum.position,frustum.position,color,label);
    };
    if (bbox && data.bbox) {
        const corners=data.bbox.corners;
        for(let i=0;i<8;i++) for(let j=0;j<3;j++) if(!(i&(1<<j))) line(corners[i],corners[i|(1<<j)],'#ff4058');
    }
    if (route && data.route) {
        drawFrustum(data.route.origin_frustum,'#f2e6ef','s0');
        for (const [side,color] of [[0,'#ffc86b'],[1,'#68d8ff']]) {
            let previous=data.route.origin, step=0;
            for (const view of data.route.views.filter(v=>v.drone===side)) {
                const label=`D${side} · ${++step}`;
                line(previous,view.position,color,label);
                drawFrustum(view.frustum,color);
                previous=view.position;
            }
        }
    }
    ctx.restore();
}
