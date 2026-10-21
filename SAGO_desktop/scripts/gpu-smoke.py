#!/usr/bin/env python3
"""Independent real-CUDA smoke; never contacts a running SAGO service."""
import argparse
import json
import os
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
for p in (ROOT/'.runtime/extensions',):
    if p.is_dir():sys.path.insert(0,str(p))
parser=argparse.ArgumentParser();parser.add_argument('scene');parser.add_argument('--output',default='/tmp/sago-desktop-gpu-smoke');args=parser.parse_args()
scene=Path(args.scene).expanduser().resolve()
output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(output/'cache'));os.chdir(output)
from backend.engine import Engine
from PIL import Image
import numpy as np
engine=Engine(ROOT/'models/sam2_hiera_large.pt',output)
result=engine.load({'path':str(scene),'aspect':1.7})
print('Loaded',result['count'],flush=True)
# Reject accidental imports from another SAGO checkout.
origins={k:getattr(v,'__file__',None) for k,v in sys.modules.items() if k.split('.')[0] in ('backend','gs','sago_core')}
assert all(not p or Path(p).is_relative_to(ROOT) for p in origins.values()),origins
print('OWN MODULES',len(origins),flush=True)
for style in ('rgb','splats','points','depth'):
    frame=engine.execute('frame',{'render_mode':style,'mode':'all'},lambda *a:None,threading.Event())
    assert frame['image'].startswith('data:image/png;base64,')
    print('RENDER',style,flush=True)
xyz=engine.model.get_xyz.detach().cpu().numpy();center=np.median(xyz,axis=0)
dist=np.linalg.norm(xyz-center,axis=1);selection=dist<=np.quantile(dist,.03)
engine.selection=engine.torch.from_numpy(selection).cuda()
result=engine.execute('gif',{'size':128,'frames_per_orbit':8,'elevation_degrees':60},lambda *a:None,threading.Event())
with Image.open(result['saved_path']) as image:
    assert image.n_frames==8 and image.size==(128,128)
(output/'report.json').write_text(json.dumps({'count':len(xyz),'gif':result,'origins':origins},indent=2))
print('PASS independent CUDA rendering and single-orbit GIF',result['saved_path'],flush=True)
