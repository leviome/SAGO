"""Real CUDA integration check, using only local weights and scene data."""
import argparse
import base64
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))

from backend.engine import Engine

parser = argparse.ArgumentParser()
parser.add_argument('--scene', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--model', choices=['sam2', 'sam3', 'sam3.1'], default='sam2')
parser.add_argument('--text', default='')
args = parser.parse_args()
e = Engine(args.checkpoint, ROOT/'outputs')
frame = e.execute('load', {'path': args.scene}, print, threading.Event())
assert frame['count'] > 0 and frame['image'].startswith('data:image/png;base64,')
Path('outputs/smoke-scene.png').write_bytes(base64.b64decode(frame['image'].split(',')[1]))
print('LOAD', frame['count'], flush=True)
c = e.freeze()
size = e.view['size']
preview = e.preview(dict(capture=c['capture'], model=args.model, text=args.text,
    box=[] if args.text else [[size*.25,size*.2],[size*.75,size*.8]]))
assert len(preview['masks']) == 4 if args.model == 'sam2' and not args.text else len(preview['masks']) > 0
Path('outputs/smoke-preview.png').write_bytes(base64.b64decode(preview['masks'][0].split(',')[1]))
print(args.model.upper(), len(preview['masks']), flush=True)
result = e.apply(dict(capture=c['capture'], model=args.model, level=0, padding=5, drone=True),
                 lambda n,m,*views: print('PROGRESS',n,m,flush=True), threading.Event())
assert result['selected'] > 0
print('TIMING', result['timing'], flush=True)
if args.model != 'sam2':
    assert result['tracking_stats']['seed_runs'] == 1
    assert result['tracking_stats']['new_frames'] == 6
    assert result['tracking_stats']['image_encodes'] == 7
    print('INCREMENTAL TRACKING', result['tracking_stats'], flush=True)
route = result['route']
assert len(route) == 6
assert route[0]['center_source'] == 'surface'
assert all(step['center_source'] == 'foreground' for step in route[1:])
assert all(step['input_count'] == previous['count'] for previous, step in zip(route, route[1:]))
print('SEQUENTIAL CENTERS', [(step['center_source'], step['input_count'], step['center']) for step in route], flush=True)
selected = result['selected']
print('SELECT', selected, flush=True)
for kind in ('ply', 'mask', 'image'):
    exported = e.export({'kind':kind})
    assert (ROOT/'outputs'/exported['download'].split('/')[-1]).stat().st_size > 0
print('EXPORT OK', flush=True)
e.objects_action({'action':'save','name':'Smoke object'})
obj = e.objects[0]['id']
e.objects_action({'action':'hide','id':obj})
assert e.objects[0]['hidden']
e.objects_action({'action':'undo'})
assert not e.objects[0]['hidden']
e.objects_action({'action':'redo'})
assert e.objects[0]['hidden']
e.objects_action({'action':'undo'})
e.objects_action({'action':'undo'})
e.objects_action({'action':'undo'})
assert e.selection is None
try:
    e.preview({'capture':c['capture'], 'foreground':[[1,1]]})
    raise AssertionError('stale capture accepted')
except ValueError:
    pass
print('HISTORY, OBJECTS, STALE CAPTURE OK', flush=True)
if getattr(e, 'sam3_client', None) is not None:
    e.sam3_client.close()
