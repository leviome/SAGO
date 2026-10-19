"""Real renderer depth -> ground plane -> camera orientation integration check."""
import os
from pathlib import Path
import sys
import threading
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
from backend.engine import Engine

fixture = Path(tempfile.mkdtemp(prefix='ground-smoke-', dir=ROOT/'outputs'))
ply = fixture/'point_cloud/iteration_30000/point_cloud.ply'
ply.parent.mkdir(parents=True)
shutil.copyfile('/home/pc/gaussian-splatting/output/for_seg1/point_cloud/iteration_30000/point_cloud.ply', ply)
shutil.copyfile('/home/pc/gaussian-splatting/output/for_seg1/cameras.json', fixture/'cameras.json')
engine = Engine('/home/pc/Desktop/SAGO/sago_core/sam2/checkpoints/sam2_hiera_large.pt', ROOT/'outputs')
engine.load(dict(path=str(fixture), aspect=1.6))
initial_up = np.asarray(engine.view['up'])
start = engine.execute('ground', dict(operation='start', aspect=1.6), print, threading.Event())
token = start['ground_capture']
height, width = engine.ground_capture['rgb'].shape[:2]
eye = engine.camera().world_view_transform.inverse()[3, :3].cpu().numpy()
for x, y in ((.27, .62), (.78, .60), (.52, .75)):
    result = engine.ground_action(dict(operation='pick', capture=token, pixel=[x*width, y*height]))
plane = result['ground_draft']
points, normal = np.asarray(plane['points']), np.asarray(plane['normal'])
assert np.max(np.abs(points @ normal+plane['offset'])) < 1e-6
assert np.dot(normal, eye-points.mean(0)) >= 0
assert np.dot(normal, initial_up) > .8, (normal, initial_up)
print('PLANE', plane, flush=True)
engine.ground_action(dict(operation='apply', capture=token))
after = engine.camera().world_view_transform.inverse()[3, :3].cpu().numpy()
assert np.linalg.norm(after-eye) < 1e-4*engine.view['radius'], (eye, after)
engine.adopt_camera(1)
np.testing.assert_allclose(engine.view['up'], normal)
engine.ground_action(dict(operation='flip'))
np.testing.assert_allclose(engine.view['up'], -normal)
engine.ground_action(dict(operation='start'))
engine.ground_action(dict(operation='cancel', capture=engine.ground_capture['id']))
np.testing.assert_allclose(engine.ground_plane['normal'], -normal)
assert engine.tracker is None
assert (fixture/'SAGO_knowledge.json').is_file()
restored = Engine(engine.checkpoint, ROOT/'outputs')
restored.load(dict(path=str(ply), aspect=1.6))
assert restored.ground_restored
np.testing.assert_allclose(restored.view['up'], -normal)
np.testing.assert_allclose(restored.ground_plane['points'], points)
print('KNOWLEDGE SAVE + FRESH ENGINE RESTORE', restored.knowledge.path, flush=True)
print('DEPTH BACKPROJECTION, PLANE, CAMERA EYE, PRESET PERSISTENCE, FLIP, CANCEL, NO SAM LOAD OK', flush=True)
