"""Verify navigation uses surface geometry first, then sequentially filtered geometry."""
from collections import deque
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from backend.engine import Engine


class DroneCentersTest(unittest.TestCase):
    def run_route(self, depth=None, reject_first=False, text_candidates=False):
        with tempfile.TemporaryDirectory() as root:
            engine = Engine('unused.pt', root)
            engine.torch = torch
            xyz = torch.arange(24, dtype=torch.float32).reshape(8, 3)
            engine.model = SimpleNamespace(get_xyz=xyz)
            engine.view = {'up': [0, -1, 0]}
            mask = torch.ones((2, 2), dtype=torch.bool)
            capture = dict(masks=[mask], depth=torch.ones((2, 2)) if depth is None else depth,
                           cam=SimpleNamespace(world_view_transform=torch.eye(4), FoVx=1., FoVy=.8), encoded=[None], preds=torch.ones((1, 1)),
                           score=None, pointers=torch.ones((1, 1)))
            if text_candidates:
                capture['masks'] = [mask]*8
                capture['sam2_candidates'] = [dict(preds=torch.full((1, 1), i),
                    pointers=torch.full((1, 1), i+10), score=i+20) for i in range(8)]
            engine.captured = lambda data: capture
            engine.active = lambda: torch.ones(8, dtype=torch.bool)
            engine.save_history = lambda: None
            engine.invalidate = lambda: None
            engine.frame = lambda mode: {}
            def memory_encoder(encoded, prediction, score, **kw):
                if text_candidates:
                    self.assertEqual(prediction.item(), 4)
                    self.assertEqual(score, 24)
                return None
            engine.tracker = SimpleNamespace(sam=SimpleNamespace(memory_encoder=memory_encoder))
            surface_masks, poses, rendered = [], [], []
            def surface(m, d, cam):
                surface_masks.append(m.clone())
                return torch.tensor([100., 200., 300.])
            class Filter:
                calls = 0
                def filtering(self, tracked, previous, pose, **kw):
                    if previous is None:
                        return torch.ones(8, dtype=torch.bool)
                    self.calls += 1
                    result = previous.clone()
                    if not (reject_first and self.calls == 1):
                        result[result.nonzero()[0]] = False
                    return result
            class Drone:
                def __init__(drone_self, **kw):
                    if text_candidates:
                        self.assertEqual(kw['ptr'][0].item(), 14)
                    drone_self.prev_mems, drone_self.prev_ptrs = deque(), deque()
                def return_camera_pose(self, yaw, pitch, center):
                    poses.append(center.clone())
                    return SimpleNamespace(world_view_transform=torch.eye(4), FoVx=1., FoVy=.8)
                def tracking_in_new_view(self, fg, *a, **kw):
                    rendered.append(fg.get_xyz.clone())
                    return mask
            engine.Drone = Drone
            engine.utils = SimpleNamespace(
                MaskShapedFrustumFilter=lambda *a, **kw: Filter(),
                masking_for_fg=lambda model, selected: SimpleNamespace(get_xyz=model.get_xyz[selected]),
                mask_average_location=surface, load_3dgs_camera=lambda *a, **kw: None)
            with patch.object(torch.cuda, 'synchronize'):
                result = engine.apply({'drone': True, 'level': 4 if text_candidates else 0}, lambda *a: None, threading.Event())
            self.assertEqual(engine.drone_path['origin'], [0., 0., 0.])
            initial = engine.drone_path['origin_frustum']
            self.assertEqual(initial['position'], engine.drone_path['origin'])
            self.assertEqual(initial['forward'], [0., 0., 1.])
            self.assertGreater(initial['depth'], 0.)
            return result['route'], poses, rendered, surface_masks

    def test_surface_first_then_latest_foreground_for_every_view(self):
        route, poses, rendered, masks = self.run_route(torch.tensor([[1., 0.], [float('nan'), 2.]]))
        self.assertEqual(masks[0].tolist(), [[True, False], [False, True]])
        self.assertEqual(poses[0].tolist(), [100., 200., 300.])
        self.assertEqual([r['input_count'] for r in route], [8, 7, 6, 5, 4, 3])
        self.assertEqual([r['center_source'] for r in route], ['surface'] + ['foreground']*5)
        for i in range(1, 6):
            torch.testing.assert_close(poses[i], rendered[i].mean(0))
            self.assertEqual(len(rendered[i]), 8-i)

    def test_rejected_first_view_uses_retained_foreground_next(self):
        route, poses, rendered, _ = self.run_route(reject_first=True)
        self.assertFalse(route[0]['changed'])
        self.assertEqual(route[1]['input_count'], 8)
        torch.testing.assert_close(poses[1], rendered[0].mean(0))

    def test_second_text_instance_uses_its_own_memory_score_and_pointer(self):
        route, *_ = self.run_route(text_candidates=True)
        self.assertEqual(len(route), 6)

    def test_no_valid_surface_fails_without_gaussian_fallback(self):
        with self.assertRaisesRegex(ValueError, '有效表面深度'):
            self.run_route(torch.zeros((2, 2)))
