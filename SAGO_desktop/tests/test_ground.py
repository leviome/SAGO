"""World-space ground geometry, depth validation and calibration session safety."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from backend.engine import Engine
from backend.ground import ground_plane, flip_plane


class GroundTest(unittest.TestCase):
    def test_plane_unit_normal_equation_and_winding_independence(self):
        points = np.array([[1, 2, 3], [2, 2, 4], [1, 4, 3]], dtype=float)
        plane = ground_plane(points, [0, 3, 10], [0, -1, 0])
        reverse = ground_plane(points[[0, 2, 1]], [0, 3, 10], [0, -1, 0])
        np.testing.assert_allclose(plane['normal'], reverse['normal'])
        self.assertAlmostEqual(np.linalg.norm(plane['normal']), 1.)
        np.testing.assert_allclose(points @ plane['normal']+plane['offset'], 0., atol=1e-12)
        self.assertGreater(np.dot(plane['normal'], np.array([0, 3, 10])-points.mean(0)), 0)
        flipped = flip_plane(plane)
        np.testing.assert_allclose(flipped['normal'], -np.array(plane['normal']))
        np.testing.assert_allclose(points @ flipped['normal']+flipped['offset'], 0., atol=1e-12)

    def test_degenerate_and_nonfinite_points_rejected(self):
        for points in ([[0, 0, 0]]*3, [[0, 0, 0], [1, 0, 0], [2, .001, 0]],
                       [[0, 0, 0], [1, 0, 0], [float('nan'), 1, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                ground_plane(points, [0, 0, 1], [0, 0, 1])

    def fixture(self, folder):
        engine = Engine('unused.pt', folder)
        engine.torch = torch
        engine.model = object()
        engine.view = dict(up=[0, -1, 0], target=[0, 0, -2], radius=2., yaw=0., pitch=0.)
        cam = SimpleNamespace(world_view_transform=torch.eye(4))
        engine.camera = lambda: cam
        def freeze():
            engine.capture = dict(id='ground', version=engine.scene_version, cam=cam,
                rgb=np.zeros((8, 10, 3), np.uint8), depth=torch.full((1, 8, 10), 2.))
            return dict(capture='ground')
        engine.freeze = freeze
        engine.load_model = Mock()
        engine.utils = SimpleNamespace(load_3dgs_camera=lambda *a, **kw: cam,
            back_projection=lambda screen, depth, cam: torch.stack([screen[:, 0], screen[:, 1], -depth], dim=1),
            params2sphere=lambda *a: (2., .2, .1))
        engine.frame = lambda: dict(camera=engine.view, ground=engine.ground_plane)
        engine.ground_action(dict(operation='start'))
        return engine

    def pick(self, engine, pixel):
        return engine.ground_action(dict(operation='pick', capture='ground', pixel=pixel))

    def test_pick_apply_flip_and_model_independence(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.fixture(folder)
            for pixel in ([1, 1], [3, 1], [1, 3]):
                result = self.pick(engine, pixel)
            self.assertEqual(result['ground_points'], [[1., 1., -2.], [3., 1., -2.], [1., 3., -2.]])
            np.testing.assert_allclose(result['ground_draft']['normal'], [0, 0, 1])
            self.assertIsNone(engine.ground_plane)
            engine.ground_action(dict(operation='apply', capture='ground'))
            self.assertIsNone(engine.ground_capture)
            self.assertEqual(engine.view['up'], [0., 0., 1.])
            self.assertIsNone(engine.capture)
            engine.ground_action(dict(operation='flip'))
            self.assertEqual(engine.view['up'], [0., 0., -1.])
            engine.load_model.assert_not_called()

    def test_invalid_pixels_depth_duplicates_and_collinearity_keep_previous_points(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.fixture(folder)
            for pixel in (None, [-1, 0], [10, 0], [0, float('nan')]):
                with self.assertRaises(ValueError):
                    self.pick(engine, pixel)
            for depth in (0., -1., float('nan'), float('inf')):
                engine.ground_capture['depth'][0, 0, 0] = depth
                with self.assertRaisesRegex(ValueError, '有效表面深度'):
                    self.pick(engine, [0, 0])
            self.pick(engine, [1, 1])
            with self.assertRaisesRegex(ValueError, '重合'):
                self.pick(engine, [1, 1])
            self.pick(engine, [2, 2])
            with self.assertRaisesRegex(ValueError, '共线'):
                self.pick(engine, [3, 3])
            self.assertEqual(len(engine.ground_capture['points']), 2)
            self.assertIsNone(engine.ground_capture['plane'])
            engine.ground_action(dict(operation='undo', capture='ground'))
            self.assertEqual(len(engine.ground_capture['points']), 1)
            engine.ground_action(dict(operation='reset', capture='ground'))
            self.assertEqual(engine.ground_capture['points'], [])

    def test_cancel_keeps_previous_plane_and_stale_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            engine = self.fixture(folder)
            plane = ground_plane([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 0, 1], [0, 0, 1])
            engine.ground_plane = plane
            engine.ground_action(dict(operation='cancel', capture='ground'))
            self.assertIs(engine.ground_plane, plane)
            engine.ground_action(dict(operation='start'))
            engine.invalidate()
            with self.assertRaisesRegex(ValueError, '失效'):
                self.pick(engine, [1, 1])
