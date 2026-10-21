import math
import tempfile
from types import SimpleNamespace
import unittest

import torch

from backend.engine import Engine
from backend.visualization import upright_bbox, camera_frustum


class VisualizationTest(unittest.TestCase):
    def assert_contains(self, points, box):
        axes = torch.tensor(box['axes'])
        center = torch.tensor(box['center'])
        size = torch.tensor(box['size'])
        local = (points-center) @ axes.T
        self.assertTrue(bool((local.abs() <= size*.5+1e-4).all()))
        torch.testing.assert_close(axes @ axes.T, torch.eye(3), atol=1e-5, rtol=1e-5)

    def test_selection_bounds_follow_replacement_up_change_and_clear(self):
        with tempfile.TemporaryDirectory() as root:
            engine = Engine('unused.pt', root)
            engine.view = {'up': [0, 0, 1]}
            points = torch.tensor([[-2., 1., 3.], [4., -3., 2.], [100., 100., 100.]])
            engine.model = SimpleNamespace(get_xyz=points)
            cam = SimpleNamespace(full_proj_transform=torch.eye(4))
            engine.selection = torch.tensor([True, True, False])
            result = engine.visualization(cam)
            self.assert_contains(points[:2], result['bbox'])
            self.assertIsNone(result['route'])
            self.assertIs(engine.visualization(cam)['bbox'], result['bbox'])
            engine.view['up'] = [0, 1, 1]
            changed = engine.visualization(cam)['bbox']
            self.assertIsNot(changed, result['bbox'])
            torch.testing.assert_close(torch.tensor(changed['axes'][2]), torch.tensor([0., 2**-.5, 2**-.5]))
            self.assert_contains(points[:2], changed)
            engine.selection = torch.tensor([False, False, True])
            self.assertEqual(engine.visualization(cam)['bbox']['size'], [0., 0., 0.])
            engine.selection = torch.zeros(3, dtype=torch.bool)
            self.assertIsNone(engine.visualization(cam)['bbox'])
            engine.selection = None
            self.assertIsNone(engine.visualization(cam)['bbox'])

    def test_rotated_box_tilted_up_has_tight_horizontal_area(self):
        up = torch.tensor([1., 2., 3.]); up /= up.norm()
        x = torch.linalg.cross(torch.tensor([0., 0., 1.]), up); x /= x.norm()
        y = torch.linalg.cross(up, x)
        angle = .61
        a = math.cos(angle)*x+math.sin(angle)*y
        b = torch.linalg.cross(up, a)
        points = torch.tensor([[i, j, k] for i in [-2., 2.] for j in [-1., 1.] for k in [-.5, .5]]) @ torch.stack([a,b,up]) + torch.tensor([10., -7., 3.])
        box = upright_bbox(points, up)
        self.assert_contains(points, box)
        torch.testing.assert_close(torch.tensor(box['axes'][2]), up)
        self.assertAlmostEqual(box['size'][0]*box['size'][1], 8., places=4)
        self.assertAlmostEqual(box['size'][2], 1., places=4)

    def test_empty_degenerate_geometry_and_reversed_up(self):
        self.assertIsNone(upright_bbox(torch.empty(0,3), [0,0,1]))
        for points in [torch.ones(1,3), torch.tensor([[0.,0.,0.],[1.,1.,1.]])]:
            self.assert_contains(points, upright_bbox(points, [0,0,-1]))

    def test_frustum_axes_and_intrinsics_match_render_camera(self):
        # Row-vector c2w: rotation about Y, plus translation.
        c2w = torch.tensor([[0.,0.,-1.,0.],[0.,1.,0.,0.],[1.,0.,0.,0.],[2.,3.,4.,1.]])
        cam = SimpleNamespace(world_view_transform=c2w.inverse(), FoVx=math.pi/2, FoVy=math.pi/3)
        result = camera_frustum(cam)
        self.assertEqual(result['position'], [2.,3.,4.])
        self.assertEqual(result['forward'], [1.,0.,0.])
        self.assertEqual(result['right'], [0.,0.,-1.])
        self.assertEqual(result['down'], [0.,1.,0.])
        self.assertAlmostEqual(result['tan_half_fov'][0], 1.)
        self.assertAlmostEqual(result['tan_half_fov'][1], 3**-.5)
