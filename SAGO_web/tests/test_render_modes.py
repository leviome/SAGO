"""Display modes preserve labels, use the active mask and resolve point depth."""
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from backend.engine import Engine
from backend.render_modes import depth_rgb, point_image
import test_knowledge


PROJECTION = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                           [0., 0., 1., 1.], [0., 0., 0., 0.]])


class RenderModesTest(unittest.TestCase):
    def test_points_use_nearest_depth_and_reject_invalid_centers(self):
        xyz = torch.tensor([[0., 0., 2.], [0., 0., 1.], [.8, 0., 1.],
                            [0., 0., -1.], [float('nan'), 0., 1.], [-4., 0., 1.]])
        colors = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.],
                               [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
        image = point_image(xyz, colors, PROJECTION, 5, 5)
        self.assertEqual(image[:, 2, 2].tolist(), [1., 0., 0.])
        self.assertEqual(image[:, 2, 4].tolist(), [0., 0., 1.])
        self.assertEqual(int((image.sum(0) > 0).sum()), 2)
        empty = point_image(xyz[:0], colors[:0], PROJECTION, 5, 5)
        self.assertFalse(bool(empty.any()))
        tie = point_image(xyz[1:2].repeat(2, 1), colors[:2], PROJECTION, 5, 5)
        self.assertEqual(tie[:, 2, 2].tolist(), [0., 1., 0.])

    def test_depth_invalid_pixels_constant_range_and_near_far_colors(self):
        values = np.array([[0., np.nan, -1.], [1., 2., 4.]])
        image = depth_rgb(values)
        self.assertEqual(image.shape, (2, 3, 3))
        self.assertFalse(image[0].any())
        self.assertFalse(np.array_equal(image[1, 0], image[1, 2]))
        self.assertTrue(depth_rgb(torch.ones(1, 1, 1)).any())
        self.assertFalse(depth_rgb(np.zeros((2, 2))).any())

    def test_engine_modes_keep_mask_selection_camera_and_labels(self):
        with TemporaryDirectory() as folder, patch.object(torch.Tensor, 'cuda', lambda t: t):
            e = test_knowledge.KnowledgeTest.engine(self, folder)
            e.model.get_xyz = torch.tensor([[0., 0., 1.], [.5, 0., 1.], [0., .5, 1.]])
            e.model.get_features = torch.ones(3, 1, 3) * .2
            e.model.active_sh_degree = 0
            e.selection = torch.tensor([True, False, False])
            e.view.update(size=64, aspect=1.)
            camera_before = dict(e.view)
            labels_before = e.labels().copy()
            e.camera = lambda: SimpleNamespace(camera_center=torch.zeros(3), full_proj_transform=PROJECTION)
            e.visualization = lambda cam: {}
            images = []
            e.image_url = lambda image: images.append(image) or 'image'
            e.render_simple = Mock(return_value={'render': torch.ones(3, 64, 64) * .3,
                                                  'depth': torch.ones(1, 64, 64) * 2})
            e.frame = Engine.frame.__get__(e)
            for style in ('splats', 'depth', 'points', 'rgb'):
                e.render_simple.reset_mock()
                result = e.execute('frame', {'render_mode': style, 'mode': 'foreground'}, None, None)
                self.assertEqual(result['render_mode'], style)
                if style == 'points':
                    e.render_simple.assert_not_called()
                    self.assertEqual(int((images[-1].sum(2) > 0).sum()), 1)
                else:
                    options = e.render_simple.call_args.kwargs
                    self.assertEqual(options['extra_gau_mask'].tolist(), [True, False, False])
                    self.assertEqual(options.get('show_gau', False), style == 'splats')
                    if style == 'splats':
                        self.assertEqual(options['scaling_modifier'], .35)
                self.assertEqual(images[-1].shape, (64, 64, 3))
                np.testing.assert_array_equal(e.labels(), labels_before)
                self.assertEqual(e.selection.tolist(), [True, False, False])
                self.assertEqual(e.view, camera_before)
            for value in ('bad', None, [], 1):
                with self.assertRaises(ValueError):
                    e.execute('frame', {'render_mode': value}, None, None)
            e.capture = {'id': 'frozen'}
            with self.assertRaisesRegex(ValueError, '退出分割提示'):
                e.execute('frame', {'render_mode': 'points'}, None, None)
            e.capture = None
            e.render_mode = 'depth'
            e.background_color = [0.125, 0.25, 0.5]
            self.assertEqual(e.freeze()['render_mode'], 'rgb')
            self.assertNotIn('show_gau', e.render_simple.call_args.kwargs)
            torch.testing.assert_close(e.render_simple.call_args.kwargs['bg_color'], torch.tensor([0.125, 0.25, 0.5]))
            e.execute('frame', {'render_mode': 'rgb'}, None, None)
            torch.testing.assert_close(e.render_simple.call_args.kwargs['bg_color'], torch.tensor([0.125, 0.25, 0.5]))


    def test_background_color_validation_and_frozen_view(self):
        with TemporaryDirectory() as folder:
            e = Engine('unused.pt', folder)
            e.frame = lambda mode='all': e.snapshot()
            e.foreground_options = lambda data: None
            self.assertEqual(e.snapshot()['background_color'], [0, 0, 0])
            result = e.execute('frame', {'background_color': [0.125, 0.25, 0.5]}, None, None)
            self.assertEqual(result['background_color'], [0.125, 0.25, 0.5])
            for value in (None, [], [1, 2], [1, 2, 3, 4], [True, 0, 0], ['1', 0, 0],
                          [-1, 0, 0], [256, 0, 0], [1.5, 0, 0], [float('nan'), 0, 0], [float('inf'), 0, 0]):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    e.execute('frame', {'background_color': value}, None, None)
                self.assertEqual(e.background_color, [0.125, 0.25, 0.5])
            for field in ('capture', 'ground_capture'):
                setattr(e, field, {'id': 'locked'})
                with self.assertRaisesRegex(ValueError, 'Exit prompts'):
                    e.execute('frame', {'background_color': [0, 0, 0]}, None, None)
                self.assertEqual(e.execute('frame', {'background_color': [0.125, 0.25, 0.5]}, None, None)['background_color'], [0.125, 0.25, 0.5])
                setattr(e, field, None)

    def test_custom_background_preserves_points_and_depth_colors(self):
        rgb = [0.125, 0.25, 0.5]
        xyz = torch.tensor([[0., 0., 1.]])
        colors = torch.tensor([[1., 0., 0.]])
        image = point_image(xyz, colors, PROJECTION, 5, 5, rgb)
        torch.testing.assert_close(image[:, 0, 0], torch.tensor(rgb))
        torch.testing.assert_close(image[:, 2, 2], colors[0])
        empty = point_image(xyz[:0], colors[:0], PROJECTION, 5, 5, rgb)
        torch.testing.assert_close(empty, (torch.tensor(rgb))[:, None, None].expand(3, 5, 5))
        depth = np.array([[0., np.nan], [1., 4.]])
        custom, original = depth_rgb(depth, rgb), depth_rgb(depth)
        np.testing.assert_array_equal(custom[0], [[32, 64, 128], [32, 64, 128]])
        np.testing.assert_array_equal(custom[1], original[1])
        white = point_image(xyz[:0], colors[:0], PROJECTION, 5, 5, [1, 1, 1])
        self.assertTrue(bool((white == 1).all()))
        self.assertTrue(bool((depth_rgb(np.zeros((2, 2)), [1, 1, 1]) == 255).all()))
