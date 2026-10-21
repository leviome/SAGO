"""Prompt captures use the current foreground while ground calibration stays independent."""
import tempfile
from types import SimpleNamespace
import unittest

import torch

from backend.engine import Engine


class CaptureScopeTest(unittest.TestCase):
    def test_prompt_rgb_and_depth_are_scoped_to_current_foreground(self):
        with tempfile.TemporaryDirectory() as root:
            engine = Engine('unused.pt', root)
            engine.torch = torch
            engine.model = SimpleNamespace(get_xyz=torch.zeros((4, 3)))
            engine.camera = lambda: None
            engine.snapshot = lambda: {}
            engine.visualization = lambda cam: {}
            engine.image_url = lambda rgb: str(rgb[0, 0, 0])
            engine.active = lambda: torch.tensor([True, True, True, False])
            rendered_scopes = []

            def render(cam, model, extra_gau_mask, **kwargs):
                rendered_scopes.append(extra_gau_mask.clone())
                count = float(extra_gau_mask.sum())
                return {'render': torch.full((3, 2, 2), count / 4),
                        'depth': torch.full((2, 2), count)}

            engine.render_simple = render
            initial = engine.execute('capture', {}, None, None)
            self.assertEqual(initial['capture_mode'], 'all')
            self.assertEqual(rendered_scopes[-1].tolist(), [True, True, True, False])
            engine.selection = torch.tensor([False, True, False, True])
            refined = engine.execute('capture', {}, None, None)
            self.assertEqual(refined['capture_mode'], 'foreground')
            self.assertEqual(rendered_scopes[-1].tolist(), [False, True, False, False])
            self.assertNotEqual(initial['image'], refined['image'])
            self.assertTrue(bool((engine.capture['depth'] == 1).all()))
            self.assertEqual(engine.capture['gaussian_mask'].tolist(), [False, True, False, False])
            # Ground calibration uses the full active scene, independently of foreground.
            ground = engine.freeze()
            self.assertEqual(ground['capture_mode'], 'all')
            self.assertEqual(rendered_scopes[-1].tolist(), [True, True, True, False])
            engine.selection.zero_()
            before = len(rendered_scopes)
            with self.assertRaisesRegex(ValueError, '当前前景为空'):
                engine.execute('capture', {}, None, None)
            self.assertEqual(len(rendered_scopes), before)
            engine.selection = None
            self.assertEqual(engine.execute('capture', {}, None, None)['capture_mode'], 'all')

    def test_removed_single_view_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            engine = Engine('unused.pt', root)
            for data in ({}, {'drone': False}):
                with self.assertRaisesRegex(ValueError, '单视角应用已移除'):
                    engine.apply(data, None, None)


if __name__ == '__main__':
    unittest.main()
