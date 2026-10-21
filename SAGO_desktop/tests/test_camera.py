"""Regression: JSON integer coordinates must become floating point tensors."""
import tempfile
import types
import unittest
import torch
from backend.engine import Engine


class CameraTest(unittest.TestCase):
    def test_json_integer_up_vector(self):
        with tempfile.TemporaryDirectory() as root:
            engine = Engine('test.pt', root)
            engine.torch = torch
            engine.view = dict(radius=2, yaw=0, pitch=0, target=[0,0,0], up=[0,-1,0], fov=50, size=768)
            engine.view['aspect'] = 2
            captured = []
            dimensions = []
            def sphere(*args, **kwargs):
                captured.append(args)
                dimensions.append((kwargs["width"], kwargs["height"]))
                return types.SimpleNamespace(world_view_transform=torch.eye(4))
            engine.utils = types.SimpleNamespace(sphere2params=sphere)
            cam = engine.camera()
            self.assertEqual(dimensions[0], (768, 384))
            self.assertEqual(captured[0][3].dtype, torch.float32)
            self.assertEqual(captured[0][4].dtype, torch.float32)
            self.assertTrue(torch.equal(cam.camera_center, torch.zeros(3)))

    def test_dimensions_and_invalid_aspect(self):
        self.assertEqual(Engine.dimensions(dict(size=768, aspect=.5)), (384, 768))
        for aspect in (0, -1, float('nan'), float('inf'), 6):
            with self.assertRaises(ValueError):
                Engine.valid_aspect(aspect)


if __name__ == '__main__':
    unittest.main()
