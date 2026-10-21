"""Model switching without a scene, reuse, and stale tensor invalidation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch
from backend.engine import Engine


class ModelLoadingTest(TestCase):
    def test_switch_reuse_and_cache_release(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'sam.pt'
            checkpoint.touch()
            engine = Engine(checkpoint, folder)
            engine.torch = Mock()
            engine.CrossMatch = Mock()
            engine.capture = dict(encoded=object(), masks=object())
            first = engine.load_model({'model': 'sam2'})
            self.assertFalse(first['reused'])
            self.assertIsNone(engine.capture['encoded'])
            self.assertIsNone(engine.model)
            second = engine.load_model({'model': 'sam2'})
            self.assertTrue(second['reused'])
            self.assertEqual(second['model_load_seconds'], 0.)
            engine.CrossMatch.assert_called_once()

    def test_invalid_model_does_not_initialize_gpu(self):
        with TemporaryDirectory() as folder:
            engine = Engine('missing.pt', folder)
            engine.initialize = Mock()
            for model in ('other','sam3','sam3.1'):
                with self.assertRaises(ValueError):
                    engine.load_model({'model': model})
            with self.assertRaises(ValueError):
                engine.load_model({'model':'sam2','text':'chair'})
            engine.initialize.assert_not_called()
