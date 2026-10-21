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
            client = Mock()
            client.load.side_effect = [1., 0., 1.]
            with patch('backend.sam3_client.Sam3Client', return_value=client):
                first = engine.load_model({'model': 'sam2'})
                self.assertFalse(first['reused'])
                self.assertIsNone(engine.capture['encoded'])
                self.assertIsNone(engine.model)
                second = engine.load_model({'model': 'sam2'})
                self.assertTrue(second['reused'])
                self.assertEqual(second['model_load_seconds'], 0.)
                engine.CrossMatch.assert_called_once()
                engine.capture['encoded'] = object()
                self.assertFalse(engine.load_model({'model': 'sam3'})['reused'])
                self.assertIsNone(engine.tracker)
                self.assertIsNone(engine.capture['encoded'])
                self.assertTrue(engine.load_model({'model': 'sam3'})['reused'])
                self.assertFalse(engine.load_model({'model': 'sam3.1'})['reused'])
                engine.load_model({'model': 'sam2'})
                client.close.assert_called_once()
                self.assertEqual(engine.CrossMatch.call_count, 2)

    def test_invalid_model_does_not_initialize_gpu(self):
        with TemporaryDirectory() as folder:
            engine = Engine('missing.pt', folder)
            engine.initialize = Mock()
            with self.assertRaises(ValueError):
                engine.load_model({'model': 'other'})
            engine.initialize.assert_not_called()
