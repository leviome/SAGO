"""First-use model delivery without GPU or large network transfers."""
import hashlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.error import URLError
from http.client import IncompleteRead
from desktop.checkpoints import CheckpointSpec, DownloadCancelled, download_checkpoint, resolve_checkpoint
from desktop.engine import DesktopEngine
from backend.engine import Engine

PAYLOAD=b'checkpoint-test-data'*100
SPEC=CheckpointSpec('https://example.invalid/sam2.pt',len(PAYLOAD),hashlib.sha256(PAYLOAD).hexdigest())

class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data);self.status=status
        self.headers=headers if headers is not None else {'Content-Length':str(len(data))}

class CheckpointDownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.target=Path(self.temp.name)/'cache/model.pt'
    def download(self,opener=None,**kwargs):
        return download_checkpoint(self.target,spec=SPEC,opener=opener or (lambda *a,**k:Response(PAYLOAD)),**kwargs)
    def partial(self,data):
        self.target.parent.mkdir(parents=True,exist_ok=True)
        path=self.target.with_suffix('.pt.part');path.write_bytes(data);return path
    def test_first_download_verified_and_atomic_cache_reuse(self):
        events=[];self.download(progress=lambda *v:events.append(v))
        self.assertEqual(self.target.read_bytes(),PAYLOAD)
        self.assertFalse(self.target.with_suffix('.pt.part').exists())
        opener=Mock(side_effect=AssertionError('cache hit must not use network'))
        self.download(opener)
        opener.assert_not_called();self.assertEqual(events[-1][0],100)
    def test_resume_range(self):
        self.partial(PAYLOAD[:400]);requests=[]
        def opener(request,**kwargs):
            requests.append(request.get_header('Range'))
            return Response(PAYLOAD[400:],206,{'Content-Range':f'bytes 400-{len(PAYLOAD)-1}/{len(PAYLOAD)}'})
        self.download(opener);self.assertEqual(requests,['bytes=400-'])
        self.assertEqual(self.target.read_bytes(),PAYLOAD)
    def test_range_ignored_restarts_cleanly(self):
        self.partial(PAYLOAD[:400]);self.download()
        self.assertEqual(self.target.read_bytes(),PAYLOAD)
    def test_invalid_range_is_rejected(self):
        self.partial(PAYLOAD[:400])
        with self.assertRaisesRegex(ValueError,'range'):
            self.download(lambda *a,**k:Response(PAYLOAD,206,{'Content-Range':f'bytes 0-{len(PAYLOAD)-1}/{len(PAYLOAD)}'}))
        self.assertFalse(self.target.exists())
    def test_invalid_length_rejected(self):
        with self.assertRaisesRegex(ValueError,'size'):
            self.download(lambda *a,**k:Response(b'wrong'))
        self.assertFalse(self.target.exists())
    def test_checksum_failure_never_publishes(self):
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.download(lambda *a,**k:Response(b'x'*len(PAYLOAD)))
        self.assertFalse(self.target.exists());self.assertFalse(self.target.with_suffix('.pt.part').exists())
    def test_corrupt_managed_cache_repaired(self):
        self.target.parent.mkdir(parents=True);self.target.write_bytes(b'x'*len(PAYLOAD))
        self.download();self.assertEqual(self.target.read_bytes(),PAYLOAD)
    def test_complete_partial_verified_without_network(self):
        self.partial(PAYLOAD);opener=Mock(side_effect=AssertionError('already downloaded'))
        self.download(opener);opener.assert_not_called()
    def test_network_failure_retries_and_keeps_partial(self):
        self.partial(PAYLOAD[:400]);opener=Mock(side_effect=URLError('offline'))
        with patch('desktop.checkpoints.threading.Event') as event:
            event.return_value.is_set.return_value=False
            with self.assertRaisesRegex(ValueError,'Retry'):
                self.download(opener)
        self.assertEqual(opener.call_count,3)
        self.assertEqual(self.target.with_suffix('.pt.part').read_bytes(),PAYLOAD[:400])
    def test_incomplete_http_read_is_retried(self):
        response=Response(PAYLOAD);response.read=Mock(side_effect=IncompleteRead(b'partial'))
        opener=Mock(side_effect=[response,Response(PAYLOAD)])
        with patch('desktop.checkpoints.threading.Event') as event:
            event.return_value.is_set.return_value=False
            self.download(opener)
        self.assertEqual(opener.call_count,2)
    def test_cancel_keeps_partial_and_does_not_publish(self):
        cancel=threading.Event()
        def progress(percent,stage,details):
            if details['downloaded']:cancel.set()
        with self.assertRaises(DownloadCancelled):self.download(progress=progress,cancel=cancel)
        self.assertFalse(self.target.exists());self.assertTrue(self.target.with_suffix('.pt.part').exists())
    def test_custom_override_and_missing_file(self):
        self.target.parent.mkdir(parents=True);self.target.write_bytes(b'custom checkpoint')
        with patch('desktop.checkpoints.download_checkpoint') as download:
            self.assertEqual(resolve_checkpoint(self.target),self.target)
            with self.assertRaisesRegex(ValueError,'not found'):resolve_checkpoint(self.target.parent/'missing.pt')
            download.assert_not_called()
    def test_default_and_missing_legacy_download(self):
        with patch('desktop.checkpoints.DEFAULT_CHECKPOINT',self.target),patch('desktop.checkpoints.ROOT',Path(self.temp.name)),patch('desktop.checkpoints.download_checkpoint',return_value=self.target) as download:
            self.assertEqual(resolve_checkpoint(self.target),self.target)
            self.assertEqual(resolve_checkpoint(Path(self.temp.name)/'models/sam2_hiera_large.pt'),self.target)
            self.assertEqual(download.call_count,2)

class DesktopDownloadIntegrationTest(unittest.TestCase):
    def test_download_precedes_gpu_loading_and_reuse_skips_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=DesktopEngine('missing.pt',folder);engine._checkpoint_progress=Mock()
            checkpoint=Path(folder)/'cache.pt'
            with patch('desktop.checkpoints.resolve_checkpoint',return_value=checkpoint) as resolve,patch.object(Engine,'load_model',return_value={'ok':True}) as load:
                self.assertEqual(engine.load_model({'model':'sam2'}),{'ok':True})
                self.assertEqual(engine.checkpoint,checkpoint);resolve.assert_called_once()
                load.assert_called_once();engine._checkpoint_progress.assert_called_once()
                engine.tracker=object();engine.load_model({'model':'sam2'})
                resolve.assert_called_once()
    def test_invalid_model_and_text_do_not_download(self):
        with tempfile.TemporaryDirectory() as folder,patch('desktop.checkpoints.resolve_checkpoint') as resolve:
            engine=DesktopEngine('missing.pt',folder)
            for data in ({'model':'sam3'},{'model':'sam2','text':'chair'}):
                with self.assertRaises(ValueError):engine.load_model(data)
            resolve.assert_not_called()
