import tempfile
import io
from unittest.mock import patch
import threading
import time
import unittest
from pathlib import Path
from backend.server import create_app


class FakeEngine:
    def __init__(self, root):
        self.checkpoint = Path('test.pt')
        self.output = Path(root)
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, action, data, progress, cancel):
        if data.get('wait'):
            self.started.set()
            self.release.wait(3)
        if data.get('fail'):
            raise ValueError('expected failure')
        progress(1, 'test progress')
        return {'action': action, 'cancelRequested': cancel.is_set()}


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = FakeEngine(self.tmp.name)
        self.app = create_app('test.pt', engine=self.engine)
        self.client = self.app.test_client()
        self.token = self.client.get('/api/config').json['token']
        self.headers = {'X-SAGO-Token':self.token}

    def tearDown(self):
        self.engine.release.set()
        self.app.extensions['sago_worker'].shutdown(wait=True)
        self.tmp.cleanup()

    def post(self, action='frame', data=None):
        return self.client.post('/api/jobs', json={'action':action,'data':data or {}}, headers=self.headers)

    def result(self, job_id):
        for _ in range(100):
            job = self.client.get('/api/jobs/'+job_id).json
            if job['status'] not in ('running','queued'):
                return job
            time.sleep(.01)
        self.fail('job timed out')

    def test_asset_downloads_allow_final_files_only(self):
        folder=self.engine.output/('assets-'+'c'*32);obj=folder/'ID_2';obj.mkdir(parents=True)
        for name in ('gaussians.ply','point_cloud.ply','ID_2_Duck.obj'):
            (obj/name).write_text('test')
            response=self.client.get('/downloads/'+obj.relative_to(self.engine.output).as_posix()+'/'+name)
            self.assertEqual(response.status_code,200);response.close()
        (folder/'assets.zip').write_bytes(b'zip')
        response=self.client.get('/downloads/'+folder.name+'/assets.zip',query_string={'filename':'SAGO_assets.zip'})
        self.assertEqual(response.status_code,200);self.assertIn('SAGO_assets.zip',response.headers['Content-Disposition']);response.close()
        for path in ('manifest.json','.work/ID_2/runtime/raw.obj','ID_2/secret.ply','../outside.zip'):
            self.assertEqual(self.client.get('/downloads/'+folder.name+'/'+path).status_code,404)

    def test_nested_mesh_download_and_private_work_files(self):
        folder=self.engine.output/('mesh-'+'a'*32)/'ID_3';folder.mkdir(parents=True)
        (folder/'ID_3_Duck.obj').write_text('v 0 0 0\nf 1 1 1\n')
        route='/downloads/'+folder.relative_to(self.engine.output).as_posix()
        response=self.client.get(route+'/ID_3_Duck.obj',query_string={'filename':'Duck.obj'})
        self.assertEqual(response.status_code,200);self.assertIn('Duck.obj',response.headers['Content-Disposition']);response.close()
        for name in ('train.log','manifest.json','../secret.obj','raw.obj/../../secret.obj'):
            self.assertEqual(self.client.get(route+'/'+name).status_code,404)
        submitted=self.post('mesh',{'ids':['duck']})
        self.assertEqual(submitted.status_code,202)
        result=self.result(submitted.json['id'])
        self.assertEqual(result['result']['action'],'mesh')

    def test_semantic_job_is_serialized_and_returns_results(self):
        response=self.post('semantic', {'operation':'search','text':'a chair'})
        self.assertEqual(response.status_code,202)
        self.assertEqual(self.result(response.json['id'])['result']['action'],'semantic')

    def test_rag_action(self):
        response=self.post('rag', {'operation':'test'})
        self.assertEqual(response.status_code,202)
        self.assertEqual(self.result(response.json['id'])['result']['action'],'rag')

    def test_gif_capability(self):
        self.assertEqual(self.client.get('/api/config').json['gif_export_version'], 3)

    def test_ui_and_assets(self):
        for path in ('/','/assets/app.js','/assets/style.css'):
            response = self.client.get(path)
            self.assertEqual(response.status_code,200)
            response.close()
        self.assertEqual(self.client.get('/assets/../../run.py').status_code,404)

    def test_session_and_host(self):
        self.assertEqual(self.client.post('/api/jobs',json={'action':'frame'}).status_code,403)
        self.assertEqual(self.client.get('/api/config',headers={'Host':'evil.example'}).status_code,403)
        self.assertEqual(self.post('invalid').status_code,400)
        self.assertEqual(self.client.post('/api/jobs',json=[],headers=self.headers).status_code,400)

    def test_serialized_cancel(self):
        first = self.post(data={'wait':True})
        self.assertEqual(first.status_code,202)
        self.assertTrue(self.engine.started.wait(2))
        self.assertEqual(self.post().status_code,409)
        job_id=first.json['id']
        self.client.post('/api/jobs/'+job_id+'/cancel',json={},headers=self.headers)
        self.engine.release.set()
        result=self.result(job_id)
        self.assertEqual(result['status'],'done')
        self.assertTrue(result['result']['cancelRequested'])
        self.assertNotIn('_cancel',result)

    def test_explicit_model_load_job(self):
        job = self.result(self.post('load_model', {'model': 'sam3'}).json['id'])
        self.assertEqual(job['status'], 'done')
        self.assertEqual(job['result']['action'], 'load_model')

    def test_failure_and_recovery(self):
        self.assertEqual(self.result(self.post(data={'fail':True}).json['id'])['status'],'error')
        self.assertEqual(self.result(self.post().json['id'])['status'],'done')

    def test_scene_upload_and_reuse(self):
        payload=b'ply\nformat ascii 1.0\nelement vertex 0\nend_header\n'
        def upload(name='scene.ply', body=payload):
            return self.client.post('/api/scenes/upload', data={'file':(io.BytesIO(body),name)}, headers=self.headers)
        first=upload('../../scene.ply')
        self.assertEqual(first.status_code,200)
        path=Path(first.json['path'])
        self.assertTrue(path.is_relative_to(self.engine.output/'uploads'))
        self.assertEqual(path.read_bytes(),payload)
        self.assertEqual(upload('../../scene.ply').json['path'],str(path))
        self.assertEqual(upload('other.txt').status_code,400)
        self.assertEqual(upload('bad.ply',b'not ply').status_code,400)
        self.assertEqual(upload('empty.ply',b'').status_code,400)
        self.assertFalse(list((self.engine.output/'uploads').glob('*.part')))
        self.assertEqual(self.client.post('/api/scenes/upload',data={'file':(io.BytesIO(payload),'a.ply')}).status_code,403)

    def test_native_picker_job_and_cancel(self):
        with patch('backend.server.file_picker.pick_scene',return_value={'path':'/tmp/scene.ply'}) as picker:
            job=self.result(self.post('pick',{'initial':'/tmp'}).json['id'])
            self.assertEqual(job['result']['path'],'/tmp/scene.ply')
            picker.assert_called_once_with('/tmp')
        with patch('backend.server.file_picker.pick_scene',return_value={'cancelled':True}):
            job=self.result(self.post('pick').json['id'])
            self.assertTrue(job['result']['cancelled'])

    def test_preview_assets_confined(self):
        run_id='a'*32
        folder=self.engine.output/'drone-previews'/run_id
        folder.mkdir(parents=True)
        (folder/'0-render.png').write_bytes(b'preview-test')
        response=self.client.get(f'/previews/{run_id}/0-render.png')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.data,b'preview-test')
        response.close()
        self.assertEqual(self.client.get(f'/previews/{run_id}/other.png').status_code,404)
        self.assertEqual(self.client.get('/previews/invalid/0-render.png').status_code,404)

    def test_gif_job_and_download(self):
        job = self.result(self.post('gif', {'size': 256}).json['id'])
        self.assertEqual(job['status'], 'done')
        self.assertEqual(job['result']['action'], 'gif')
        filename = 'b'*32 + '.gif'
        (self.engine.output/filename).write_bytes(b'GIF89a')
        response = self.client.get('/downloads/' + filename)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'image/gif')
        self.assertEqual(response.data, b'GIF89a')
        response.close()
        response = self.client.get('/downloads/' + filename, query_string={'filename':'ID_2_椅子.gif'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('ID_2_', response.headers['Content-Disposition'])
        self.assertIn("filename*=UTF-8''", response.headers['Content-Disposition'])
        response.close()
        response = self.client.get('/downloads/' + filename, query_string={'filename':'../bad.gif'})
        self.assertIn(filename, response.headers['Content-Disposition'])
        response.close()
        self.assertEqual(self.client.get('/downloads/not-an-export.gif').status_code, 404)

    def test_exports_confined(self):
        self.assertEqual(self.client.get('/downloads/run.py').status_code,404)
        self.assertEqual(self.client.get('/downloads/'+'a'*32+'.pt').status_code,404)
        filename='a'*32+'.ply'
        (self.engine.output/filename).write_bytes(b'ply')
        response=self.client.get('/downloads/'+filename)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.data,b'ply')
        response.close()


if __name__ == '__main__':
    unittest.main()
