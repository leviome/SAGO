"""Local HTTP UI and a serialized, cancellable GPU job queue."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import secrets
import hashlib
import os
import threading
import traceback
import uuid

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from backend import file_picker

from backend.engine import Engine, Cancelled

ROOT = Path(__file__).resolve().parent.parent
ACTIONS = {'ground', 'load_model', 'load', 'view', 'frame', 'capture', 'preview', 'apply', 'objects', 'export', 'pick'}


def create_app(checkpoint, output=None, engine=None):
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024**3
    engine = engine or Engine(checkpoint, output or ROOT/'outputs')
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sago-gpu')
    lock = threading.Lock()
    jobs = {}
    token = secrets.token_urlsafe(32)
    app.extensions['sago_worker'] = worker
    app.extensions['sago_engine'] = engine

    @app.before_request
    def local_access():
        # Reject cross-origin writes and DNS rebinding against the local GPU server.
        if request.path != '/api/scenes/upload' and (request.content_length or 0) > 64 * 1024:
            return jsonify(error='请求过大'), 413
        host = request.host.split(':')[0].lower()
        if host not in ('localhost', '127.0.0.1', '[', '::1'):
            return jsonify(error='只允许通过 localhost 或 127.0.0.1 访问'), 403
        if request.method == 'POST' and not secrets.compare_digest(request.headers.get('X-SAGO-Token', ''), token):
            return jsonify(error='页面会话已失效，请刷新'), 403

    @app.get('/')
    def index():
        return send_from_directory(ROOT/'frontend', 'index.html')

    @app.get('/assets/<path:name>')
    def assets(name):
        return send_from_directory(ROOT/'frontend', name)

    @app.get('/api/config')
    def config():
        from backend.sam3_client import checkpoint_paths
        paths = checkpoint_paths()
        return jsonify(token=token, checkpoint=str(engine.checkpoint),
            models=[dict(id='sam2', checkpoint=str(engine.checkpoint), available=engine.checkpoint.is_file())]+
                   [dict(id=k, checkpoint=str(v), available=v.is_file()) for k, v in paths.items()],
            architecture='Python CUDA / SAM2 / SAM3 / SAM3.1 / Virtual Drone', nativePicker=file_picker.available(),
            scene_loaded=getattr(engine, 'model', None) is not None)

    def run(job_id, action, data, cancel):
        def progress(done, message, views=None):
            with lock:
                jobs[job_id].update(done=done, message=message)
                if views is not None:
                    jobs[job_id]['views'] = views
        try:
            with lock:
                jobs[job_id]['status'] = 'running'
            if cancel.is_set():
                raise Cancelled('任务已取消')
            if action == 'pick':
                progress(0, '请在系统文件选择框中选择场景')
                result = file_picker.pick_scene(data.get('initial', ''))
            else:
                result = engine.execute(action, data, progress, cancel)
            with lock:
                jobs[job_id].update(status='done', result=result, message='完成')
        except Exception as exc:
            if not isinstance(exc, (ValueError, Cancelled)):
                traceback.print_exc()
            with lock:
                jobs[job_id].update(status='cancelled' if isinstance(exc, Cancelled) else 'error', error=str(exc))
        finally:
            with lock:
                jobs[job_id].pop('_cancel', None)

    @app.post('/api/jobs')
    def submit():
        body = request.get_json()
        if not isinstance(body, dict) or body.get('action') not in ACTIONS or not isinstance(body.get('data', {}), dict):
            return jsonify(error='无效操作或请求格式'), 400
        with lock:
            if any(j['status'] in ('queued', 'running') for j in jobs.values()):
                return jsonify(error='后端正在处理任务，请稍候'), 409
            # Retain only a small number of potentially large image results.
            while len(jobs) >= 8:
                del jobs[next(iter(jobs))]
            job_id = uuid.uuid4().hex
            cancel = threading.Event()
            jobs[job_id] = dict(id=job_id, status='queued', done=0, message='等待 GPU', _cancel=cancel)
        worker.submit(run, job_id, body['action'], body.get('data', {}), cancel)
        return jsonify(id=job_id), 202

    @app.get('/api/jobs/<job_id>')
    def poll(job_id):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return jsonify(error='任务不存在或已过期'), 404
            return jsonify({k: v for k, v in job.items() if not k.startswith('_')})

    @app.post('/api/jobs/<job_id>/cancel')
    def cancel_job(job_id):
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return jsonify(error='任务不存在'), 404
            if '_cancel' in job:
                job['_cancel'].set()
        return jsonify(ok=True)

    @app.post('/api/scenes/upload')
    def upload_scene():
        uploaded = request.files.get('file')
        if uploaded is None or Path(uploaded.filename or '').suffix.lower() != '.ply':
            return jsonify(error='请选择一个 PLY 场景文件'), 400
        folder = engine.output / 'uploads'
        folder.mkdir(parents=True, exist_ok=True)
        temporary = folder / (uuid.uuid4().hex + '.part')
        digest = hashlib.sha256()
        try:
            with temporary.open('wb') as target:
                first = True
                for chunk in iter(lambda: uploaded.stream.read(1024 * 1024), b''):
                    if first and not chunk.startswith(b'ply'):
                        return jsonify(error='文件不是有效的 PLY'), 400
                    first = False
                    digest.update(chunk)
                    target.write(chunk)
            if first:
                return jsonify(error='文件为空'), 400
            # A content directory prevents collisions and reuses repeated drops.
            destination = folder / digest.hexdigest()
            destination.mkdir(exist_ok=True)
            name = secure_filename(uploaded.filename) or 'scene.ply'
            if not name.lower().endswith('.ply'):
                name = 'scene.ply'
            path = destination / name
            os.replace(temporary, path)
            return jsonify(path=str(path.resolve()), name=uploaded.filename)
        finally:
            temporary.unlink(missing_ok=True)

    @app.get('/previews/<run_id>/<name>')
    def preview_image(run_id, name):
        if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
            return jsonify(error='图像不存在'), 404
        if name not in {f'{i}-{kind}.png' for i in range(6) for kind in ('render', 'overlay', 'mask')}:
            return jsonify(error='图像不存在'), 404
        return send_from_directory(engine.output/'drone-previews'/run_id, name)

    @app.get('/downloads/<name>')
    def download(name):
        # Only generated exports, never arbitrary server paths.
        if len(name.split('.')[0]) != 32 or any(c not in '0123456789abcdef' for c in name.split('.')[0]):
            return jsonify(error='文件不存在'), 404
        if Path(name).suffix not in ('.ply', '.npy', '.png'):
            return jsonify(error='文件不存在'), 404
        return send_from_directory(engine.output, name, as_attachment=True)

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        app.logger.exception('request failed')
        return jsonify(error=str(exc)), 500

    return app
