"""Selected Gaussian IDs -> orbit RGB/masks -> NeuS2 -> world-space OBJ.

Dataset preparation and mesh normalization follow SAGO's mega_use pipeline.
Only the selected ID is rendered; no interactive SAM memory is required.
"""
import json
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from urllib.parse import quote

import numpy as np
from PIL import Image

from backend.gif_export import orbit_geometry

ROOT = Path(__file__).resolve().parent.parent


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        from backend.engine import Cancelled
        raise Cancelled('Mesh export cancelled')


def settings(data):
    def integer(key, default, low, high):
        value = data.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'Invalid mesh setting: {key} ({low}–{high})')
        return value
    root = Path(data.get('neus2_root') or os.environ.get('SAGO_MESH_ROOT') or os.environ.get('SAGO_NEUS2_ROOT') or Path.home()/'NeuS2').expanduser().resolve()
    python = data.get('neus2_python') or os.environ.get('SAGO_MESH_PYTHON') or os.environ.get('SAGO_NEUS2_PYTHON')
    if not python:
        candidates = [root/'.venv/bin/python', Path.home()/'anaconda3/envs/neus2/bin/python',
                      Path.home()/'miniconda3/envs/neus2/bin/python']
        python = next((p for p in candidates if p.is_file()), None)
    if not python:
        raise ValueError('Set the mesh exporting Python executable or SAGO_MESH_PYTHON')
    python = Path(python).expanduser().resolve()
    network = Path(data.get('network') or 'workspace_mask002.json').expanduser()
    if not network.is_absolute():
        network = root/'configs/nerf'/network
    for path in (root/'utils/mlp_weights.txt', network, python):
        if not path.is_file():
            raise ValueError(f'Required mesh exporting file not found: {path}')
    return dict(neus2_root=root, python_executable=python, network=network.resolve(),
        steps=integer('steps', 15000, 100, 100000),
        marching_cubes_resolution=integer('resolution', 512, 64, 1024),
        size=integer('size', 512, 128, 1024), views=integer('views_per_orbit', 72, 8, 144),
        parallel_jobs=integer('parallel_jobs', 2, 1, 4))


def run_process(command, cwd, log_path, progress, cancel, steps=None):
    """Poll a child with bounded log reads; terminate its process group on cancel."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    check_cancel(cancel)
    with log_path.open('w') as output, log_path.open('r') as reader:
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True)
        pending = ''
        try:
            while True:
                check_cancel(cancel)
                pending += reader.read(65536)
                lines = pending.split('\n'); pending = lines.pop()[-65536:]
                for line in lines:
                    if steps and line.startswith('SAGO_MESH_PROGRESS '):
                        row = json.loads(line.split(' ', 1)[1])
                        value = min(100, max(0, round(row['step']/steps*100)))
                        message = 'Extracting mesh' if row.get('stage') == 'extract' else f"Mesh exporting {row['step']}/{steps}"
                        progress(value, message)
                if child.poll() is not None:
                    break
                if cancel is None:
                    time.sleep(.2)
                else:
                    cancel.wait(.2)
            check_cancel(cancel)
            if child.returncode:
                with log_path.open('rb') as log:
                    log.seek(max(0, log_path.stat().st_size-6000))
                    tail = log.read().decode(errors='replace')
                raise RuntimeError(f'Mesh exporting exited with code {child.returncode}. Log: {log_path}\n{tail}')
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()


def run_neus2(pipeline, progress, cancel):
    workspace = pipeline.output_dir/'runtime'
    (workspace/'utils').mkdir(parents=True, exist_ok=True)
    for source in (pipeline.neus2_root/'utils').glob('mlp_weights*.txt'):
        shutil.copyfile(source, workspace/'utils'/source.name)
    raw = workspace/'raw.obj'
    command = [str(pipeline.python_executable), str(ROOT/'scripts/neus2-mesh.py'),
        '--root', str(pipeline.neus2_root), '--scene', str(pipeline.transform_path),
        '--network', str(pipeline.network), '--steps', str(pipeline.steps),
        '--resolution', str(pipeline.marching_cubes_resolution), '--output', str(raw)]
    run_process(command, workspace, pipeline.log_path, progress, cancel, pipeline.steps)
    if not raw.is_file() or not raw.stat().st_size:
        raise ValueError('Mesh exporting did not produce a mesh')
    return raw


def render_dataset(engine, selection, folder, options, progress, cancel):
    (folder/'images').mkdir(parents=True)
    (folder/'masks').mkdir()
    cameras = []
    count = options['views']*4
    with engine.torch.no_grad():
        target = engine.utils.masking_for_fg(engine.model, selection.clone())
        center, up, distance = orbit_geometry(target.get_xyz.detach().cpu().numpy(),
            target.get_scaling.detach().cpu().numpy(), engine.view['up'], engine.view['fov'])
        center = engine.torch.tensor(center, dtype=engine.torch.float32)
        up = engine.torch.tensor(up, dtype=engine.torch.float32)
        silhouette_colors = engine.torch.ones_like(target.get_xyz)
        for elevation in (0, 20, 30, 60):
            for azimuth in range(options['views']):
                check_cancel(cancel)
                cam = engine.utils.sphere2params(distance,
                    math.radians(engine.view['yaw'])+2*math.pi*azimuth/options['views'],
                    math.radians(elevation), center, up, math.radians(engine.view['fov']),
                    width=options['size'], height=options['size'])
                cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                index = len(cameras)
                cam.image_name = f'{index:03d}.png'
                rendered = engine.render_simple(cam, target, bg='white')
                # The standard RGB rasterizer does not return alpha. Rendering all
                # selected splats white on black yields accumulated opacity directly.
                silhouette = engine.render_simple(cam, target, bg='black', override_color=silhouette_colors)
                alpha = silhouette['render'][0].detach().cpu().numpy()
                if alpha.shape != (options['size'], options['size']) or not np.isfinite(alpha).all():
                    raise ValueError('Invalid rendered mesh silhouette')
                mask = np.where(alpha >= .5, 255, 0).astype(np.uint8)
                if not mask.any():
                    raise ValueError('The selected object has an empty silhouette; check its opacity')
                Image.fromarray(engine.rgb(rendered['render'])).save(folder/'images'/cam.image_name)
                Image.fromarray(mask).save(folder/'masks'/cam.image_name)
                cameras.append(cam)
                progress(round(len(cameras)/count*100), f'Rendering mesh views {len(cameras)}/{count}')
        check_cancel(cancel)
        engine.utils.save_cams_to_colmap(cameras, target, out_dir=folder/'sparse/0')
    return count


def generate_mesh(engine, data, progress, cancel):
    from backend.engine import Cancelled
    from backend.mesh_distillation import MeshDistillation
    engine.require_scene()
    requested = data.get('ids')
    if not isinstance(requested, list) or not requested or any(not isinstance(key, str) for key in requested):
        raise ValueError('Select at least one foreground object ID for mesh export')
    by_id = {obj['id']: obj for obj in engine.objects}
    if any(key not in by_id for key in requested):
        raise ValueError('Object IDs changed; select the mesh objects again')
    objects = sorted((by_id[key] for key in set(requested)), key=lambda obj: obj['number'])
    options = settings(data)
    started = time.perf_counter()
    engine.output.mkdir(parents=True, exist_ok=True)
    # Preflight import before spending time on any multi-view rendering.
    progress(0, 'Checking mesh exporting environment')
    with tempfile.TemporaryDirectory(prefix='mesh-check-', dir=engine.output) as temporary:
        run_process([str(options['python_executable']), str(ROOT/'scripts/neus2-mesh.py'),
            '--root', str(options['neus2_root']), '--probe'], temporary, Path(temporary)/'probe.log', progress, cancel)
    batch = engine.output/('mesh-'+uuid.uuid4().hex)
    batch.mkdir()
    # Rendering stays on the engine thread. Each reconstruction worker owns an
    # independent external process and directory; no engine/GPU tensors cross threads.
    workers = min(options['parallel_jobs'], len(objects))
    values = [0.0]*len(objects)
    lock = threading.Lock()
    stopped = threading.Event()
    failures = []
    class BatchCancel:
        def is_set(self):
            return stopped.is_set() or (cancel is not None and cancel.is_set())
        def wait(self, timeout):
            stopped.wait(timeout)
            return self.is_set()
    batch_cancel = BatchCancel()

    def report(index, value, message):
        with lock:
            values[index] = max(values[index], min(100, value))
            progress(round(sum(values)/len(objects)), f"ID {objects[index]['number']} · {message}")

    def reconstruct(index, obj, pipeline, count, filename):
        try:
            check_cancel(batch_cancel)
            pipeline.prepare_dataset()
            check_cancel(batch_cancel)
            raw = run_neus2(pipeline, lambda value, message: report(index, 32+value*.64, message), batch_cancel)
            report(index, 97, 'Restoring scene coordinates and validating triangles')
            stats = pipeline.postprocess_mesh(raw)
            check_cancel(batch_cancel)
            if stats.vertices < 3 or stats.faces < 1:
                raise ValueError('Reconstruction produced no usable mesh')
            relative = stats.path.relative_to(engine.output).as_posix()
            item = dict(id=obj['id'], number=obj['number'], name=obj['name'], filename=filename,
                saved_path=str(stats.path.resolve()), download='/downloads/'+quote(relative, safe='/')+'?filename='+quote(filename, safe=''),
                vertices=stats.vertices, faces=stats.faces, views=count, coordinate_space='scene')
            report(index, 100, 'Mesh exported')
            return item
        except BaseException as error:
            with lock:
                failures.append(error)
            stopped.set()
            raise

    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='sago-mesh') as pool:
            futures = []
            try:
                for index, obj in enumerate(objects):
                    check_cancel(batch_cancel)
                    selection = engine.object_mask(obj['number'])
                    if not bool(selection.any()):
                        raise ValueError(f"Object ID {obj['number']} has no Gaussians")
                    stem = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', obj['name']).strip(' .')[:80] or 'Object'
                    filename = f"ID_{obj['number']}_{stem}.obj"
                    folder = batch/f"ID_{obj['number']}"
                    folder.mkdir()
                    report(index, 0, 'Preparing mesh views')
                    count = render_dataset(engine, selection, folder/'dataset', options,
                        lambda value, message, i=index: report(i, value*.3, message), batch_cancel)
                    pipeline = MeshDistillation(folder/'dataset', folder/'reconstruction',
                        name=f"ID_{obj['number']}", output_obj=folder/filename,
                        **{key:options[key] for key in ('neus2_root','python_executable','network','steps','marching_cubes_resolution')})
                    futures.append(pool.submit(reconstruct, index, obj, pipeline, count, filename))
                items = [future.result() for future in futures]
            except BaseException:
                stopped.set()
                for future in futures:
                    future.cancel()
                raise
        check_cancel(cancel)
        result = dict(items=items, generation_seconds=time.perf_counter()-started,
            output_directory=str(batch.resolve()), coordinate_space='scene', format='obj', parallel_jobs=workers)
        (batch/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        progress(100, 'Mesh export complete')
        return result
    except BaseException as error:
        # Never present a partial batch as a completed export. Keep logs/data for debugging.
        for path in batch.glob('ID_*/*.obj'):
            path.unlink(missing_ok=True)
        if failures and isinstance(error, Cancelled):
            raise failures[0]
        raise
