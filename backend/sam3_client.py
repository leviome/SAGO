"""Serialized subprocess bridge; keeps SAM2/CUDA renderer dependencies unchanged."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def checkpoint_paths():
    defaults = {
        'sam3': Path('/home/pc/haptic/sam-3d-objects/sam3_ckpts/sam3/sam3.pt'),
        'sam3.1': Path('/home/pc/haptic/sam-3d-objects/sam3_ckpts/sam3.1/sam3.1_multiplex.pt'),
    }
    result = {}
    for version, fallback in defaults.items():
        local = ROOT/'models'/fallback.name
        key = 'SAGO_SAM3_CHECKPOINT' if version == 'sam3' else 'SAGO_SAM31_CHECKPOINT'
        result[version] = Path(os.environ.get(key, str(local if local.is_file() else fallback))).expanduser()
    return result


class Sam3Client:
    def __init__(self, output):
        self.output = Path(output)
        self.process = None
        self.version = None

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        self.version = None

    def request(self, data):
        proc = self.process
        if proc is None or proc.poll() is not None:
            raise RuntimeError('SAM3 推理进程已退出，请检查 outputs/sam3-worker.log')
        proc.stdin.write(json.dumps(data)+'\n')
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], 600)
        if not ready:
            self.close()
            raise RuntimeError('SAM3 推理超时，进程已停止')
        line = proc.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError('SAM3 推理进程异常退出，请检查 outputs/sam3-worker.log')
        response = json.loads(line)
        if 'error' in response:
            raise ValueError(response['error'])
        return response['result']

    def load(self, version):
        if self.version == version and self.process is not None and self.process.poll() is None:
            return 0.
        self.close()
        checkpoint = checkpoint_paths()[version]
        if not checkpoint.is_file():
            raise ValueError(f'未找到 {version.upper()} 权重：{checkpoint}')
        python = Path(os.environ.get('SAGO_SAM3_PYTHON', str(ROOT/'.venv-sam3/bin/python')))
        if not python.is_file():
            raise ValueError('SAM3 运行环境未安装，请按 README 配置 SAGO_SAM3_PYTHON')
        env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                   MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'), PYTHONUNBUFFERED='1')
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output/'sam3-worker.log').open('a') as log:
            self.process = subprocess.Popen([str(python), str(ROOT/'backend/sam3_worker.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                cwd=ROOT, env=env)
        try:
            response = self.request(dict(action='load', version=version, checkpoint=str(checkpoint)))
        except Exception:
            self.close()
            raise
        self.version = version
        return response['model_load_seconds']

    def infer(self, frames, prompts):
        # Request paths are generated here, never accepted from the browser.
        with tempfile.TemporaryDirectory(prefix='sam3-', dir=self.output) as folder:
            root = Path(folder)
            paths = []
            for i, frame in enumerate(frames):
                path = root/f'{i}.png'
                Image.fromarray(frame).save(path)
                paths.append(str(path))
            request = dict(action='preview', frames=paths,
                           prompts=prompts, output=str(root/'result.npz'))
            timing = self.request(request)
            with np.load(root/'result.npz', allow_pickle=False) as data:
                return data['masks'].copy(), data['scores'].copy(), timing

    def start_route(self, image, prompts, target):
        with tempfile.TemporaryDirectory(prefix='sam3-start-', dir=self.output) as folder:
            root = Path(folder)
            Image.fromarray(image).save(root/'initial.png')
            np.save(root/'target.npy', target)
            return self.request(dict(action='start_route', image=str(root/'initial.png'),
                                     target=str(root/'target.npy'), prompts=prompts))

    def step_route(self, route, side, image):
        with tempfile.TemporaryDirectory(prefix='sam3-step-', dir=self.output) as folder:
            root = Path(folder)
            Image.fromarray(image).save(root/'new.png')
            report = self.request(dict(action='step_route', route=route, side=side,
                                      image=str(root/'new.png'), output=str(root/'result.npz')))
            with np.load(root/'result.npz', allow_pickle=False) as output:
                return output['masks'].copy(), report

    def finish_step(self, route, side, accepted):
        return self.request(dict(action='finish_step', route=route, side=side, accepted=bool(accepted)))

    def close_route(self, route):
        try:
            self.request(dict(action='close_route', route=route))
        except Exception:
            # Cleanup must not mask cancellation/inference errors or turn a
            # successful selection commit into a reported failure.
            self.close()
