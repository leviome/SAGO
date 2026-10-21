#!/usr/bin/python3
"""Relocatable launcher. A CUDA Python runtime is selectable, never project-linked."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if any(arg in ('--help','-h','--version') for arg in sys.argv[1:]):
    os.execv(sys.executable,[sys.executable,str(ROOT/'run.py'),*sys.argv[1:]])
config=Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'sago-desktop/settings.json'
try:settings=json.loads(config.read_text())
except (OSError,ValueError):settings={}
candidates=[os.environ.get('SAGO_PYTHON'),settings.get('python'),str(ROOT/'.venv/bin/python'),
    str(ROOT/'.runtime/python/bin/python'),str(Path.home()/'.local/share/sago-desktop/venv/bin/python')]
# Discover a usable CUDA installation without referring to any SAGO source tree.
for base in (Path.home()/'anaconda3/envs',Path.home()/'miniconda3/envs'):
    if base.is_dir():candidates.extend(str(p) for p in sorted(base.glob('*/bin/python')))
candidates.extend([shutil.which('python3')])
for candidate in dict.fromkeys(candidates):
    if not candidate or not Path(candidate).is_file():continue
    try:
        probe=subprocess.run([candidate,'-c','import torch, cv2; from diff_gaussian_rasterization_depth import GaussianRasterizer; from diff_gaussian_rasterization_mask import GaussianRasterizer; assert torch.cuda.is_available()'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
        if probe.returncode:continue
        env=dict(os.environ)
        if (ROOT/'.runtime/lib').is_dir():
            env['LD_LIBRARY_PATH']=str(ROOT/'.runtime/lib')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        for key in ('PYTHONHOME','QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):env.pop(key,None)
        os.execve(candidate,[candidate,str(ROOT/'run.py'),*sys.argv[1:]],env)
    except (OSError,subprocess.TimeoutExpired):continue
message='SAGO Desktop needs a CUDA-enabled Python runtime. Run scripts/setup-runtime.sh --python /path/to/cuda/python, or set SAGO_PYTHON. See README.md.'
print(message,file=sys.stderr)
if shutil.which('zenity'):subprocess.run(['zenity','--error','--text',message])
raise SystemExit(1)
