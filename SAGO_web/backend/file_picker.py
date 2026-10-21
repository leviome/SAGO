"""Native local file selection in a subprocess, without a GUI dependency in CUDA code."""
import os
from pathlib import Path
import shutil
import subprocess


def available():
    return bool((os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
                and (shutil.which('zenity') or shutil.which('kdialog')))


def pick_scene(initial=''):
    if not available():
        raise ValueError('当前环境无法打开系统文件选择框，请使用浏览器选择文件或拖入 PLY。')
    initial = Path(initial).expanduser() if initial else Path.home()
    directory = initial if initial.is_dir() else initial.parent
    if shutil.which('zenity'):
        command = ['zenity', '--file-selection', '--title=打开 SAGO 场景',
                   '--file-filter=3DGS 场景 | *.ply *.PLY', '--filename=' + str(directory) + '/']
    else:
        command = ['kdialog', '--getopenfilename', str(directory), '*.ply *.PLY|3DGS 场景', '--title', '打开 SAGO 场景']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 1:
        return {'cancelled': True}
    if result.returncode != 0:
        raise ValueError('无法打开系统文件选择框，请使用浏览器选择文件或拖入 PLY。')
    path = result.stdout.strip()
    if not path:
        return {'cancelled': True}
    if not Path(path).is_file() or Path(path).suffix.lower() != '.ply':
        raise ValueError('请选择存在的 3DGS PLY 文件')
    return {'path': str(Path(path).resolve())}
