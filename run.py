#!/usr/bin/env python3
"""Run from any directory: python /path/to/SAGO_web/run.py."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='SAGO Python backend + browser UI')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--checkpoint', default=os.environ.get('SAGO_CHECKPOINT', str(ROOT/'models/sam2_hiera_large.pt')))
    parser.add_argument('--sam3-checkpoint', help='SAM3 本地权重')
    parser.add_argument('--sam31-checkpoint', help='SAM3.1 multiplex 本地权重')
    parser.add_argument('--sam3-python', help='独立 SAM3 推理环境的 Python 路径')
    args = parser.parse_args()
    for value, key in ((args.sam3_checkpoint, 'SAGO_SAM3_CHECKPOINT'),
                       (args.sam31_checkpoint, 'SAGO_SAM31_CHECKPOINT'),
                       (args.sam3_python, 'SAGO_SAM3_PYTHON')):
        if value:
            os.environ[key] = str(Path(value).expanduser().resolve())
    # Legacy SAGO debug paths are relative: keep every generated file in this project.
    os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    from backend.server import create_app
    app = create_app(args.checkpoint)
    print(f'SAGO Web: http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
