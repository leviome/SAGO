#!/usr/bin/env python3
"""Headless NeuS2 worker; run using the separately configured NeuS2 Python."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--scene')
    parser.add_argument('--network')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.root)
    for pattern in ('build*/**/pyngp*.so', 'build*/**/pyngp*.pyd'):
        sys.path.extend(str(p.parent) for p in root.glob(pattern))
    import pyngp as ngp
    if args.probe:
        assert hasattr(ngp.Testbed, 'compute_and_save_marching_cubes_mesh')
        print('NeuS2 ready', flush=True)
        return
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.nerf.sharpen = 0.0
    testbed.load_training_data(args.scene)
    testbed.reload_network_from_file(args.network)
    testbed.nerf.render_with_camera_distortion = True
    testbed.shall_train = True
    last = 0.0
    while testbed.training_step < args.steps:
        if not testbed.frame():
            raise RuntimeError('NeuS2 stopped before completing training')
        now = time.monotonic()
        if now-last >= 1:
            print('SAGO_MESH_PROGRESS '+json.dumps(dict(step=testbed.training_step, loss=testbed.loss)), flush=True)
            last = now
    testbed.shall_train = False
    print('SAGO_MESH_PROGRESS '+json.dumps(dict(step=args.steps, stage='extract')), flush=True)
    testbed.compute_and_save_marching_cubes_mesh(args.output, [args.resolution]*3)


if __name__ == '__main__':
    main()
