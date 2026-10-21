#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
runtime_python=python3
if [[ ${1:-} == --python ]]; then runtime_python=$2; shift 2; fi
if [[ $# -ne 0 ]]; then echo 'Usage: setup-runtime.sh [--python /path/to/cuda/python]' >&2; exit 2; fi
runtime_dir=${XDG_DATA_HOME:-$HOME/.local/share}/sago-desktop/venv
"$runtime_python" -m venv --system-site-packages "$runtime_dir"
"$runtime_dir/bin/python" -m pip install -r "$root/requirements.txt"
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT
for module in simple-knn rasterizer-depth rasterizer-mask rasterizer-topk; do
    cp -a "$root/vendor/$module" "$build_dir/$module"
done
"$runtime_dir/bin/python" -m pip install --no-build-isolation "$build_dir/simple-knn" "$build_dir/rasterizer-depth" "$build_dir/rasterizer-mask" "$build_dir/rasterizer-topk"
"$runtime_dir/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA driver/runtime unavailable"'
printf 'Runtime ready: %s\n' "$runtime_dir/bin/python"
