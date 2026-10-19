#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run in the target Python environment with a matching CUDA toolkit installed.
python -m pip install --no-build-isolation ./vendor/simple-knn ./vendor/rasterizer-depth ./vendor/rasterizer-mask ./vendor/rasterizer-topk
