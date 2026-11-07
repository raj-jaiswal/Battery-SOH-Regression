#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v trtexec >/dev/null 2>&1 || { echo 'trtexec not found; use NVIDIA JetPack TensorRT tools.'; exit 1; }
for name in wide_branch deep_feature_extractor; do
  trtexec --onnx="$HERE/${name}.onnx" --saveEngine="$HERE/${name}.engine" --fp16     --minShapes=input:1x24 --optShapes=input:16x24 --maxShapes=input:128x24
done
