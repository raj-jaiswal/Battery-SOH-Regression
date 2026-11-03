#!/usr/bin/env bash
set -euo pipefail

# Run this script ON THE JETSON ORIN NANO, after copying the project and running
# models/translate.py on the laptop. TensorRT engines are device/JetPack specific.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPT_ROOT="$ROOT/models/optimized"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "ERROR: trtexec not found. Install/use the TensorRT tools from NVIDIA JetPack."
  exit 1
fi

count=0
while IFS= read -r -d '' metadata; do
  model_dir="$(dirname "$metadata")"
  onnx_file=""
  read -r onnx_file min_shape opt_shape max_shape < <(python3 - "$metadata" <<'PY'
import json, sys
m=json.load(open(sys.argv[1]))
name=m.get('onnx_file')
shape=m.get('onnx_input_shape')
if not name or not shape:
    print('', '', '', '')
    raise SystemExit
rank = len(shape)
dims = [int(x) for x in shape[1:]]
if not dims or any(x <= 0 for x in dims):
    print('', '', '', '')
    raise SystemExit

def fmt(batch): return 'x'.join(map(str, [batch] + dims))
print(name, 'input:'+fmt(1), 'input:'+fmt(32), 'input:'+fmt(256))
PY
  )
  if [[ -z "${onnx_file:-}" ]]; then
    continue
  fi
  onnx_path="$model_dir/$onnx_file"
  engine_path="$model_dir/${onnx_file%.onnx}.engine"
  if [[ ! -f "$onnx_path" ]]; then
    echo "Skipping missing ONNX file: $onnx_path"
    continue
  fi
  echo "Building TensorRT engine: $onnx_path"
  trtexec \
    --onnx="$onnx_path" \
    --saveEngine="$engine_path" \
    --fp16 \
    --minShapes="$min_shape" \
    --optShapes="$opt_shape" \
    --maxShapes="$max_shape"
  count=$((count + 1))
done < <(find "$OPT_ROOT" -name metadata.json -print0)

echo "Built $count TensorRT engine(s). They will be preferred automatically by jetson_runtime.py when TensorRT and PyCUDA are importable."
