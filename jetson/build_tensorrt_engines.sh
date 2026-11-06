#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPT_ROOT="$ROOT/models/optimized"
if ! command -v trtexec >/dev/null 2>&1; then
  echo "ERROR: trtexec not found. Use NVIDIA JetPack TensorRT tools on the Orin Nano."
  exit 1
fi
if [[ ! -d "$OPT_ROOT" ]]; then
  echo "ERROR: $OPT_ROOT does not exist. Run the laptop translator and copy models/optimized/ first."
  exit 1
fi
built=0
skipped=0
while IFS= read -r -d '' metadata; do
  model_dir="$(dirname "$metadata")"
  read -r onnx_file input_name min_shape opt_shape max_shape < <(
    python3 - "$metadata" <<'PY'
import json, sys
m=json.load(open(sys.argv[1], encoding="utf-8"))
name=m.get("onnx_file")
shape=m.get("onnx_input_shape")
input_name=m.get("onnx_input_name", "input")
if not name or not shape or len(shape) < 2:
    print("", "", "", "", "")
    raise SystemExit(0)
dims=[int(x) for x in shape[1:]]
if any(x <= 0 for x in dims):
    print("", "", "", "", "")
    raise SystemExit(0)
def fmt(batch): return "x".join(map(str,[batch]+dims))
print(name, input_name, "input:"+fmt(1), "input:"+fmt(32), "input:"+fmt(128))
PY
  )
  if [[ -z "${onnx_file:-}" ]]; then skipped=$((skipped+1)); continue; fi
  onnx_path="$model_dir/$onnx_file"
  engine_path="$model_dir/${onnx_file%.onnx}.engine"
  if [[ ! -f "$onnx_path" ]]; then
    echo "Skipping missing ONNX: $onnx_path"; skipped=$((skipped+1)); continue
  fi
  echo "Building TensorRT engine: $onnx_path"
  trtexec --onnx="$onnx_path" --saveEngine="$engine_path" --fp16 \
    --minShapes="$min_shape" --optShapes="$opt_shape" --maxShapes="$max_shape"
  built=$((built+1))
done < <(find "$OPT_ROOT" -name metadata.json -print0)
echo "TensorRT engine build complete. Built=$built, skipped=$skipped"
