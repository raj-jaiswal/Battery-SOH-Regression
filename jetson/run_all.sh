#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
models=(gpr_rbf gpr_matern gpr_ard sparse_gpr deep_kernel_gpr svr random_forest xgboost wide_nn deep_nn wide_deep_nn bilstm symbolic_regression)
cycles="${CYCLES_PER_CELL:-8}"
repeats="${REPEATS:-30}"
warmup="${WARMUP:-3}"
provider="${PROVIDER:-auto}"
failed=0
for model in "${models[@]}"; do
  script="$(find "$ROOT/jetson" -maxdepth 1 -name "*_${model}.py" -print -quit)"
  if [[ -z "$script" ]]; then echo "Missing runner for $model"; failed=$((failed+1)); continue; fi
  echo; echo "========== $model =========="
  python3 "$script" --cycles-per-cell "$cycles" --repeats "$repeats" --warmup "$warmup" --provider "$provider"
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "FAILED: $model (exit $rc)"; failed=$((failed+1)); fi
done
echo; echo "Jetson benchmark run complete; failed models: $failed"
[[ $failed -eq 0 ]]
