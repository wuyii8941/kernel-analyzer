#!/usr/bin/env bash
set -u

# Retry the frozen Gemma RMSNorm candidate with a small reachable carrier.
# The original projection carrier can require an additional multi-GB Adam
# buffer; this retry preserves the same endpoint and runtime release while
# avoiding that memory-only failure.  A failed retry remains unresolved.
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/envs/pt_nightly_transformers5/bin/python"
GPU="${1:-0}"
LOG="$ROOT/results/property/declared_persistent_4096/expanded_controls/logs/gemma4_norm_small_carrier_4096.log"
OUT="$ROOT/results/property/declared_persistent_4096/gemma4_norm_v3_small_carrier_4096"
mkdir -p "$OUT" "$(dirname "$LOG")"

while true; do
  free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F, -v g="$GPU" '$1+0==g {gsub(/ /,"",$2); print $2}')
  # The small-carrier retry needs several GB, but other jobs may safely
  # share a 49-GB card.  Gate on available memory rather than requiring the
  # whole card to be almost empty.
  if [[ -n "$free" && "$free" -ge 12000 ]]; then break; fi
  echo "[$(date -Is)] waiting gpu=$GPU free_memory=${free:-unknown}" >>"$LOG"
  sleep 120
done

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/archive/round1_code/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_WORKER_START=subprocess
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PY" "$ROOT/scripts/run_gemma4_norm_consequence.py" \
  --model /data1/tzh/models/google/gemma-4-E2B \
  --input-bank "$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json" \
  --trajectory-input-bank "$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory4096.json" \
  --runtime-release "$ROOT/results/property/declared_persistent_4096/gemma4_norm_v3_long/runtime_release" \
  --prediction "$ROOT/results/property/declared_persistent_4096/gemma4_norm_v3_long/prediction.json" \
  --carrier model.language_model.layers.0.input_layernorm.weight \
  --region-id forward:2 --endpoint in_out_ptr0 --case-id gemma4_norm_small_carrier \
  --output "$OUT/consequence.json" --steps 4096 --state-role TRAJECTORY \
  --learning-rate 1e-5 --device cuda:0 >>"$LOG" 2>&1
code=$?
echo "[$(date -Is)] exit=$code" >>"$LOG"
exit "$code"
