#!/usr/bin/env bash
set -u
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/envs/pt_nightly_transformers5/bin/python"
cd "$ROOT"
LOG="$ROOT/results/property/declared_persistent_4096/expanded_controls/logs/gemma4_norm_safe.log"
OUT="$ROOT/results/property/declared_persistent_4096/gemma4_norm_4096_rebound.json"
mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] START Gemma4 RMSNorm safe replay" >>"$LOG"
# The rebound capture was not byte-reproducible under the current compiler.
# Use the older byte-frozen release as the exact runtime and feed it the
# independent long trajectory bank. This keeps the F+B contrast exact while
# separating compile-time engineering inputs from the long-run states.
if CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT:$ROOT/src:$ROOT/archive/round1_code/src" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
    TORCHINDUCTOR_CACHE_DIR=/data1/tzh/cache/torchinductor/gemma4_e2b_frozen \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/run_gemma4_norm_consequence.py \
      --model /data1/tzh/models/google/gemma-4-E2B \
      --input-bank "$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json" \
      --trajectory-input-bank "$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory4096.json" \
      --runtime-release "$ROOT/results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/runtime_release" \
      --prediction "$ROOT/results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/norm_prediction_freeze.json" \
      --case-id gemma4_e2b_ple_rmsnorm --steps 4096 --state-role TRAJECTORY --device cuda:0 \
      --output "$OUT" \
      --optimizer-ablation-output "$ROOT/results/property/declared_persistent_4096/gemma4_norm_4096_rebound_optimizer.json" \
      >>"$LOG" 2>&1; then
  echo "[$(date -Is)] COMPLETE Gemma4 RMSNorm safe replay" >>"$LOG"
else
  code=$?
  "$PY" - "$code" "$LOG" "$OUT" >"$ROOT/results/property/declared_persistent_4096/unresolved/gemma4_norm_4096_rebound_unresolved.json" <<'PY'
import json, sys
code, log, output = sys.argv[1:]
print(json.dumps({
  "schema": "kernel-analyzer-gemma4-long-replay-retry-failure-v1",
  "status": "UNRESOLVED_LONG_REPLAY_RESOURCE",
  "exit_code": int(code), "log": log, "partial_output": output,
  "claim_boundary": "A single-worker dedicated-GPU retry was attempted; failure is unresolved, not a negative result."
}, indent=2, sort_keys=True))
PY
  echo "[$(date -Is)] FAILED Gemma4 RMSNorm safe replay exit=$code" >>"$LOG"
fi
