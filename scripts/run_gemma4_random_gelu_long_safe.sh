#!/usr/bin/env bash
set -u

# Long consequence replay for the previously measured Gemma GELU/loss
# feedback control.  It is intentionally queued after the carrier-scoped
# RMSNorm retry so it never competes for the same GPU.
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/envs/pt_nightly_transformers5/bin/python"
GPU="${1:-3}"
WAIT_PID="${2:-437966}"
OUT="$ROOT/results/property/declared_persistent_4096/gemma4_random_gelu_loss_backward_long"
LOG="$ROOT/results/property/declared_persistent_4096/expanded_controls/logs/gemma4_random_gelu_loss_backward_long.log"
MODEL="/data1/tzh/models/google/gemma-4-E2B"
INPUT="$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"
TRAJ="$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory4096.json"
TARGET="backward:1401"
CARRIER="model.language_model.per_layer_model_projection.weight"

mkdir -p "$(dirname "$LOG")" "$ROOT/results/property/declared_persistent_4096/unresolved"
echo "[$(date -Is)] queued Gemma GELU long replay; waiting for pid=$WAIT_PID" >>"$LOG"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done

if [[ -s "$OUT/consequence.json" ]]; then
  echo "[$(date -Is)] SKIP existing $OUT/consequence.json" >>"$LOG"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  archive="$ROOT/results/property/declared_persistent_4096/unresolved/gemma4_random_gelu_loss_backward_long_partial_$(date +%Y%m%dT%H%M%S)"
  mv "$OUT" "$archive"
  echo "[$(date -Is)] archived partial output to $archive" >>"$LOG"
fi

export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/archive/round1_code/src"
echo "[$(date -Is)] START Gemma GELU long gpu=$GPU" >>"$LOG"
if CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 \
   TORCHINDUCTOR_WORKER_START=subprocess OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   "$PY" "$ROOT/scripts/run_gemma4_v3_validation.py" \
     --model "$MODEL" --input-bank "$INPUT" --consequence-bank "$TRAJ" \
     --output-dir "$OUT" --steps 16 --consequence-steps 4096 \
     --runtime-seed 24000 --learning-rate 1e-5 --device cuda:0 \
     --carrier "$CARRIER" --target-region "$TARGET" \
     >>"$LOG" 2>&1; then
  echo "[$(date -Is)] COMPLETE Gemma GELU long" >>"$LOG"
else
  code=$?
  "$PY" - "$code" "$LOG" "$OUT" >"$ROOT/results/property/declared_persistent_4096/unresolved/gemma4_random_gelu_loss_backward_long_unresolved.json" <<'PY'
import json, sys
code, log, out = sys.argv[1:]
print(json.dumps({
    "schema": "kernel-analyzer-gemma4-gelu-long-replay-v1",
    "status": "UNRESOLVED_LONG_REPLAY_RESOURCE",
    "exit_code": int(code),
    "log": log,
    "partial_output": out,
    "claim_boundary": "The previously observed 32-step feedback consequence received a dedicated 4096-step retry; failure is unresolved, not negative.",
}, indent=2, sort_keys=True))
PY
  echo "[$(date -Is)] UNRESOLVED Gemma GELU long exit=$code" >>"$LOG"
fi
