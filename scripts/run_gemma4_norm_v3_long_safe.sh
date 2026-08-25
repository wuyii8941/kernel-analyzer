#!/usr/bin/env bash
set -u

# Same-process Gemma long replay.  Unlike the old frozen-release wrapper this
# compiles and consumes one release in one Python process, so wrapper bytes
# cannot silently drift between formation and consequence.
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/envs/pt_nightly_transformers5/bin/python"
GPU="${1:-0}"
OUT="$ROOT/results/property/declared_persistent_4096/gemma4_norm_v3_long_projection"
LOG="$ROOT/results/property/declared_persistent_4096/expanded_controls/logs/gemma4_norm_v3_long.log"
MODEL="/data1/tzh/models/google/gemma-4-E2B"
INPUT="$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"
TRAJ="$ROOT/results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory4096.json"

if [[ -s "$OUT/consequence.json" ]]; then
  echo "[$(date -Is)] SKIP existing $OUT/consequence.json" >>"$LOG"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  echo "[$(date -Is)] UNRESOLVED output directory already exists: $OUT" >>"$LOG"
  exit 2
fi
mkdir -p "$(dirname "$LOG")"
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/archive/round1_code/src"
echo "[$(date -Is)] START Gemma v3 long gpu=$GPU" >>"$LOG"
if CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 \
   TORCHINDUCTOR_WORKER_START=subprocess OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   "$PY" "$ROOT/scripts/run_gemma4_v3_validation.py" \
     --model "$MODEL" --input-bank "$INPUT" --consequence-bank "$TRAJ" \
     --output-dir "$OUT" --steps 16 --consequence-steps 4096 \
     --runtime-seed 24000 --learning-rate 1e-5 --device cuda:0 \
     --carrier model.language_model.per_layer_model_projection.weight \
     >>"$LOG" 2>&1; then
  echo "[$(date -Is)] COMPLETE Gemma v3 long" >>"$LOG"
else
  code=$?
  "$PY" - "$code" "$LOG" "$OUT" >"$ROOT/results/property/declared_persistent_4096/unresolved/gemma4_norm_v3_long_unresolved.json" <<'PY'
import json, sys
code, log, out = sys.argv[1:]
print(json.dumps({
    "schema": "kernel-analyzer-gemma4-long-replay-v1",
    "status": "UNRESOLVED_LONG_REPLAY_RESOURCE",
    "exit_code": int(code),
    "log": log,
    "partial_output": out,
    "claim_boundary": "A same-process retry was attempted; this is unresolved and is not a negative result.",
}, indent=2, sort_keys=True))
PY
  echo "[$(date -Is)] UNRESOLVED Gemma v3 long exit=$code" >>"$LOG"
fi
