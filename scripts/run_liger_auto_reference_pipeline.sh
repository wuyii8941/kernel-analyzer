#!/usr/bin/env bash
# Capture (liger env, GPU) and analyze (ka_main, CPU) the 96 Liger units in batches.
# Analysis of batch k runs while batch k+1 is captured; capture packages are deleted after analysis.
set -euo pipefail
ROOT=/data1/tzh/kernel-analyzer
cd "$ROOT"
export HOME=$ROOT/.cache XDG_CACHE_HOME=$ROOT/.cache TRITON_CACHE_DIR=$ROOT/.cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$ROOT/.cache/inductor TMPDIR=$ROOT/.cache/tmp HF_HOME=$ROOT/.cache/hf
WORK=$ROOT/.cache/liger_auto
OUT=${OUT:-$ROOT/.cache/liger_auto/compact}
mkdir -p "$WORK" "$OUT"
BATCH=${BATCH:-8}
pids=()
for start in $(seq 0 $BATCH 95); do
  end=$((start + BATCH - 1))
  cap=$WORK/capture_$start
  if ls "$OUT"/unit$(printf %03d $end).json >/dev/null 2>&1; then continue; fi
  /data1/tzh/envs/liger/bin/python scripts/capture_liger_order_accumulation.py --units "$start-$end" --out "$cap" \
      > "$WORK/capture_$start.log" 2>&1
  /data1/tzh/envs/ka_main/bin/python scripts/analyze_liger_order_auto_reference.py --capture "$cap" --out "$OUT" \
      > "$WORK/analyze_$start.log" 2>&1 &
  pids+=($!)
  # keep at most three analyses in flight
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
done
wait
echo PIPELINE_DONE
