#!/usr/bin/env bash
# Rerun Liger step 5 through the unified entry: capture (liger env) -> reference stage (ka_main) -> statistics.
set -euo pipefail
ROOT=/data1/tzh/kernel-analyzer
cd "$ROOT"
export HOME=$ROOT/.cache XDG_CACHE_HOME=$ROOT/.cache TRITON_CACHE_DIR=$ROOT/.cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$ROOT/.cache/inductor TMPDIR=$ROOT/.cache/tmp HF_HOME=$ROOT/.cache/hf
WORK=$ROOT/.cache/liger_unified
DECL=$ROOT/results/reference_eval/declarations/liger_fp32_order.json
REF=$WORK/reference
mkdir -p "$WORK" "$REF"
for start in $(seq 0 8 95); do
  end=$((start + 7))
  cap=$WORK/capture_$start
  if [ -f "$REF/unit$(printf %03d $end).json" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=0 /data1/tzh/envs/liger/bin/python scripts/capture_liger_order_accumulation.py \
      --units "$start-$end" --out "$cap" > "$WORK/capture_$start.log" 2>&1
  units=$(seq -f "unit%03g" $start $end | paste -sd,)
  /data1/tzh/envs/ka_main/bin/python scripts/run_reference_analysis.py --declaration "$DECL" --stage reference \
      --capture-root "$cap" --out "$REF" --units "$units" --delete-captures > "$WORK/reference_$start.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
done
wait
CUDA_VISIBLE_DEVICES=0 /data1/tzh/envs/ka_main/bin/python scripts/run_reference_analysis.py --declaration "$DECL" \
    --stage statistics --out "$REF" --report "$ROOT/results/reference_eval/liger_fp32_order_unified_entry.json" \
    > "$WORK/statistics.log" 2>&1
echo UNIFIED_DONE
