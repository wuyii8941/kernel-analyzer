#!/bin/bash
# Group B runs (protocol appendix B).  B-1: seeds 4000-4031, fp32 parameters + bf16 autocast, eager RAdam, 100 steps,
# captures before steps 6/7/10/20/100.  B-2: seeds 5000-5007 x {compiled RAdam, eager RAdam}, 200 steps, eval at 200.
R=/data1/tzh/kernel-analyzer
W=$R/1_experiments/tolerance_study/workload/finetune.py
PY=/data1/tzh/envs/latest_train/bin/python
export HOME=$R/.cache HF_HOME=$R/.cache/hf XDG_CACHE_HOME=$R/.cache TRITON_CACHE_DIR=$R/.cache/triton TMPDIR=$R/.cache/tmp \
       PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
LOG=$R/.cache/round2/B_runs.log
job() {
  local g=$1 out=$2; shift 2
  [ -f "$out/log.json" ] && { echo "skip $out" >> $LOG; return; }
  CUDA_VISIBLE_DEVICES=$g timeout 14400 $PY $W --out "$out" "$@" > "$out.stdout" 2>&1
  echo "$(date +%H:%M:%S) $out exit $?" >> $LOG
}
mkdir -p $R/.cache/round2/captures_b $R/.cache/round2/paired_b/{compiled,eager}
queue() {
  local g=$1
  for s in $(seq $((4000 + g)) 4 4031); do
    job $g $R/.cache/round2/captures_b/seed$(printf %04d $s) --seed $s --steps 100 --param-dtype fp32 --autocast bf16 \
        --optimizer radam_eager --capture-steps 6,7,10,20,100
  done
  for s in $(seq $((5000 + g)) 4 5007); do
    job $g $R/.cache/round2/paired_b/compiled/seed$s --seed $s --steps 200 --param-dtype fp32 --autocast bf16 \
        --optimizer radam_compiled --capture-steps 1,10,50,100,200 --eval-steps 200
    job $g $R/.cache/round2/paired_b/eager/seed$s --seed $s --steps 200 --param-dtype fp32 --autocast bf16 \
        --optimizer radam_eager --capture-steps 1,10,50,100,200 --eval-steps 200
  done
}
for g in 0 1 2 3; do queue $g & done
wait
echo "B_RUNS_DONE" >> $LOG
