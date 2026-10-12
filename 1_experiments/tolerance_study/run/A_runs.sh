#!/bin/bash
# Group A runs (protocol 5A).  A-1: confirmation seeds 1000-1031, torchtune settings, 300 steps, captures at 10/100/300.
# A-2/A-3: seeds 3000-3011 x {pure bf16 fused + three EMAs, fp32 master, torchao stochastic rounding}, 600 steps.
# Outputs under .cache/round2 (raw captures stay local; summaries go to results/).  Four GPUs, one queue each.
R=/data1/tzh/kernel-analyzer
W=$R/1_experiments/tolerance_study/workload/finetune.py
PY=/data1/tzh/envs/latest_train/bin/python
export HOME=$R/.cache HF_HOME=$R/.cache/hf XDG_CACHE_HOME=$R/.cache TRITON_CACHE_DIR=$R/.cache/triton TMPDIR=$R/.cache/tmp \
       PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
LOG=$R/.cache/round2/A_runs.log
job() {  # gpu out args...
  local g=$1 out=$2; shift 2
  [ -f "$out/log.json" ] && { echo "skip $out" >> $LOG; return; }
  CUDA_VISIBLE_DEVICES=$g timeout 14400 $PY $W --out "$out" "$@" > "$out.stdout" 2>&1
  echo "$(date +%H:%M:%S) $out exit $?" >> $LOG
}
mkdir -p $R/.cache/round2/captures $R/.cache/round2/paired/{bf16,master,sr}
queue() {  # gpu index: A-1 seeds and paired seeds assigned round-robin
  local g=$1
  for s in $(seq $((1000 + g)) 4 1031); do
    job $g $R/.cache/round2/captures/seed$(printf %04d $s) --seed $s --steps 300 --capture-steps 10,100,300 --eval-steps 300
  done
  for s in $(seq $((3000 + g)) 4 3011); do
    job $g $R/.cache/round2/paired/bf16/seed$s --seed $s --steps 600 --ema all --capture-steps 1,10,50,100,300,600 --eval-steps 100,300,600
    job $g $R/.cache/round2/paired/master/seed$s --seed $s --steps 600 --master fp32 --capture-steps 1,10,50,100,300,600 --eval-steps 100,300,600
    job $g $R/.cache/round2/paired/sr/seed$s --seed $s --steps 600 --optimizer adamw_torchao_sr --capture-steps 1,10,50,100,300,600 --eval-steps 100,300,600
  done
}
for g in 0 1 2 3; do queue $g & done
wait
echo "A_RUNS_DONE" >> $LOG
