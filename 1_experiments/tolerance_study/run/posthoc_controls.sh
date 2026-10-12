#!/bin/bash
# Post-hoc controls (added after seeing the paired results; labelled as post-hoc in the records): same-configuration
# reruns that measure the run-to-run divergence of GPU training itself (non-deterministic kernels), against which the
# trajectory distances of B-2 and A-2 are read.
R=/data1/tzh/kernel-analyzer
W=$R/1_experiments/tolerance_study/workload/finetune.py
PY=/data1/tzh/envs/latest_train/bin/python
export HOME=$R/.cache HF_HOME=$R/.cache/hf XDG_CACHE_HOME=$R/.cache TRITON_CACHE_DIR=$R/.cache/triton TMPDIR=$R/.cache/tmp \
       PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
LOG=$R/.cache/round2/posthoc_controls.log
mkdir -p $R/.cache/round2/posthoc/eager_rerun $R/.cache/round2/posthoc/bf16_rerun
job() { local g=$1 out=$2; shift 2; [ -f "$out/log.json" ] && return
  CUDA_VISIBLE_DEVICES=$g timeout 14400 $PY $W --out "$out" "$@" > "$out.stdout" 2>&1; echo "$(date +%H:%M:%S) $out exit $?" >> $LOG; }
queue() { local g=$1
  for s in $(seq $((5000 + g)) 4 5007); do
    job $g $R/.cache/round2/posthoc/eager_rerun/seed$s --seed $s --steps 200 --param-dtype fp32 --autocast bf16 \
        --optimizer radam_eager --capture-steps 1,10,50,100,200 --eval-steps 200; done
  s=$((3000 + g))
  job $g $R/.cache/round2/posthoc/bf16_rerun/seed$s --seed $s --steps 600 --capture-steps 1,10,50,100,300,600 --eval-steps 100,300,600
}
for g in 0 1 2 3; do queue $g & done; wait; echo POSTHOC_DONE >> $LOG
