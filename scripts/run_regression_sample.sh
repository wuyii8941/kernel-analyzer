#!/bin/bash
# Regression sample (scripts/data/regression_sample.txt: GROUP CASE...) with 6 seeds, from a code checkout CODE_ROOT
# (e.g. a git worktree of the commit to compare against), one process per group, 3 groups at a time.
#   scripts/run_regression_sample.sh CODE_ROOT OUTDIR [SAMPLE_FILE]
# Per-group paths: vLLM shim, flash_attn / mamba_ssm shims, vendored FLA source (see docs/environments.md).
CODE=$1; OUT=$2; SAMPLE=${3:-scripts/data/regression_sample.txt}
cd /data1/tzh/kernel-analyzer
C=/data1/tzh/kernel-analyzer/.cache
export PYTHONPATH=$C/pylibs/vllm_shim:$C/pylibs/shim:$C/src/flash-linear-attention-main:$CODE/src OPINFO_SELECT=onesample \
       HOME=$C XDG_CACHE_HOME=$C TRITON_CACHE_DIR=$C/triton TMPDIR=$C/tmp TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=3
mkdir -p "$OUT"
i=0
while read g cases; do
  gpu=$(( i % 4 )); i=$((i + 1))
  echo "$gpu $g $(echo $cases | tr ' ' ',')"
done < "$SAMPLE" | xargs -P 3 -L 1 sh -c '
  OPINFO_CASES="$2" CUDA_VISIBLE_DEVICES=$0 timeout 7200 /data1/tzh/envs/ka_main/bin/python '"$CODE"'/scripts/tool_spec_check.py \
    --group $1 --case "$2" --out '"$OUT"' --seeds 6 > '"$OUT"'/log_$1.txt 2>&1; echo done $1'
