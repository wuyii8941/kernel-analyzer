#!/bin/bash
# Run tool_spec_check cases in parallel, one process per case, round-robin over GPUs.
#   scripts/run_tool_cases_parallel.sh GROUP OUTDIR NPROC SEEDS CASE...
set -u
GROUP=$1; OUT=$2; NPROC=$3; SEEDS=$4; shift 4
ROOT=/data1/tzh/kernel-analyzer
export CASE_TIMEOUT=${CASE_TIMEOUT:-1200}
# shared machine: no Inductor compile worker pool, few BLAS / OpenMP threads per process
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export HOME=$ROOT/.cache XDG_CACHE_HOME=$ROOT/.cache TRITON_CACHE_DIR=$ROOT/.cache/triton TMPDIR=$ROOT/.cache/tmp
mkdir -p "$OUT"
i=0
for c in "$@"; do echo "$i $c"; i=$((i+1)); done | xargs -P "$NPROC" -n 2 sh -c '
  gpu=$(( $0 % 4 ))
  OPINFO_CASES="$1" CUDA_VISIBLE_DEVICES=$gpu timeout '"${CASE_TIMEOUT:-1200}"' /data1/tzh/envs/ka_main/bin/python '"$ROOT"'/scripts/tool_spec_check.py --group '"$GROUP"' --case "$1" --out '"$OUT"' --seeds '"$SEEDS"' > '"$OUT"'/log_"$1".txt 2>&1
  echo "done $1"'
