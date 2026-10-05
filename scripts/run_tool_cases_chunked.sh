#!/bin/bash
# Run tool_spec_check cases in parallel, CHUNK cases per process (amortizes torch import / case building / warm-up),
# round-robin over the 4 GPUs.  A chunk that dies leaves its remaining cases without JSON; rerun those separately.
#   scripts/run_tool_cases_chunked.sh GROUP OUTDIR NPROC SEEDS CHUNK CASE...
set -u
GROUP=$1; OUT=$2; NPROC=$3; SEEDS=$4; CHUNK=$5; shift 5
ROOT=/data1/tzh/kernel-analyzer
export CASE_TIMEOUT=${CASE_TIMEOUT:-1200}
# shared machine: no Inductor compile worker pool, few BLAS / OpenMP threads per process
export TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export HOME=$ROOT/.cache XDG_CACHE_HOME=$ROOT/.cache TRITON_CACHE_DIR=$ROOT/.cache/triton TMPDIR=$ROOT/.cache/tmp
mkdir -p "$OUT"
printf '%s\n' "$@" | paste -d, $(printf -- '- %.0s' $(seq "$CHUNK")) | awk '{print NR-1, $0}' | sed 's/,*$//' |
  xargs -P "$NPROC" -n 2 sh -c '
  gpu=$(( $0 % 4 ))
  n=$(echo "$1" | tr "," "\n" | wc -l)
  first=$(echo "$1" | cut -d, -f1)
  OPINFO_CASES="$1" CUDA_VISIBLE_DEVICES=$gpu timeout $(( '"$CASE_TIMEOUT"' * n )) /data1/tzh/envs/ka_main/bin/python '"$ROOT"'/scripts/tool_spec_check.py --group '"$GROUP"' --case "$1" --out '"$OUT"' --seeds '"$SEEDS"' > '"$OUT"'/log_chunk_"$first".txt 2>&1
  echo "done chunk $first ($n cases)"'
