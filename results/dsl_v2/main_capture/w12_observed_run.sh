#!/bin/bash
# DSL v2 increment 12: W0 observed inventory, the same 5952 official TTIR recompiled for gfx942 / gfx950 (stages to TTGIR)
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_w12_obs
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2
cd $R
/data1/tzh/envs/triton_main/bin/python scripts/dsl_v2/w0_observed.py --dump $R/.cache/dsl_v2/w0_dump \
  --targets hip:gfx942 hip:gfx950 --sources results/dsl_v2/w0/source_ops_e50b186e8bd2.json \
  results/dsl_v2/w0/source_upstream_ops_e50b186e8bd2.json --out $R/.cache/dsl_v2/w12_observed_amd.json --workers 16 \
  > $R/.cache/dsl_v2/w12_observed.log 2>&1
echo "EXIT $?" >> $R/.cache/dsl_v2/w12_observed.log
