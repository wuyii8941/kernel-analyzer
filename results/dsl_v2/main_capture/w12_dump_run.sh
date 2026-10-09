#!/bin/bash
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_w12_obs
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2
cd $R
/data1/tzh/envs/triton_main/bin/python scripts/dsl_v2/amd_ttgir_dump.py --dump $R/.cache/dsl_v2/w0_dump \
  --archs gfx942 gfx950 --out $R/.cache/dsl_v2/w12_amd_ttgir > $R/.cache/dsl_v2/w12_dump.log 2>&1
echo "EXIT $?" >> $R/.cache/dsl_v2/w12_dump.log
