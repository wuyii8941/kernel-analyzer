#!/bin/bash
# DSL v2 increment 11 evaluation: TTIR vs sm_90 / sm_100 TTGIR references on the same captured launches
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_xlevel_run
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
KSEL=${KSEL:-test_dot or test_reduce1d or test_reduce2d or test_scan2d or test_where or test_cast}
export KA_CROSS_LEVEL_OUT=${OUT:-$R/.cache/dsl_v2/w11_cross.jsonl}
rm -f $KA_CROSS_LEVEL_OUT
cd $R/.cache/tmp/w4broad
timeout 7200 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p cross_level_plugin test_core.py \
  -k "$KSEL" \
  > $R/.cache/dsl_v2/w11_cross.log 2>&1
echo "EXIT $?" >> $R/.cache/dsl_v2/w11_cross.log
