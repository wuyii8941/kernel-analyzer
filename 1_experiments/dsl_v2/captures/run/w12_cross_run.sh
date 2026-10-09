#!/bin/bash
# DSL v2 increment 12: TTIR vs AMD (gfx942 / gfx950, official stages to TTGIR) or NVIDIA TTGIR references on the same captured launches
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_xlevel_run
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/2_tool/src:$R/2_tool/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
KSEL=${KSEL:-test_dot or test_reduce1d or test_reduce2d or test_scan2d or test_where or test_cast}
export KA_CROSS_TARGETS=${TGTS:-gfx942,gfx950}
export KA_CROSS_LEVEL_OUT=${OUT:-$R/.cache/dsl_v2/w12_cross.jsonl}
rm -f $KA_CROSS_LEVEL_OUT
LOG=${LOG:-$R/.cache/dsl_v2/w12_cross.log}
cd $R/.cache/tmp/w4broad
timeout 7200 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p cross_level_plugin ${TFILE:-test_core.py} \
  -k "$KSEL" \
  > $LOG 2>&1
echo "EXIT $?" >> $LOG
