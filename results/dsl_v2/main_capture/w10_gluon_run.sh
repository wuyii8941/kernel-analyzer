#!/bin/bash
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_main_cache
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
export KA_MAIN_CAPTURE_OUT=$R/.cache/dsl_v2/w10_gluon.jsonl
rm -f $KA_MAIN_CAPTURE_OUT
cd $R/.cache/tmp/gluontests
timeout 3600 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p main_capture_plugin test_core.py > $R/.cache/dsl_v2/w10_gluon.log 2>&1
echo "EXIT $?" >> $R/.cache/dsl_v2/w10_gluon.log
