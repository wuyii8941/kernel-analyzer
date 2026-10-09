#!/bin/bash
# DSL v2 increment 14 evaluation: official test_core atomic tests (classic) and official Gluon test_core local-atomic
# tests under the official main build, every captured launch evaluated (main_capture_plugin)
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_main_cache
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/2_tool/src:$R/2_tool/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
( export CUDA_VISIBLE_DEVICES=0 KA_MAIN_CAPTURE_OUT=$R/.cache/dsl_v2/w14_core.jsonl
  rm -f $KA_MAIN_CAPTURE_OUT; cd $R/.cache/tmp/w3run
  timeout 7200 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p main_capture_plugin test_core.py \
    -k "atomic" > $R/.cache/dsl_v2/w14_core.log 2>&1; echo "EXIT $?" >> $R/.cache/dsl_v2/w14_core.log ) &
( export CUDA_VISIBLE_DEVICES=1 KA_MAIN_CAPTURE_OUT=$R/.cache/dsl_v2/w14_gluon.jsonl
  rm -f $KA_MAIN_CAPTURE_OUT; cd $R/.cache/tmp/gluontests
  timeout 3600 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p main_capture_plugin test_core.py \
    -k "atomic" > $R/.cache/dsl_v2/w14_gluon.log 2>&1; echo "EXIT $?" >> $R/.cache/dsl_v2/w14_gluon.log ) &
wait
