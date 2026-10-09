#!/bin/bash
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_main_cache
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
export KA_MAIN_CAPTURE_OUT=$R/.cache/dsl_v2/w3_capture.jsonl PYTHONDONTWRITEBYTECODE=1
rm -f $KA_MAIN_CAPTURE_OUT
cd $R/.cache/tmp/w3run
timeout 5400 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p main_capture_plugin test_core.py \
  -k "test_bin_op or test_unary_op or test_reduce1d or test_reduce2d or test_scan2d or test_where or test_cast or test_dot or test_atomic_rmw or test_load_store_same_ptr or test_masked_load or test_math_op or test_abs or test_umulhi" \
  > $R/.cache/dsl_v2/w3_capture_pytest.log 2>&1
echo "EXIT $?" >> $R/.cache/dsl_v2/w3_capture_pytest.log
