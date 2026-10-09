#!/bin/bash
# Rerun of the test files the capture fixes of 2e30681 affect (DSL v2 increment 6 evaluation)
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_main_cache
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
export KA_MAIN_CAPTURE_OUT=$R/.cache/dsl_v2/w6c_rerun.jsonl PYTHONDONTWRITEBYTECODE=1
rm -f $KA_MAIN_CAPTURE_OUT $R/.cache/dsl_v2/w6c_rerun.log
cd $R/.cache/tmp/w4broad
for spec in "test_standard.py" "test_tensor_descriptor.py" "test_core.py -k test_cat_nd or test_reshape or test_const"; do
  set -- $spec
  f=$1; shift
  if [ $# -gt 0 ]; then shift; K="$*"; timeout 3600 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p main_capture_plugin $f -k "$K" > $R/.cache/dsl_v2/w6c_$f.log 2>&1
  else timeout 3600 /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p main_capture_plugin $f > $R/.cache/dsl_v2/w6c_$f.log 2>&1; fi
  echo "$f exit $? $(tail -1 $R/.cache/dsl_v2/w6c_$f.log)" >> $R/.cache/dsl_v2/w6c_rerun.log
done
echo RERUN_DONE >> $R/.cache/dsl_v2/w6c_rerun.log
