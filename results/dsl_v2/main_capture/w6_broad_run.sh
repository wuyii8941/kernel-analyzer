#!/bin/bash
# Broad capture (DSL v2 increment 4 evaluation): every test of the listed official unit-test files under the official
# main build, each captured launch evaluated by the tool-4.0 reference evaluator (scripts/dsl_v2/main_capture_plugin.py).
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_main_cache
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
export KA_MAIN_CAPTURE_OUT=$R/.cache/dsl_v2/w6_broad.jsonl PYTHONDONTWRITEBYTECODE=1
rm -f $KA_MAIN_CAPTURE_OUT
cd $R/.cache/tmp/w4broad
for f in test_core.py test_random.py test_standard.py test_libdevice.py test_conversions.py test_tensor_descriptor.py; do
  t=3600; [ $f = test_core.py ] && t=14400
  timeout $t /data1/tzh/envs/triton_main/bin/python -m pytest -q -p no:cacheprovider -p main_capture_plugin $f \
    > $R/.cache/dsl_v2/w6_broad_$f.log 2>&1
  echo "$f exit $? $(tail -1 $R/.cache/dsl_v2/w6_broad_$f.log)" >> $R/.cache/dsl_v2/w6_broad_run.log
done
echo BROAD_DONE >> $R/.cache/dsl_v2/w6_broad_run.log
