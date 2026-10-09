#!/bin/bash
# Broad capture after increment 14 (status run, not a registered evaluation): the same official unit-test files as
# inc6_broad, each captured launch evaluated by the current evaluator (main_capture_plugin.py)
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_main_cache
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/src:$R/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
OUT=$R/.cache/dsl_v2/w14_broad
mkdir -p $OUT; rm -f $OUT/*.jsonl $OUT/run.log
cd $R/.cache/tmp/w4broad
run() {  # gpu file timeout
  CUDA_VISIBLE_DEVICES=$1 KA_MAIN_CAPTURE_OUT=$OUT/$2.jsonl timeout $3 /data1/tzh/envs/triton_main/bin/python -m pytest -q \
    -p no:cacheprovider -p main_capture_plugin $2 > $OUT/$2.log 2>&1
  echo "$2 exit $? $(tail -1 $OUT/$2.log)" >> $OUT/run.log
}
run 0 test_core.py 14400 &
( run 1 test_standard.py 3600; run 1 test_tensor_descriptor.py 3600 ) &
( run 2 test_random.py 3600; run 2 test_conversions.py 3600 ) &
run 3 test_libdevice.py 3600 &
wait
cat $OUT/test_core.py.jsonl $OUT/test_random.py.jsonl $OUT/test_standard.py.jsonl $OUT/test_libdevice.py.jsonl \
  $OUT/test_conversions.py.jsonl $OUT/test_tensor_descriptor.py.jsonl > $OUT/combined.jsonl
echo BROAD_DONE >> $OUT/run.log
