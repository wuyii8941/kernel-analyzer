#!/bin/bash
R=/data1/tzh/kernel-analyzer
export HOME=$R/.cache TMPDIR=$R/.cache/tmp TRITON_CACHE_DIR=$R/.cache/triton_main_cache MPLCONFIGDIR=$R/.cache/mpl
export LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/2_tool/src:$R/2_tool/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages
S=$R/.cache/upstream/triton/python/tutorials
rm -f $R/.cache/dsl_v2/w7_tutorials.jsonl
cd $R/.cache/tmp/w4tut
for t in 05-layer-norm; do
  timeout 7200 /data1/tzh/envs/triton_main/bin/python $R/2_tool/scripts/dsl_v2/tutorial_capture.py --out $R/.cache/dsl_v2/w7_tutorials.jsonl $S/$t.py \
    >> $R/.cache/dsl_v2/w7_tutorials.log 2>&1
  echo "$t exit $?" >> $R/.cache/dsl_v2/w7_tutorials.log
done
echo TUT_DONE >> $R/.cache/dsl_v2/w7_tutorials.log
