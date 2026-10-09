#!/bin/bash
# Follow-up batch (commit c3dd596): rerun every capture whose statuses the certificates and the evaluator changes can
# move.  Same official test files / selections / plugins as the recorded runs; outputs under .cache/audit_work/rerun2.
R=/data1/tzh/kernel-analyzer
O=$R/.cache/audit_work/rerun2
mkdir -p $O; rm -f $O/*.jsonl $O/*.log $O/run.log
BASE="HOME=$R/.cache TMPDIR=$R/.cache/tmp LD_LIBRARY_PATH=/data1/tzh/envs/tilelang_gcc11/lib PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$R/.cache/mpl PYTHONPATH=/data1/tzh/envs/triton_main/lib/python3.11/site-packages:$R/2_tool/src:$R/2_tool/scripts/dsl_v2:/data1/tzh/envs/ka_main/lib/python3.11/site-packages"
PY=/data1/tzh/envs/triton_main/bin/python
cap() {  # gpu dir outname timeout args...
  local g=$1 d=$2 n=$3 t=$4; shift 4
  ( cd $d && env $BASE TRITON_CACHE_DIR=$R/.cache/triton_main_cache CUDA_VISIBLE_DEVICES=$g KA_MAIN_CAPTURE_OUT=$O/$n.jsonl \
      timeout $t $PY -m pytest -q -p no:cacheprovider -p main_capture_plugin "$@" > $O/$n.log 2>&1 )
  echo "$n exit $? $(tail -1 $O/$n.log)" >> $O/run.log
}
xl() {  # gpu targets outname file kselect
  ( cd $R/.cache/tmp/w4broad && env $BASE TRITON_CACHE_DIR=$R/.cache/triton_xlevel_run CUDA_VISIBLE_DEVICES=$1 KA_CROSS_TARGETS=$2 \
      KA_CROSS_LEVEL_OUT=$O/$3.jsonl timeout 7200 $PY -m pytest -q -p no:cacheprovider -p cross_level_plugin $4 -k "$5" > $O/$3.log 2>&1 )
  echo "$3 exit $? $(tail -1 $O/$3.log)" >> $O/run.log
}
tut() {  # gpu outname tutorials...
  local g=$1 n=$2; shift 2
  for t in "$@"; do
    ( cd $R/.cache/tmp/w4tut && env $BASE TRITON_CACHE_DIR=$R/.cache/triton_main_cache CUDA_VISIBLE_DEVICES=$g \
        timeout 7200 $PY $R/2_tool/scripts/dsl_v2/tutorial_capture.py --out $O/$n.jsonl $R/.cache/upstream/triton/python/tutorials/$t.py >> $O/$n.log 2>&1 )
    echo "$n/$t exit $?" >> $O/run.log
  done
}
B=$R/.cache/tmp/w4broad
DEF="test_dot or test_reduce1d or test_reduce2d or test_scan2d or test_where or test_cast"
( cap 0 $B broad_test_core 14400 test_core.py; xl 0 90,100 cross_nvidia test_core.py "$DEF" ) &
( cap 1 $B broad_test_standard 3600 test_standard.py; cap 1 $B broad_test_tensor_descriptor 3600 test_tensor_descriptor.py
  xl 1 gfx942,gfx950 cross_amd test_core.py "$DEF" ) &
( cap 2 $B broad_test_random 3600 test_random.py; cap 2 $B broad_test_conversions 3600 test_conversions.py
  xl 2 gfx942,gfx950 cross_amd_supp_core test_core.py "test_atomic_rmw or test_tensor_atomic_rmw or test_atomic_cas or test_scaled_dot or test_argmax"
  xl 2 gfx942,gfx950 cross_amd_supp_td test_tensor_descriptor.py "test_tma_gather_dot_pipeline"
  tut 2 tutorials 01-vector-add 02-fused-softmax 03-matrix-multiplication 05-layer-norm 06-fused-attention 08-grouped-gemm 09-persistent-matmul ) &
( cap 3 $B broad_test_libdevice 3600 test_libdevice.py; cap 3 $R/.cache/tmp/gluontests gluon_test_core 3600 test_core.py ) &
wait
echo RERUN_DONE >> $O/run.log
