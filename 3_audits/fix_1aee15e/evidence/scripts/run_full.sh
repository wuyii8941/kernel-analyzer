#!/bin/bash
# full regression for the audit fix: trigger trace (W1 contracts) and log under .cache/audit_work/full
R=/data1/tzh/kernel-analyzer
cd $R
export HOME=$R/.cache XDG_CACHE_HOME=$R/.cache TRITON_CACHE_DIR=$R/.cache/triton TMPDIR=$R/.cache/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
export KA_TRIGGER_TRACE=$R/.cache/audit_work/full/trig
rm -f $R/.cache/audit_work/full/trig.*
timeout 7200 /data1/tzh/envs/ka_main/bin/python -m pytest -q -p no:cacheprovider -rfEs 2_tool/tests > $R/.cache/audit_work/full/full_tests.log 2>&1
echo "EXIT $?" >> $R/.cache/audit_work/full/full_tests.log
