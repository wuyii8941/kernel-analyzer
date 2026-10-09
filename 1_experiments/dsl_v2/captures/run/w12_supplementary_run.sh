#!/bin/bash
R=/data1/tzh/kernel-analyzer
KSEL="test_atomic_rmw or test_tensor_atomic_rmw or test_atomic_cas or test_scaled_dot or test_argmax" OUT=$R/.cache/dsl_v2/w12_supp_core.jsonl LOG=$R/.cache/dsl_v2/w12_supp_core.log $R/1_experiments/dsl_v2/captures/run/w12_cross_run.sh
KSEL="test_tma_gather_dot_pipeline" TFILE=test_tensor_descriptor.py OUT=$R/.cache/dsl_v2/w12_supp_td.jsonl LOG=$R/.cache/dsl_v2/w12_supp_td.log $R/1_experiments/dsl_v2/captures/run/w12_cross_run.sh
