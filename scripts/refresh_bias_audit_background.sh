#!/usr/bin/env bash
set -u
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
LOG="$ROOT/results/property/declared_persistent_4096/audit_refresh.log"
while pgrep -f 'run_bound_endpoint_consequence_v21.py' >/dev/null 2>&1 || \
      pgrep -f 'run_gemma4_v3_validation.py' >/dev/null 2>&1 || \
      pgrep -f 'run_gemma_llama_target_queue.sh' >/dev/null 2>&1; do
  "$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$LOG" 2>&1 || true
  sleep 300
done
"$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$LOG" 2>&1 || true
