#!/usr/bin/env bash
set -u

# This queue deliberately waits for the already-running 4096-step jobs.  It
# retries only runs whose first failure was a documented out-of-memory event;
# a second failure remains unresolved and is never relabeled as a negative.
ROOT=/data1/tzh/kernel-analyzer
LOG="$ROOT/results/property/declared_persistent_4096/expanded_controls/logs/retries_after_queue.log"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

WAIT_PIDS=(255786 256693 341778 347417 347920 351994 353519 354508 362222 363168)
while :; do
  alive=0
  for pid in "${WAIT_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then alive=1; fi
  done
  [ "$alive" -eq 0 ] && break
  sleep 60
done

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/data1/tzh/miniconda3/envs/pt_nightly/bin/python

run_case() {
  local arch="$1" model="$2" bank="$3" release="$4" plan="$5" case="$6" out="$7" ckpt="$8"
  echo "START $case $(date -Is)"
  "$PY" scripts/run_bound_endpoint_consequence_v21.py \
    --architecture "$arch" --model "$model" --input-bank "$bank" \
    --release-dir "$release" --case-plan "$plan" --case-id "$case" \
    --output "$out" --checkpoint "$ckpt" --steps 4096 \
    --state-role TRAJECTORY --device cuda:0 --allow-graph-breaks
  rc=$?
  echo "END $case rc=$rc $(date -Is)"
}

run_case deepseek /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  results/property/tcmp_allop_v1/input_banks/deepseek8b_seq256_trajectory4096.json \
  results/coverage/runtime_releases/deepseek8b_seq256_r2 \
  results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq256.json \
  multishape-backward-cell-0103 \
  results/property/declared_persistent_4096/expanded_controls/multishape-backward-cell-0103_4096.json \
  /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0103_retry.pt

run_case deepseek /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json \
  results/coverage/runtime_releases/deepseek8b_seq64_r1 \
  results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json \
  multishape-backward-cell-0153 \
  results/property/declared_persistent_4096/expanded_controls/multishape-backward-cell-0153_4096.json \
  /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0153_retry.pt

run_case deepseek /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json \
  results/coverage/runtime_releases/deepseek8b_seq64_r1 \
  results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json \
  multishape-backward-cell-0190 \
  results/property/declared_persistent_4096/expanded_controls/multishape-backward-cell-0190_4096.json \
  /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0190_retry2.pt

run_case phi /data1/tzh/models/microsoft/Phi-4-mini-instruct \
  results/property/declared_persistent_4096/expanded_controls/input_banks/phi4_seq128_cycled_4224.json \
  results/coverage/runtime_releases/phi4_seq128_r1 \
  results/property/joint_bias_formation_v1/negative_consequence_plans/phi4_seq128.json \
  multishape-backward-cell-0501 \
  results/property/declared_persistent_4096/expanded_controls/multishape-backward-cell-0501_4096.json \
  /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0501_retry.pt

echo "REBUILD_AUDIT $(date -Is)"
"$PY" scripts/build_all_bias_case_audit.py
