#!/usr/bin/env bash
set -euo pipefail

cd /data1/tzh/forkcert
PY=/data1/tzh/conda-envs/forkcert/bin/python
export PYTHONPATH=/data1/tzh/forkcert/src:/data1/tzh/forkcert
export HF_HOME=/data1/tzh/forkcert/cache/huggingface
export HF_HUB_CACHE=/data1/tzh/forkcert/cache/huggingface/hub
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/data1/tzh/forkcert/cache/torchinductor_bf16_external
export TRITON_CACHE_DIR=/data1/tzh/forkcert/cache/triton_bf16_external
export XDG_CACHE_HOME=/data1/tzh/forkcert/cache/xdg
export MPLCONFIGDIR=/data1/tzh/forkcert/cache/matplotlib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

ROOT=data/bf16_external
RESULTS=results/bf16_external
REPORTS=reports/bf16_external
LOGS=logs/bf16_external
mkdir -p "$ROOT" "$RESULTS" "$REPORTS" "$LOGS" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
LOG="$LOGS/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Native BF16 replay log: $LOG"

# Select one native-BF16 GPU when the caller has not already isolated one.
# There is deliberately no FP16 fallback.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  SELECTED_GPU=$("$PY" scripts/select_gpu.py --require-bf16)
  export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
fi
echo "Selected physical GPU: $CUDA_VISIBLE_DEVICES"

"$PY" scripts/phase15_bf16_preflight.py --out "$RESULTS/preflight.json"

"$PY" scripts/phase0_grpo_train.py \
  --config configs/phase15_bf16_grpo.yaml \
  --out-jsonl "$ROOT/grpo_dump.jsonl" \
  --samples-jsonl "$ROOT/samples.jsonl" \
  --final-rollout-jsonl "$ROOT/final_rollout.jsonl" \
  --output-dir "$ROOT/policy_final" \
  --online-compile-scan-jsonl "$RESULTS/online_compile.jsonl"

"$PY" scripts/phase0_margin_hist.py \
  --input "$ROOT/grpo_dump.jsonl" \
  --metadata "$ROOT/grpo_dump.metadata.json" \
  --out-json "$RESULTS/margin_summary.json" \
  --report "$REPORTS/margin.md" \
  --histogram-svg "$REPORTS/margin_hist.svg" \
  --fail-on-downgrade \
  --require-real-training

"$PY" scripts/phase4_online_analysis.py \
  --input "$RESULTS/online_compile.jsonl" \
  --expected-tokens 51200 \
  --out-json "$RESULTS/online_analysis.json" \
  --report "$REPORTS/online_analysis.md"

"$PY" scripts/enrich_online_scan.py \
  --input "$RESULTS/online_compile.jsonl" \
  --samples "$ROOT/samples.jsonl" \
  --training-metadata "$ROOT/grpo_dump.metadata.json" \
  --out "$RESULTS/online_enriched.jsonl"

"$PY" scripts/phase4_natural_scan.py \
  --logprob-jsonl "$RESULTS/online_enriched.jsonl" \
  --rollout-jsonl "$RESULTS/online_enriched.jsonl" \
  --eps 0.2 \
  --out-jsonl "$RESULTS/certificates.jsonl" \
  --report "$REPORTS/natural_scan.md" \
  --fail-on-missing-rollout \
  --require-rollout-state pre_minibatch \
  --require-rollout-token-id

"$PY" scripts/phase15_bf16_external_audit.py \
  --preflight "$RESULTS/preflight.json" \
  --training-metadata "$ROOT/grpo_dump.metadata.json" \
  --online-jsonl "$RESULTS/online_compile.jsonl" \
  --certificates "$RESULTS/certificates.jsonl" \
  --expected-rows 51200 \
  --out-json "$RESULTS/audit.json" \
  --report "$REPORTS/audit.md"

echo "Native BF16 external replay completed and audited: $RESULTS/audit.json"
