#!/usr/bin/env bash
set -euo pipefail

cd /data1/tzh/forkcert
export HF_HOME=/data1/tzh/forkcert/cache/huggingface
export HF_HUB_CACHE=/data1/tzh/forkcert/cache/huggingface/hub
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export XDG_CACHE_HOME=/data1/tzh/forkcert/cache/xdg
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" results logs

PY=/data1/tzh/conda-envs/forkcert/bin/python
PYTHONPATH=/data1/tzh/forkcert/src "$PY" scripts/prefetch_model.py \
  --model Qwen/Qwen3-0.6B \
  --cache-dir "$HF_HOME" \
  --out results/model_prefetch.qwen3_0_6b.json
