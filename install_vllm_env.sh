#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-/data1/tzh/conda/bin/conda}"
ENV_DIR="${FORKCERT_VLLM_ENV:-/data1/tzh/conda-envs/forkcert-vllm}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA_BIN}" create -y --solver=classic -p "${ENV_DIR}" python=3.11 pip
fi

# vLLM 0.9.2 retains the V0 engine needed by Turing/T4. Pin Transformers to
# the contemporary Qwen3-capable 4.x release rather than accepting a future
# major version through vLLM's open-ended lower bound.
"${ENV_DIR}/bin/python" -m pip install \
  "vllm==0.9.2" \
  "transformers==4.53.2"

"${ENV_DIR}/bin/python" -m pip check
"${ENV_DIR}/bin/python" -m pip freeze > "${ROOT}/results/vllm_environment.freeze.txt"
