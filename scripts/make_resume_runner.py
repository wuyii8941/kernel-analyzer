#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT = """#!/usr/bin/env bash
set -euo pipefail

cd /data1/tzh/forkcert
export HF_HOME=/data1/tzh/forkcert/cache/huggingface
export HF_HUB_CACHE=/data1/tzh/forkcert/cache/huggingface/hub
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=/data1/tzh/forkcert/cache/torchinductor
export TRITON_CACHE_DIR=/data1/tzh/forkcert/cache/triton
export XDG_CACHE_HOME=/data1/tzh/forkcert/cache/xdg
export PIP_CACHE_DIR=/data1/tzh/forkcert/cache/pip
export MPLCONFIGDIR=/data1/tzh/forkcert/cache/matplotlib
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" data results reports logs
LOG="logs/resume_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "ForkCert resume log: $LOG"

PY={python}
export PYTHONPATH=/data1/tzh/forkcert/src
if [[ -z "${{CUDA_VISIBLE_DEVICES:-}}" || "${{CUDA_VISIBLE_DEVICES}}" == *,* ]]; then
  SELECTED_GPU=$("$PY" scripts/select_gpu.py --require-bf16 --require-flash --fallback-any)
  if [[ -n "$SELECTED_GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
    echo "Selected one CUDA device: physical index $CUDA_VISIBLE_DEVICES"
  fi
fi
RUNTIME_DIR=results/runtime_configs
"$PY" scripts/resolve_runtime_configs.py \\
  --out-dir "$RUNTIME_DIR" \\
  --config configs/phase0_grpo.example.yaml \\
  --config configs/hf_pair.example.yaml \\
  --config configs/hf_debug_fp32_bf16.example.yaml \\
  --config configs/hf_sdpa_math_flash.example.yaml \\
  --config configs/hf_logsoftmax_upcast.example.yaml \\
  --config configs/hf_rmsnorm_reference.example.yaml \\
  --config configs/hf_materialization.example.yaml \\
  --config configs/hf_matmul_reduction.example.yaml
PHASE0_CONFIG="$RUNTIME_DIR/phase0_grpo.example.yaml"
PAIR_TEMPLATE="$RUNTIME_DIR/hf_pair.example.yaml"
DEBUG_TEMPLATE="$RUNTIME_DIR/hf_debug_fp32_bf16.example.yaml"
SDPA_TEMPLATE="$RUNTIME_DIR/hf_sdpa_math_flash.example.yaml"
LOGSOFTMAX_TEMPLATE="$RUNTIME_DIR/hf_logsoftmax_upcast.example.yaml"
RMSNORM_TEMPLATE="$RUNTIME_DIR/hf_rmsnorm_reference.example.yaml"
MATERIALIZATION_TEMPLATE="$RUNTIME_DIR/hf_materialization.example.yaml"
MATMUL_TEMPLATE="$RUNTIME_DIR/hf_matmul_reduction.example.yaml"
PAIR_CONFIG=results/configs/hf_pair.phase0_final.yaml
DEBUG_CONFIG=results/configs/hf_debug_fp32_bf16.phase0_final.yaml
SDPA_CONFIG=results/configs/hf_sdpa_math_flash.phase0_final.yaml
LOGSOFTMAX_CONFIG=results/configs/hf_logsoftmax_upcast.phase0_final.yaml
RMSNORM_CONFIG=results/configs/hf_rmsnorm_reference.phase0_final.yaml
MATERIALIZATION_CONFIG=results/configs/hf_materialization.phase0_final.yaml
MATMUL_CONFIG=results/configs/hf_matmul_reduction.phase0_final.yaml
SAMPLES=data/phase0_grpo_samples.jsonl

run_step() {{
  local name="$1"
  local output="$2"
  shift 2
  if [[ -s "$output" ]]; then
    echo "[skip] $name: $output exists"
    return 0
  fi
  echo "[run] $name"
  "$@"
  if [[ ! -s "$output" ]]; then
    echo "[fail] $name did not create non-empty output: $output" >&2
    return 1
  fi
  echo "[done] $name -> $output"
}}

run_jsonl_step() {{
  local name="$1"
  local output="$2"
  local min_lines="$3"
  shift 3
  if [[ -s "$output" ]]; then
    local lines
    lines=$(wc -l < "$output")
    if [[ "$lines" -ge "$min_lines" ]]; then
      echo "[skip] $name: $output exists with $lines lines"
      return 0
    fi
    echo "[rerun] $name: $output has $lines lines, expected at least $min_lines"
    rm -f "$output"
  fi
  echo "[run] $name"
  "$@"
  local lines
  lines=$(wc -l < "$output")
  if [[ "$lines" -lt "$min_lines" ]]; then
    echo "[fail] $name produced $lines lines, expected at least $min_lines: $output" >&2
    return 1
  fi
  echo "[done] $name -> $output ($lines lines)"
}}

echo "[run] audit_env (always refresh runtime fingerprint)"
"$PY" scripts/audit_env.py --out results/env_audit.json

run_step prompt_pairs data/prompt_pairs.jsonl \\
  "$PY" scripts/prepare_prompt_pairs.py --count {sample_count} --out data/prompt_pairs.jsonl

echo "[run] preflight (always verify imports and CUDA in this shell)"
"$PY" scripts/preflight.py \\
    --config "$PHASE0_CONFIG" \\
    --config "$PAIR_TEMPLATE" \\
    --config "$DEBUG_TEMPLATE" \\
    --config "$SDPA_TEMPLATE" \\
    --config "$LOGSOFTMAX_TEMPLATE" \\
    --config "$RMSNORM_TEMPLATE" \\
    --config "$MATERIALIZATION_TEMPLATE" \\
    --config "$MATMUL_TEMPLATE" \\
    --samples data/prompt_pairs.jsonl \\
    --require-ml \\
    --require-rl \\
    --require-cuda \\
    --out results/preflight.json

run_step phase0_grpo data/phase0_grpo_dump.metadata.json \\
  "$PY" scripts/phase0_grpo_train.py \\
    --config "$PHASE0_CONFIG" \\
    --out-jsonl data/phase0_grpo_dump.jsonl \\
    --samples-jsonl "$SAMPLES" \\
    --final-rollout-jsonl data/phase0_final_rollout.jsonl \\
    --output-dir data/phase0_policy_final
test -s data/phase0_grpo_dump.jsonl
test -s "$SAMPLES"
test -s data/phase0_final_rollout.jsonl
test -s data/phase0_policy_final/config.json

run_step checkpoint_configs results/configs/phase0_final_configs.json \\
  "$PY" scripts/write_checkpoint_configs.py \\
    --checkpoint data/phase0_policy_final \\
    --out-dir results/configs \\
    --config "$PAIR_TEMPLATE" \\
    --config "$DEBUG_TEMPLATE" \\
    --config "$SDPA_TEMPLATE" \\
    --config "$LOGSOFTMAX_TEMPLATE" \\
    --config "$RMSNORM_TEMPLATE" \\
    --config "$MATERIALIZATION_TEMPLATE" \\
    --config "$MATMUL_TEMPLATE"

echo "[run] phase0_final_preflight (always verify final checkpoint inputs)"
"$PY" scripts/preflight.py \\
    --config "$PAIR_CONFIG" \\
    --config "$DEBUG_CONFIG" \\
    --config "$SDPA_CONFIG" \\
    --config "$LOGSOFTMAX_CONFIG" \\
    --config "$RMSNORM_CONFIG" \\
    --config "$MATERIALIZATION_CONFIG" \\
    --config "$MATMUL_CONFIG" \\
    --samples "$SAMPLES" \\
    --require-ml \\
    --out results/preflight.phase0_final.json

run_step phase0_report reports/phase0.md \\
  "$PY" scripts/phase0_margin_hist.py \\
    --input data/phase0_grpo_dump.jsonl \\
    --out-json results/phase0_margin_summary.json \\
    --report reports/phase0.md \\
    --fail-on-downgrade \\
    --require-real-training
"$PY" scripts/check_gates.py --phase phase0 --input results/phase0_margin_summary.json

run_jsonl_step phase1_debug results/phase1_debug_fp32_bf16.jsonl 50000 \\
  "$PY" scripts/phase1_logprob_pipeline.py \\
    --config "$DEBUG_CONFIG" \\
    --samples "$SAMPLES" \\
    --out-jsonl results/phase1_debug_fp32_bf16.jsonl \\
    --report reports/phase1_debug.md \\
    --enforce-self-gate \\
    --enforce-scale-gate
"$PY" scripts/check_gates.py --phase phase1 --input results/phase1_debug_fp32_bf16.jsonl

run_jsonl_step phase1_logprobs results/phase1_logprobs.jsonl 50000 \\
  "$PY" scripts/phase1_logprob_pipeline.py \\
    --config "$PAIR_CONFIG" \\
    --samples "$SAMPLES" \\
    --out-jsonl results/phase1_logprobs.jsonl \\
    --report reports/phase1.md \\
    --enforce-self-gate \\
    --enforce-scale-gate
"$PY" scripts/check_gates.py --phase phase1 --input results/phase1_logprobs.jsonl

run_jsonl_step phase1_sdpa results/phase1_sdpa_logprobs.jsonl 50000 \\
  "$PY" scripts/phase1_logprob_pipeline.py \\
    --config "$SDPA_CONFIG" \\
    --samples "$SAMPLES" \\
    --out-jsonl results/phase1_sdpa_logprobs.jsonl \\
    --report reports/phase1_sdpa.md \\
    --enforce-self-gate \\
    --enforce-scale-gate
"$PY" scripts/check_gates.py --phase phase1 --input results/phase1_sdpa_logprobs.jsonl

run_step phase1_manifest results/phase1_pair_manifest.json \\
  "$PY" scripts/write_phase1_manifest.py \\
    --debug results/phase1_debug_fp32_bf16.jsonl \\
    --claim-compile results/phase1_logprobs.jsonl \\
    --claim-sdpa results/phase1_sdpa_logprobs.jsonl \\
    --env-audit results/env_audit.json \\
    --out results/phase1_pair_manifest.json \\
    --report reports/phase1_pairs.md \\
    --fail-on-incomplete

run_jsonl_step phase15_measurements results/phase15_measurements.jsonl 6 \\
  bash -lc 'rm -f results/phase15_measurements.jsonl && \\
    "$0" scripts/phase15_measure_hf.py --config "$1" --samples "$4" --out-jsonl results/phase15_measurements.jsonl --max-samples 4 --max-modules 64 && \\
    "$0" scripts/phase15_measure_hf.py --config "$2" --samples "$4" --out-jsonl results/phase15_measurements.jsonl --max-samples 4 --max-modules 64 && \\
    "$0" scripts/phase15_measure_hf.py --config "$3" --samples "$4" --out-jsonl results/phase15_measurements.jsonl --max-samples 4 --max-modules 64 && \\
    "$0" scripts/phase15_measure_hf.py --config "$5" --samples "$4" --out-jsonl results/phase15_measurements.jsonl --max-samples 4 --max-modules 64 && \\
    "$0" scripts/phase15_measure_hf.py --config "$6" --samples "$4" --out-jsonl results/phase15_measurements.jsonl --max-samples 4 --max-modules 64 && \\
    "$0" scripts/phase15_measure_hf.py --config "$7" --samples "$4" --out-jsonl results/phase15_measurements.jsonl --max-samples 4 --max-modules 64' "$PY" "$PAIR_CONFIG" "$SDPA_CONFIG" "$LOGSOFTMAX_CONFIG" "$SAMPLES" "$RMSNORM_CONFIG" "$MATERIALIZATION_CONFIG" "$MATMUL_CONFIG"
"$PY" scripts/check_gates.py --phase phase15 --input results/phase15_measurements.jsonl

run_step phase15_report reports/phase15.md \\
  "$PY" scripts/phase15_attribution_ladder.py \\
    --measurements results/phase15_measurements.jsonl \\
    --out-json results/phase15_attribution.json \\
    --report reports/phase15.md

run_step phase2_analytic_draft results/phase2_sources.analytic_draft.json \\
  "$PY" scripts/make_analytic_source_template.py \\
    --measurements results/phase15_measurements.jsonl \\
    --out results/phase2_sources.analytic_draft.json

run_step phase2_sources results/phase2_sources.initial.json \\
  "$PY" scripts/make_bounds_sources.py \\
    --measurements results/phase15_measurements.jsonl \\
    --out results/phase2_sources.initial.json

run_step phase2_bounds results/phase2_bounds.json \\
  "$PY" scripts/phase2_bounds.py \\
    --sources results/phase2_sources.initial.json \\
    --measurements results/phase15_measurements.jsonl \\
    --logprob-jsonl results/phase1_logprobs.jsonl \\
    --out-json results/phase2_bounds.json \\
    --report reports/phase2.md \\
    --fail-on-unusable
"$PY" scripts/check_gates.py --phase phase2 --input results/phase2_bounds.json

run_step phase3_controlled results/phase3_controlled_certificates.jsonl \\
  "$PY" scripts/phase3_calibration.py \\
    --logprob-jsonl results/phase1_logprobs.jsonl \\
    --margin-jsonl data/phase0_grpo_dump.jsonl \\
    --out-model-json results/phase3_calibration.json \\
    --out-jsonl results/phase3_controlled_certificates.jsonl \\
    --report reports/phase3.md
"$PY" scripts/check_gates.py --phase phase3 --input results/phase3_controlled_certificates.jsonl --calibration-json results/phase3_calibration.json

run_step phase4_certificates results/phase4_certificates.jsonl \\
  "$PY" scripts/phase4_natural_scan.py \\
    --logprob-jsonl results/phase1_logprobs.jsonl \\
    --rollout-jsonl data/phase0_final_rollout.jsonl \\
    --bounds-json results/phase2_bounds.json \\
    --calibration-json results/phase3_calibration.json \\
    --attribution-json results/phase15_attribution.json \\
    --out-jsonl results/phase4_certificates.jsonl \\
    --report reports/phase4.md \\
    --fail-on-missing-rollout \\
    --require-rollout-state final \\
    --require-rollout-token-id
"$PY" scripts/check_gates.py --phase phase4 --input results/phase4_certificates.jsonl --logprob-jsonl results/phase1_logprobs.jsonl --require-token-match

run_step fork_cases reports/fork_cases.md \\
  "$PY" scripts/report_fork_cases.py \\
    --certificates results/phase4_certificates.jsonl \\
    --out reports/fork_cases.md \\
    --max-cases 20

run_step phase5_bug results/phase5_bug_certificates.jsonl \\
  "$PY" scripts/phase5_hf_bug_injection.py \\
    --config "$PAIR_CONFIG" \\
    --samples "$SAMPLES" \\
    --logprob-jsonl results/phase1_logprobs.jsonl \\
    --rollout-jsonl data/phase0_final_rollout.jsonl \\
    --valid-certificates results/phase4_certificates.jsonl \\
    --bounds-json results/phase2_bounds.json \\
    --out-jsonl results/phase5_bug_certificates.jsonl \\
    --report reports/phase5.md
"$PY" scripts/check_gates.py --phase phase5 --input results/phase5_bug_certificates.jsonl --require-token-match

run_step phase6_grad results/phase6_grad_certificates.jsonl \\
  "$PY" scripts/phase6_grad_contrib.py \\
    --certificates results/phase4_certificates.jsonl \\
    --samples "$SAMPLES" \\
    --config "$PAIR_CONFIG" \\
    --out-jsonl results/phase6_grad_certificates.jsonl \\
    --report reports/phase6.md
"$PY" scripts/check_gates.py --phase phase6 --input results/phase6_grad_certificates.jsonl --phase4-certificates results/phase4_certificates.jsonl --require-autograd

run_step phase6_twin results/phase6_twin_summary.json \\
  "$PY" scripts/phase6_twin_training.py \\
    --config "$PAIR_CONFIG" \\
    --samples "$SAMPLES" \\
    --rollout-jsonl data/phase0_final_rollout.jsonl \\
    --phase4-certificates results/phase4_certificates.jsonl \\
    --steps 200 \\
    --measure-every 5 \\
    --out-jsonl results/phase6_twin_trajectory.jsonl \\
    --out-summary results/phase6_twin_summary.json \\
    --report reports/phase6_twin.md
"$PY" scripts/check_gates.py --phase phase6_twin --input results/phase6_twin_summary.json --phase4-certificates results/phase4_certificates.jsonl

run_step final_audit reports/audit.md \\
  "$PY" scripts/audit_results.py --report reports/audit.md
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a resumable GPU-visible ForkCert runner.")
    parser.add_argument("--python", default="/data1/tzh/conda-envs/forkcert/bin/python")
    parser.add_argument("--config", default="configs/hf_pair.example.yaml")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--out", default="run_phase1_gpu_resume.sh")
    args = parser.parse_args()

    out = Path(args.out)
    out.write_text(
        SCRIPT.format(
            python=args.python,
            config=args.config,
            sample_count=args.sample_count,
        ),
        encoding="utf-8",
    )
    out.chmod(0o755)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
