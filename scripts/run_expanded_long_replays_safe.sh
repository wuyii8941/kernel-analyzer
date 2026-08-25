#!/usr/bin/env bash
set -u

# Resource-safe continuation queue for the result-blind long-replay candidates.
# Each queue is assigned one physical GPU by the caller.  The Python runner
# resumes an existing checkpoint when one exists; a failed run is recorded only
# after this retry actually exits non-zero.

ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
LOG_ROOT="$ROOT/results/property/declared_persistent_4096/expanded_controls/logs"
FAIL_ROOT="$ROOT/results/property/declared_persistent_4096/expanded_controls/retry_failures"
mkdir -p "$LOG_ROOT" "$FAIL_ROOT"

run_case() {
  local gpu="$1" arch="$2" model="$3" input="$4" release="$5" plan="$6" id="$7" checkpoint="$8" extra="${9:-}"
  local output="$ROOT/results/property/declared_persistent_4096/expanded_controls/${id}_4096.json"
  local log="$LOG_ROOT/${id}_safe.log"
  local -a cmd=("$PY" "$ROOT/scripts/run_bound_endpoint_consequence_v21.py"
    --architecture "$arch" --model "$model" --input-bank "$input"
    --release-dir "$release" --case-plan "$plan" --case-id "$id"
    --output "$output" --checkpoint "$checkpoint" --steps 4096
    --state-role TRAJECTORY --device cuda:0)
  if [[ -s "$checkpoint" ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "$extra" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $extra )
    cmd+=("${extra_args[@]}")
  fi
  echo "[$(date -Is)] START $id gpu=$gpu resume=$([[ -s "$checkpoint" ]] && echo yes || echo no)" >>"$log"
  if CUDA_VISIBLE_DEVICES="$gpu" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${cmd[@]}" >>"$log" 2>&1; then
    echo "[$(date -Is)] COMPLETE $id" >>"$log"
  else
    local code=$?
    "$PY" - "$id" "$code" "$log" "$output" >"$FAIL_ROOT/${id}.json" <<'PY'
import json, sys
case_id, code, log, output = sys.argv[1:]
print(json.dumps({
    "schema": "kernel-analyzer-long-replay-retry-failure-v1",
    "case_id": case_id,
    "status": "UNRESOLVED_LONG_REPLAY_RESOURCE",
    "exit_code": int(code),
    "log": log,
    "partial_output": output,
    "claim_boundary": "This row was retried with a single compiler worker and a dedicated GPU; it is unresolved only because the retry itself failed. It is not a negative result.",
}, indent=2, sort_keys=True))
PY
    echo "[$(date -Is)] FAILED $id exit=$code" >>"$log"
  fi
}

# GPU assignment is passed as the first argument.  One process per GPU is
# intentional: the previous 32-worker-per-process fan-out starved all kernels.
GPU="${1:?usage: $0 GPU QUEUE_NAME}"
QUEUE="${2:?usage: $0 GPU QUEUE_NAME}"
case "$QUEUE" in
  deepseek)
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq128_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq128_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq128.json" \
      multishape-backward-cell-0057 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0057.pt
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq256_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq256_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq256.json" \
      multishape-backward-cell-0103 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0103.pt
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq64_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json" \
      multishape-backward-cell-0153 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0153.pt
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq64_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json" \
      multishape-backward-cell-0190 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0190.pt
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq64_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json" \
      multishape-backward-cell-0191 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0191.pt
    ;;
  deepseek_tail)
    # The tail is intentionally separate so a second dedicated 49-GiB GPU can
    # process the remaining DeepSeek rows in parallel with the first row.
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq256_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq256_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq256.json" \
      multishape-backward-cell-0103 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0103.pt
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq64_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json" \
      multishape-backward-cell-0153 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0153.pt
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq64_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json" \
      multishape-backward-cell-0190 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0190.pt
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq64_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json" \
      multishape-backward-cell-0191 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0191.pt
    ;;
  deepseek_last)
    run_case "$GPU" deepseek8b /data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/deepseek8b_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/deepseek8b_seq64_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/deepseek8b_seq64.json" \
      multishape-backward-cell-0191 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0191.pt
    ;;
  phi)
    run_case "$GPU" phi /data1/tzh/models/microsoft/Phi-4-mini-instruct \
      "$ROOT/results/property/declared_persistent_4096/expanded_controls/input_banks/phi4_seq128_cycled_4224.json" \
      "$ROOT/results/coverage/runtime_releases/phi4_seq128_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/phi4_seq128.json" \
      multishape-backward-cell-0508 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0508_retry.pt "--allow-graph-breaks"
    run_case "$GPU" phi /data1/tzh/models/microsoft/Phi-4-mini-instruct \
      "$ROOT/results/property/declared_persistent_4096/expanded_controls/input_banks/phi4_seq256_cycled_4224.json" \
      "$ROOT/results/coverage/runtime_releases/phi4_seq256_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/phi4_seq256.json" \
      multishape-backward-cell-0543 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0543.pt "--allow-graph-breaks"
    run_case "$GPU" phi /data1/tzh/models/microsoft/Phi-4-mini-instruct \
      "$ROOT/results/property/declared_persistent_4096/expanded_controls/input_banks/phi4_seq128_cycled_4224.json" \
      "$ROOT/results/coverage/runtime_releases/phi4_seq128_r1" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/phi4_seq128.json" \
      multishape-backward-cell-0501 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0501.pt "--allow-graph-breaks"
    ;;
  qwen)
    run_case "$GPU" qwen /data1/tzh/models/Qwen/Qwen3-1.7B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/qwen_seq256_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/qwen_seq256_r2" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/qwen_seq256.json" \
      multishape-backward-cell-0654 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0654.pt
    run_case "$GPU" qwen /data1/tzh/models/Qwen/Qwen3-1.7B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/qwen_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/qwen_seq64_r2" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/qwen_seq64.json" \
      multishape-backward-cell-0745 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0745.pt
    run_case "$GPU" qwen /data1/tzh/models/Qwen/Qwen3-1.7B \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/qwen_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/qwen_seq64_r2" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/qwen_seq64.json" \
      multishape-backward-cell-0747 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0747.pt
    run_case "$GPU" mamba /data1/tzh/models/state-spaces/mamba-130m-hf \
      "$ROOT/results/property/tcmp_allop_v1/input_banks/mamba_seq64_trajectory4096.json" \
      "$ROOT/results/coverage/runtime_releases/mamba_seq64_r2" \
      "$ROOT/results/property/joint_bias_formation_v1/negative_consequence_plans/mamba_seq64.json" \
      multishape-backward-cell-0450 /data1/tzh/cache/bias_long_expanded/multishape-backward-cell-0450.pt
    ;;
  *) echo "unknown queue: $QUEUE" >&2; exit 2;;
esac
