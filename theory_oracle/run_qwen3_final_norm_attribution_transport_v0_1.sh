#!/usr/bin/env bash
set -euo pipefail

repo=/data1/tzh/forkcert
python=/data1/tzh/pt_nightly_env/bin/python
root="$repo/results/operator_oracle/qwen3_final_norm_attribution_transport_v0_1"

run_arm() {
  local snapshot=$1 anchor=$2 state=$3 repeat=$4 vectors=$5
  "$python" -m theory_oracle.qwen3_backward_singleton_repair_v0_4 \
    --treatment final_norm_backward \
    --snapshot-dir "$snapshot" \
    --anchor-states "$anchor" \
    --arm compiled \
    --repeat "$repeat" \
    --out-dir "$root/$state/final_norm_$repeat" \
    $vectors
}

run_arm \
  "$repo/data/qwen3_grpo_transport_a_replay_step29_transition_v0_1" \
  "$repo/results/training_step_oracle/qwen3_grpo_cross_state_capture_v0_1/a_grad_states.jsonl" \
  a_replay 1 --save-vectors
run_arm \
  "$repo/data/qwen3_grpo_transport_a_replay_step29_transition_v0_1" \
  "$repo/results/training_step_oracle/qwen3_grpo_cross_state_capture_v0_1/a_grad_states.jsonl" \
  a_replay 2 ""
run_arm \
  "$repo/data/qwen3_grpo_heldout_transport_v01_b_step29_transition" \
  "$repo/results/training_step_oracle/qwen3_grpo_heldout_transport_v0_1/b_grad_states.jsonl" \
  b_original 2 ""
run_arm \
  "$repo/data/qwen3_grpo_transport_c_replay_step29_transition_v0_1" \
  "$repo/results/training_step_oracle/qwen3_grpo_cross_state_capture_v0_1/c_grad_states.jsonl" \
  c_replay 1 --save-vectors
run_arm \
  "$repo/data/qwen3_grpo_transport_c_replay_step29_transition_v0_1" \
  "$repo/results/training_step_oracle/qwen3_grpo_cross_state_capture_v0_1/c_grad_states.jsonl" \
  c_replay 2 ""
