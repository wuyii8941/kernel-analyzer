#!/usr/bin/env bash
set -euo pipefail

repo=/data1/tzh/forkcert
python=/data1/tzh/pt_nightly_env/bin/python
root="$repo/results/operator_oracle/qwen3_operator_attribution_transport_v0_1"

run_baseline() {
  local snapshot=$1 anchor=$2 state=$3 arm=$4 repeat=$5 vectors=$6
  local out="$root/$state/${arm}_$repeat"
  "$python" -m theory_oracle.qwen3_grpo_natural_transition_v0_2 \
    --snapshot-dir "$snapshot" \
    --anchor-states "$anchor" \
    --arm "$arm" \
    --repeat "$repeat" \
    --out-dir "$out" \
    $vectors
}

run_silu() {
  local snapshot=$1 anchor=$2 state=$3 repeat=$4 vectors=$5
  local out="$root/$state/silu_middle_$repeat"
  "$python" -m theory_oracle.qwen3_backward_repeated_family_repair_v0_1 \
    --selected-position middle \
    --snapshot-dir "$snapshot" \
    --anchor-states "$anchor" \
    --arm compiled \
    --repeat "$repeat" \
    --out-dir "$out" \
    $vectors
}

run_cast() {
  local snapshot=$1 anchor=$2 state=$3 repeat=$4 vectors=$5
  local out="$root/$state/cast_up_control_$repeat"
  "$python" -m theory_oracle.qwen3_backward_multirole_cast_repair_v0_1 \
    --selected-role up_projection_weight_gradient_cast \
    --snapshot-dir "$snapshot" \
    --anchor-states "$anchor" \
    --arm compiled \
    --repeat "$repeat" \
    --out-dir "$out" \
    $vectors
}

a_snapshot="$repo/data/qwen3_grpo_transport_a_replay_step29_transition_v0_1"
a_anchor="$repo/results/training_step_oracle/qwen3_grpo_cross_state_capture_v0_1/a_grad_states.jsonl"
b_snapshot="$repo/data/qwen3_grpo_heldout_transport_v01_b_step29_transition"
b_anchor="$repo/results/training_step_oracle/qwen3_grpo_heldout_transport_v0_1/b_grad_states.jsonl"
c_snapshot="$repo/data/qwen3_grpo_transport_c_replay_step29_transition_v0_1"
c_anchor="$repo/results/training_step_oracle/qwen3_grpo_cross_state_capture_v0_1/c_grad_states.jsonl"

run_baseline "$a_snapshot" "$a_anchor" a_replay compiled 1 --save-vectors
run_baseline "$a_snapshot" "$a_anchor" a_replay compiled 2 ""
run_silu "$a_snapshot" "$a_anchor" a_replay 1 --save-vectors
run_silu "$a_snapshot" "$a_anchor" a_replay 2 ""
run_cast "$a_snapshot" "$a_anchor" a_replay 1 --save-vectors
run_cast "$a_snapshot" "$a_anchor" a_replay 2 ""

run_silu "$b_snapshot" "$b_anchor" b_original 2 ""
run_cast "$b_snapshot" "$b_anchor" b_original 2 ""

run_baseline "$c_snapshot" "$c_anchor" c_replay eager 1 --save-vectors
run_baseline "$c_snapshot" "$c_anchor" c_replay eager 2 ""
run_baseline "$c_snapshot" "$c_anchor" c_replay compiled 1 --save-vectors
run_baseline "$c_snapshot" "$c_anchor" c_replay compiled 2 ""
run_silu "$c_snapshot" "$c_anchor" c_replay 1 --save-vectors
run_silu "$c_snapshot" "$c_anchor" c_replay 2 ""
run_cast "$c_snapshot" "$c_anchor" c_replay 1 --save-vectors
run_cast "$c_snapshot" "$c_anchor" c_replay 2 ""
