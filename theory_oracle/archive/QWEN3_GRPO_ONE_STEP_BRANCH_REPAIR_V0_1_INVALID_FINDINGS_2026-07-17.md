# Qwen3 GRPO One-Step Branch-Repair v0.1 Invalid Findings — 2026-07-17

## Verdict

`INVALID` due to compiled endpoint anchor-parity failure.

The pre-minibatch state reconstruction itself was exact. In the A/B/C executor:

- A selected log-probability exactly matched the confirmed eager value;
- B/C matched each other but differed from the confirmed compiled value;
- the changed B value placed B on the same clipping branch as A;
- C therefore made no intervention and `B-C=0`.

The apparent residual ratio of `1` is not a causal result. The independent scorer
omitted the Trainer's Qwen `logits_to_keep` realization, changing compiled fusion and
the compiled numerical endpoint. v0.2 restores the exact Trainer scoring path and
adds mandatory A/B/C anchor equality.

Evidence: `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/branch_repair.json`.

