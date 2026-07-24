# Qwen3 GRPO Training-Control Confirmation v0.1 Invalid Findings — 2026-07-17

## Verdict

```text
complete confirmation: INVALID
descriptive valid-state disagreements: 5
compiler correctness: NO CLAIM
```

Both trajectories completed 10 rollout states and 5,120 token rows. Exact eager and
compiled self repeats were stable, values were finite, and five clipping
disagreements were observed in the first three states across the two trajectories.

The candidate identity gate nevertheless failed. The default Dynamo
`recompile_limit=8` produced tracked compiled invocations for states 1–8, then the
last two shape specializations in each trajectory ran without invoking the tracking
backend. The compile audits recorded eight graphs and only 24 invocations rather
than the required 30 warmup/first/second calls.

All five descriptive events happened in states whose three compiled calls were
tracked. They remain useful discovery evidence, but the contract required complete
identity coverage and prohibited post-hoc exclusion. Therefore v0.1 is not relabelled
`REJECT`.

This is a positive validation of the Oracle's execution-identity layer: without it,
a mixed compiled/fallback bank would have been reported as a valid semantic result.

The protocol repair is frozen in v0.2 with new prompt-disjoint banks and
`recompile_limit=64`; the event definition and strict endpoint are unchanged.

Evidence: `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_1/evaluation.json`.
