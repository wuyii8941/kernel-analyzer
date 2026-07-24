# Qwen3 GRPO Branch-Repair v0.3 Invalid Findings — 2026-07-17

## Verdict

The frozen v0.3 execution is **INVALID**, and the proposed v0.4 repair execution is
**not licensed by its endpoint preflight**.

This is not evidence of zero branch effect.  It is evidence that the selected
`no_grad` measurement event is not the branch treatment realized by the
gradient-enabled compiled training transition.

## v0.3 failure

The manifest, selected event, model snapshot, four-response batch, 512 token
alignment and target index all passed the independent preflight audit.  Execution
then stopped at the frozen A-reference anchor gate:

```text
parent eager target:       -3.8464813232421875
raw reloaded A target:     -3.8464815616607666
difference:                -2.384185791015625e-7
```

The failure was retained as `INVALID`; no tolerance was selected after seeing the
result.  B and C were not interpreted.

## Diagnosed missing execution context

The mismatch is not explained by autograd mode, gradient checkpointing or
same-process repeat variability.  Across all four combinations, the complete
512-token raw-reloaded tensor was exactly self-stable and identical across modes.

The original FP16 Trainer model was prepared by Accelerate, which converts the
model output structure to FP32 before TRL slices logits and evaluates
`selective_log_softmax`.  The v0.3 reloader omitted that wrapper.  Restoring it:

- exactly reproduced the parent eager target in both `no_grad` and grad-enabled
  contexts;
- was exactly self-stable in both contexts;
- changed 505 of 512 token log-probabilities relative to the raw reloader, with
  maximum absolute difference about `2.03e-6`.

Therefore the v0.3 declaration that the scorer was Trainer-equivalent was false.
The omitted wrapper was outcome-relevant implementation context, not runtime
variance.

## v0.4 pair-endpoint preflight

With the Accelerate output-FP32 wrapper restored, eager reproduced in both
execution contexts.  Compiled behavior separated by specialization:

| Realization | target log-probability | ratio | negative-advantage clip? |
|---|---:|---:|---|
| eager, grad-enabled | -3.8464813232421875 | 0.7851501 | yes |
| compiled, `no_grad` | -3.8221607208251953 | 0.8044795 | no |
| compiled, grad-enabled | -3.8328704833984375 | 0.7959097 | yes |

Each compiled context was exactly self-stable across two measured repeats.  The
compiled `no_grad` value exactly reproduced the parent event, but the grad-enabled
compiled specialization did not.  Most importantly, eager and grad-enabled
compiled executions selected the same clipping branch.

Consequently the selected witness is a valid disagreement for the declared
online measurement paths, but not a clipping disagreement for the differentiable
one-step transition.  A B/C branch-repair contrast at this witness would have no
valid branch treatment and would confound execution-specialization change with
branch intervention.

A subsequently frozen full-state scan ruled out merely choosing another token in
the same batch.  Among all 512 nonzero-advantage token objectives:

- eager versus compiled `no_grad` had exactly one disagreement, flat index 228;
- eager versus compiled grad-enabled had zero disagreements.

Thus this entire frozen state is unsuitable as a transition-context branch-repair
witness.

## Oracle consequence

For a semantic event to be used as a cause of a downstream transition endpoint,
the event must be realized in the same execution context that computes that
transition.  A `no_grad` compatibility event may remain a valid measurement
endpoint, but it cannot be transported into a backward/update claim without an
explicit endpoint-realization gate.

The next Qwen update-impact bank must therefore discover and freeze events directly
on gradient-enabled eager/compiled scorer realizations.  It must not select events
from the existing `no_grad` bank and then assume that the branch survives autograd
specialization.

## Claim boundary

These diagnostics establish neither compiler incorrectness nor population
prevalence.  They also do not invalidate all historical Qwen branch witnesses;
they reject this selected v0.3 witness as evidence for this declared one-step
branch-repair estimand.

## Evidence

- frozen executor result:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/branch_repair_v0_3.json`;
- independent audit:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/branch_repair_v0_3_audit.json`;
- unified fail-closed result:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/branch_repair_unified_result_v0_3_invalid.json`;
- reference-context diagnostic:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/reference_endpoint_upcast_diagnostic_v0_3.json`;
- pair-endpoint preflight:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/pair_endpoint_diagnostic_v0_4.json`;
- full-state grad-event scan:
  `results/training_step_oracle/qwen3_grpo_training_control_confirmation_v0_2/grad_event_scan_v0_4.json`.
