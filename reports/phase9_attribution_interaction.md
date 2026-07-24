# Phase 9 Attribution Interaction Audit

## Objective

Determine whether replayable natural clipping forks require interactions between Inductor settings or can be eliminated by a single configuration intervention.

## Controls

- Every target probe uses the frozen checkpoint, replay batch, token IDs, old logprob and advantage for that fork.
- Every setting uses a fresh Dynamo reset and independent Inductor cache; generated-code hash is the intervention canary.
- Repeated measurements must agree within `2e-6` in logprob and exactly in branch outcome.
- A combination is called interaction-only only when none of its constituent singleton settings eliminates the fork.

## Replayable Forks

| Fork | Settings | Effective singleton settings | Effective combinations | Interaction required |
|---|---:|---|---|---|
| clip-step5-grpo_000001_2817771126c0-t80 | 24/24 | max_fusion_size_1, max_fusion_size_2, no_persistent_reductions, no_pick_loop_orders | max_fusion_2_no_mix_reduction, max_fusion_2_deterministic, full_stabilize | False |
| clip-step11-grpo_000003_692fbb817526-t72 | 24/24 | max_fusion_size_1, max_fusion_size_2, no_pick_loop_orders | max_fusion_2_no_mix_reduction, max_fusion_2_deterministic, full_stabilize | False |
| clip-step11-grpo_000003_692fbb817526-t88 | 24/24 | max_fusion_size_2, no_pick_loop_orders | max_fusion_2_no_mix_reduction, max_fusion_2_deterministic, full_stabilize | False |
| clip-step14-grpo_000004_50bbbbeba833-t34 | 24/24 | max_fusion_size_1, no_persistent_reductions | none | False |
| clip-step14-grpo_000004_50bbbbeba833-t116 | 24/24 | no_persistent_reductions | none | False |

## Unreplayable Forks

_None._

## Interpretation

`5/5` replayable forks have at least one singleton elimination setting. The step-14 state was deterministically reconstructed and accepted only after 512/512 online rows and both eager/compile fork logprobs matched the canonical run exactly within the registered gates.

The evidence does not identify a unique source-level operator. Multiple settings can alter fusion partitioning, loop order or reduction scheduling and converge to a repaired branch despite different generated-code inventories. The supported attribution is therefore a compile scheduling class, with the exact effective subclasses listed in the structured artifact.

Configuration-class coverage and unique-operator attribution remain distinct: a 5/5 singleton scheduling repair rate does not identify five unique source operators.

## Artifacts

- `results/phase9_attribution_interaction.json`
- `scripts/phase9_attribution_interaction.py`
