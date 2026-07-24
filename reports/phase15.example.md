# Phase 1.5 Attribution Ladder

## Confound Checklist
- one_variable_changed_per_level: required by measurement run
- same_samples_and_tokens: required by measurement run
- activation_dump_compact: PASS
- large_tensor_outputs_avoided: PASS
- additive_percent_claim_disabled: PASS

## Delta Self Control
Phase 1.5 consumes Phase 1 self-consistent path pairs.

## Summary
Attribution ladder is incomplete. These measurements are sensitivity comparisons, not an additive decomposition.

## Attribution
| level | variable | mechanism | first_observed_diff_l2 | max_activation_diff_l2 | propagation_gain_first_to_last | final_logprob_delta | relative_to_composite_percent | additive_attribution_valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L6 | torch.compile | mixed | 0.0 | 0.0 | None | 0.1 | 100.0 | False |
| L1 | attention backend | algorithm_structure | 0.0 | 0.0 | None | 0.2 | 200.0 | False |
| L4 | log_softmax precision | rounding_precision | 0.0 | 0.0 | None | 0.3 | 300.0 | False |
