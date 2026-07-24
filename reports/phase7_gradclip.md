# Phase 7 Gradient-Clipping Trigger

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- same_step5_snapshot: PASS
- same_four_response_batch: PASS
- all_online_tokens_replayed_within_explicit_2e-6_tolerance: PASS
- backend_only_difference: PASS
- math_sdpa_locked_both_paths: PASS
- compile_warmed_before_measurement: PASS
- trl_grpo_loss_normalization_matched_for_equal_128_token_responses: PASS
- no_parameter_update_between_self_runs: PASS

## Delta Self Control
Two backward passes per path: ref grad-norm delta=0, alt grad-norm delta=0.

## Summary
The standard max_grad_norm=1.0 trigger was evaluated on a step-5 replay aligned to all 512 online token logprobs within the explicit `2e-6` tolerance, with zero PPO branch changes. A midpoint threshold is reported only as a controlled detector calibration.

## Natural Decision
| natural_threshold | ref_grad_norm | alt_grad_norm | grad_norm_delta | ref_clip_trigger | alt_clip_trigger | natural_actual_fork | controlled_threshold | controlled_actual_fork |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 7.829676405012935 | 7.814733985902471 | 0.014942419110464122 | True | True | False | 7.822205195457704 | True |

## Interpretation
The natural result is a decision-boundary observation, not a fragile/bug classification, because Phase 2 did not produce a usable legal bound. The controlled midpoint threshold is not part of the natural claim.

## Eager Training Coverage

The canonical Trainer log contains 300 pre-clip global gradient norms. At the natural threshold `1.0`, 234 steps trigger and 66 do not; all 66 non-trigger values are exactly zero. The smallest positive norm is `4.200996`, the minimum decision margin is `1.0`, and no step has margin below `0.2`. This is demand-side coverage for the eager path, not a paired backend certification.

The reproducible summary is `results/phase7_gradclip_ref_history.json`.

## External Validity
This audit uses FP16 autocast on Tesla T4. T4 has no native BF16 tensor-core support; a zero-fork FP16 result cannot rule out BF16 forks.
