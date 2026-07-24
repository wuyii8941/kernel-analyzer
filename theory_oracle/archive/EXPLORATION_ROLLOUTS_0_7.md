# Existing-Data Oracle Exploration

## Scope

This is an exploratory reanalysis of the online scan, restricted to rollout batches [0,8). Compiled-minus-eager is an implementation-relative discrepancy, not a truth-relative error. Results are conditional on one canonical trajectory and the recorded T4 FP16 configuration.

## Coverage

| rows | applicable_rows | cases | rollouts | optimizer_steps |
| --- | --- | --- | --- | --- |
| 4096 | 4096 | 32 | 8 | 8 |

## Numerical Endpoints

| signed_n | signed_mean | signed_std | signed_p01 | signed_p50 | signed_p99 | signed_min | signed_max | absolute_n | absolute_mean | absolute_std | absolute_p01 | absolute_p50 | absolute_p99 | absolute_min | absolute_max | event_oriented_n | event_oriented_mean | event_oriented_std | event_oriented_p01 | event_oriented_p50 | event_oriented_p99 | event_oriented_min | event_oriented_max | fraction_signed_positive | fraction_signed_negative | fraction_signed_zero | self_ref_max | self_alt_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 3.0277297e-06 | 0.0044291398 | -0.014432335 | 0 | 0.01428709 | -0.026567459 | 0.030731201 | 4096 | 0.0021303156 | 0.0038830318 | 0 | 0.00018310547 | 0.015880013 | 0 | 0.030731201 | 4096 | 4.0190294e-05 | 0.0044289585 | -0.014233589 | 0 | 0.014573193 | -0.030731201 | 0.026292801 | 0.41625977 | 0.41894531 | 0.16479492 | 0 | 0 |

## Semantic Endpoints

| applicable_tokens | reference_event_rate | compiled_event_rate | up_flip_count | down_flip_count | semantic_disagreement | directional_semantic_shift | compiled_minus_reference_event_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 0.059326172 | 0.060058594 | 4 | 1 | 0.0012207031 | 0.00073242188 | 0.00073242188 |

## Training-Stage Conditioning

| stage | rollouts | tokens | signed_mean | signed_std | oriented_mean | up_flips | down_flips |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early | 3 | 1536 | -5.3991874e-05 | 0.0042880618 | 6.2753757e-05 | 1 | 0 |
| middle | 3 | 1536 | 6.4236422e-05 | 0.0042566331 | 5.7439009e-05 | 3 | 1 |
| late | 2 | 1024 | -3.2559037e-06 | 0.0048722334 | -1.9527972e-05 | 0 | 0 |

## Boundary Conditioning

| distance_bin | tokens | signed_mean | abs_delta_mean | oriented_mean | up_flips | down_flips | disagreement_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1e-05,0.0001) | 2 | -0.0063314438 | 0.0063314438 | 0.0063314438 | 1 | 0 | 0.5 |
| [0.0001,0.001) | 2 | -0.0031099319 | 0.0031099319 | -0.0031099319 | 0 | 0 | 0 |
| [0.001,0.01) | 32 | -4.0113926e-05 | 0.0039801002 | 0.00072365999 | 2 | 0 | 0.0625 |
| [0.01,0.1) | 380 | -0.00023670698 | 0.0054090751 | -0.00042181015 | 1 | 1 | 0.0052631579 |
| [0.1,inf) | 3680 | 3.3292563e-05 | 0.0017728474 | 8.0246511e-05 | 0 | 0 | 0 |

## Descriptive Variance Decomposition

The between-case and between-rollout terms describe heterogeneity across sampled units. They are not runtime variance components.

| level | groups | total_variance | between_group_variance | mean_within_group_variance | identity_residual | between_fraction | unweighted_group_mean_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case | 32 | 1.961249e-05 | 1.3925553e-07 | 1.9473235e-05 | 0 | 0.0071003493 | 0.00037914066 |
| rollout | 8 | 1.961249e-05 | 3.1515294e-08 | 1.9580975e-05 | 0 | 0.0016068992 | 0.00018978272 |

## Block-Bootstrap Intervals

| estimand | block_length_rollouts | draws | ci95_low | ci95_high |
| --- | --- | --- | --- | --- |
| signed_mean | 2 | 5000 | -0.00011487398 | 0.00013934016 |
| event_oriented_mean | 2 | 5000 | -2.3930799e-05 | 0.00011577737 |
| semantic_disagreement | 2 | 5000 | 0.00024414062 | 0.0026855469 |
| directional_semantic_shift | 2 | 5000 | 0 | 0.0014648438 |

## Interpretation Limits

- Attached self-pair maxima describe the observed deterministic floor; they do not estimate a general runtime-noise law.
- The scan has no gradient/update/next-state endpoints, so it cannot select a complete impact Oracle by itself.
- Five clipping disagreements are too sparse for a stable universal event-rate claim.
- All inferential numbers remain conditional on this single serial trajectory.
