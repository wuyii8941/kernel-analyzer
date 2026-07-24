# Existing-Data Oracle Exploration

## Scope

This is an exploratory reanalysis of the complete online scan. Compiled-minus-eager is an implementation-relative discrepancy, not a truth-relative error. Results are conditional on one canonical trajectory and the recorded T4 FP16 configuration.

## Coverage

| rows | applicable_rows | cases | rollouts | optimizer_steps |
| --- | --- | --- | --- | --- |
| 51200 | 39936 | 400 | 100 | 100 |

## Numerical Endpoints

| signed_n | signed_mean | signed_std | signed_p01 | signed_p50 | signed_p99 | signed_min | signed_max | absolute_n | absolute_mean | absolute_std | absolute_p01 | absolute_p50 | absolute_p99 | absolute_min | absolute_max | event_oriented_n | event_oriented_mean | event_oriented_std | event_oriented_p01 | event_oriented_p50 | event_oriented_p99 | event_oriented_min | event_oriented_max | fraction_signed_positive | fraction_signed_negative | fraction_signed_zero | self_ref_max | self_alt_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 39936 | 3.1053638e-07 | 0.0014183049 | -0.0037011147 | 0 | 0.0038059235 | -0.026567459 | 0.030731201 | 39936 | 0.00021849391 | 0.0014013736 | 0 | 0 | 0.0079595566 | 0 | 0.030731201 | 39936 | 4.1220815e-06 | 0.0014182989 | -0.0036914825 | 0 | 0.0038302422 | -0.030731201 | 0.026292801 | 0.042693309 | 0.04296875 | 0.91433794 | 0 | 0 |

## Semantic Endpoints

| applicable_tokens | reference_event_rate | compiled_event_rate | up_flip_count | down_flip_count | semantic_disagreement | directional_semantic_shift | compiled_minus_reference_event_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 39936 | 0.02353766 | 0.02361278 | 4 | 1 | 0.00012520032 | 7.5120192e-05 | 7.5120192e-05 |

## Training-Stage Conditioning

| stage | rollouts | tokens | signed_mean | signed_std | oriented_mean | up_flips | down_flips |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early | 33 | 16384 | 7.5693242e-07 | 0.0022143675 | 1.0047574e-05 | 4 | 1 |
| middle | 33 | 14336 | 0 | 0 | 0 | 0 | 0 |
| late | 34 | 9216 | 0 | 0 | 0 | 0 | 0 |

## Boundary Conditioning

| distance_bin | tokens | signed_mean | abs_delta_mean | oriented_mean | up_flips | down_flips | disagreement_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [1e-05,0.0001) | 3 | -0.0042209625 | 0.0042209625 | 0.0042209625 | 1 | 0 | 0.33333333 |
| [0.0001,0.001) | 18 | -0.00034554799 | 0.00034554799 | -0.00034554799 | 0 | 0 | 0 |
| [0.001,0.01) | 144 | -8.9142058e-06 | 0.0008844667 | 0.00016081333 | 2 | 0 | 0.013888889 |
| [0.01,0.1) | 1866 | -4.8203995e-05 | 0.0011015265 | -8.5899173e-05 | 1 | 1 | 0.0010718114 |
| [0.1,inf) | 37905 | 3.2322024e-06 | 0.00017211656 | 7.7907178e-06 | 0 | 0 | 0 |

## Descriptive Variance Decomposition

The between-case and between-rollout terms describe heterogeneity across sampled units. They are not runtime variance components.

| level | groups | total_variance | between_group_variance | mean_within_group_variance | identity_residual | between_fraction | unweighted_group_mean_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case | 312 | 2.0115383e-06 | 1.4283462e-08 | 1.9972548e-06 | 4.2351647e-22 | 0.0071007658 | 0.00011970543 |
| rollout | 78 | 2.0115383e-06 | 3.2331817e-09 | 2.0083051e-06 | 0 | 0.001607318 | 5.722911e-05 |

## Block-Bootstrap Intervals

| estimand | block_length_rollouts | draws | ci95_low | ci95_high |
| --- | --- | --- | --- | --- |
| signed_mean | 5 | 5000 | -7.7980833e-06 | 1.0598929e-05 |
| event_oriented_mean | 5 | 5000 | -5.6017859e-07 | 1.2403206e-05 |
| semantic_disagreement | 5 | 5000 | 0 | 0.00037560096 |
| directional_semantic_shift | 5 | 5000 | 0 | 0.00022536058 |

## Interpretation Limits

- Attached self-pair maxima describe the observed deterministic floor; they do not estimate a general runtime-noise law.
- The scan has no gradient/update/next-state endpoints, so it cannot select a complete impact Oracle by itself.
- Five clipping disagreements are too sparse for a stable universal event-rate claim.
- All inferential numbers remain conditional on this single serial trajectory.
