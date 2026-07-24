# Phase A0 Signed Bias Audit

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- rollout token alignment: PASS
- signed delta uses alt minus ref: PASS
- debug FP32/FP16 pair excluded: PASS
- prompt-cluster bootstrap used: PASS

## Delta Self Control
Phase 1 aggregate self deltas are carried forward; process-independence is audited separately in A1.

## External Validity
This server uses Tesla T4 FP16 (u approximately 4.9e-4). It does not reproduce production BF16 kernels (u approximately 3.9e-3). An FP16 fork is evidence that the mechanism can occur at higher precision; a zero-FP16 result would be scoped to FP16 and would not rule out BF16. A BF16-hardware replication remains required.

## Alignment
| pair | source_rows | joined_rows | missing_rollout | token_mismatch | zero_advantage |
| --- | --- | --- | --- | --- | --- |
| online_compile | 51200 | 51200 | 0 | 0 | 11264 |

## Signed Bias
| pair | advantage_group | n_tokens | n_cases | signed_mean | signed_std | token_standard_error | t_statistic | t_pvalue | cluster_bootstrap_ci95_low | cluster_bootstrap_ci95_high | cluster_bootstrap_significant | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| online_compile | all | 51200 | 400 | 2.4221837520599364e-07 | 0.001252609488104298 | 5.535804145107245e-06 | 0.043754867198485606 | 0.9650999739924757 | -1.0205691680312157e-05 | 1.0575016960501669e-05 | False | alt_higher |
| online_compile | positive | 23424 | 183 | 3.77862505574044e-06 | 0.0010134142151560573 | 6.621501200939689e-06 | 0.5706598762232643 | 0.5682356964226587 | -6.958630567039948e-06 | 1.625133871706457e-05 | False | alt_higher |
| online_compile | negative | 16512 | 129 | -4.609310349752736e-06 | 0.0018461918558573347 | 1.436736055985921e-05 | -0.3208181718937737 | 0.7483522478601786 | -3.3944042385086535e-05 | 2.188170256540754e-05 | False | alt_lower |

## Advantage Sign Association
| pair | positive_minus_negative | ci95_low | ci95_high | significant |
| --- | --- | --- | --- | --- |
| online_compile | 8.387935405493177e-06 | -2.0746355276275034e-05 | 3.966227413352433e-05 | False |

## Conclusion
Neither claim pair shows a signed mean distinguishable from zero under prompt-cluster bootstrap.

A significant all-token mean supports directional path bias. Positive/negative advantage rows test association, not causality; differing group means can also reflect token-distribution differences.
