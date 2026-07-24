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
| eager_compile | 51200 | 51200 | 0 | 0 | 11264 |
| eager_attention_sdpa_math | 51200 | 51200 | 0 | 0 | 11264 |

## Signed Bias
| pair | advantage_group | n_tokens | n_cases | signed_mean | signed_std | token_standard_error | t_statistic | t_pvalue | cluster_bootstrap_ci95_low | cluster_bootstrap_ci95_high | cluster_bootstrap_significant | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eager_compile | all | 51200 | 400 | -7.675250364793773e-05 | 0.005122270052530988 | 2.2637449307583967e-05 | -3.3905102383696653 | 0.0006981531236891248 | -0.0001247845849830004 | -2.8373788615685166e-05 | True | alt_lower |
| eager_compile | positive | 23424 | 183 | -2.758986117408195e-05 | 0.004638892872715563 | 3.0309851853604197e-05 | -0.9102605089374988 | 0.36269450663473674 | -9.249404338230495e-05 | 3.628803257220646e-05 | False | alt_lower |
| eager_compile | negative | 16512 | 129 | -7.926476601670492e-05 | 0.006126336453862866 | 4.767613098519637e-05 | -1.6625670829144448 | 0.09641810605486877 | -0.0001804220372918202 | 1.9413018455716016e-05 | False | alt_lower |
| eager_attention_sdpa_math | all | 51200 | 400 | -6.106921259893496e-05 | 0.005901353288084426 | 2.6080543301137668e-05 | -2.3415621328820646 | 0.019207042688076915 | -0.00011573866803584612 | -5.949184647011885e-06 | True | alt_lower |
| eager_attention_sdpa_math | positive | 23424 | 183 | -9.329745238982938e-05 | 0.005156774392145274 | 3.36936144371202e-05 | -2.768995073649437 | 0.005627370199498753 | -0.00016448480381933876 | -2.0406618981803602e-05 | True | alt_lower |
| eager_attention_sdpa_math | negative | 16512 | 129 | -2.955989477445033e-05 | 0.0072047741013346305 | 5.606870539364426e-05 | -0.5272084412671516 | 0.5980559308028971 | -0.00014787872117566318 | 8.957469330135153e-05 | False | alt_lower |

## Advantage Sign Association
| pair | positive_minus_negative | ci95_low | ci95_high | significant |
| --- | --- | --- | --- | --- |
| eager_compile | 5.167490484262297e-05 | -6.671476153720841e-05 | 0.00016630794433721848 | False |
| eager_attention_sdpa_math | -6.373755761537906e-05 | -0.00020172432972238286 | 7.485402404280721e-05 | False |

## Conclusion
At least one claim pair has a directionally biased signed delta under prompt-cluster bootstrap.

A significant all-token mean supports directional path bias. Positive/negative advantage rows test association, not causality; differing group means can also reflect token-distribution differences.
