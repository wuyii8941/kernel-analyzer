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
| hf_vllm | 512 | 512 | 0 | 0 | 0 |

## Signed Bias
| pair | advantage_group | n_tokens | n_cases | signed_mean | signed_std | token_standard_error | t_statistic | t_pvalue | cluster_bootstrap_ci95_low | cluster_bootstrap_ci95_high | cluster_bootstrap_significant | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hf_vllm | all | 512 | 4 | -0.14295375091677176 | 0.7814199802145744 | 0.03453421043527396 | -4.139482244272069 | 4.0713307535658534e-05 | -0.18747488160463366 | -0.09979610116909754 | True | alt_lower |
| hf_vllm | positive | 128 | 1 | -0.08333350927100014 | 0.4908346160591098 | 0.0433840606820615 | -1.9208323969880716 | 0.05699419497792428 | -0.08333350927100014 | -0.08333350927100014 | True | alt_lower |
| hf_vllm | negative | 384 | 3 | -0.16282716479869563 | 0.8562808490935245 | 0.04369689909992206 | -3.7262864906353523 | 0.00022358660888915593 | -0.20530095444291874 | -0.13399666308977842 | True | alt_lower |

## Advantage Sign Association
| pair | positive_minus_negative | ci95_low | ci95_high | significant |
| --- | --- | --- | --- | --- |
| hf_vllm | 0.0794936555276955 | 0.050663153818778284 | 0.1219674451719186 | True |

## Conclusion
At least one claim pair has a directionally biased signed delta under prompt-cluster bootstrap.

A significant all-token mean supports directional path bias. Positive/negative advantage rows test association, not causality; differing group means can also reflect token-distribution differences.
