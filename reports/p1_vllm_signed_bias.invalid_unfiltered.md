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
| hf_vllm | 1024 | 1024 | 0 | 0 | 0 |

## Signed Bias
| pair | advantage_group | n_tokens | n_cases | signed_mean | signed_std | token_standard_error | t_statistic | t_pvalue | cluster_bootstrap_ci95_low | cluster_bootstrap_ci95_high | cluster_bootstrap_significant | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hf_vllm | all | 1024 | 8 | -0.1070928984835116 | 0.7267913778517551 | 0.022712230557867346 | -4.715208319616825 | 2.7488920207952135e-06 | -0.14511105550068287 | -0.06494271019978175 | True | alt_lower |
| hf_vllm | positive | 256 | 2 | -0.09923179235827106 | 0.5028165270494537 | 0.031426032940590855 | -3.157630253422797 | 0.0017820061107362015 | -0.11513007544554199 | -0.08333350927100014 | True | alt_lower |
| hf_vllm | negative | 768 | 6 | -0.1097132671919251 | 0.7876851762073557 | 0.028423140532499665 | -3.8599980556855233 | 0.00012293562402244885 | -0.15822577726176826 | -0.0563180830202043 | True | alt_lower |

## Advantage Sign Association
| pair | positive_minus_negative | ci95_low | ci95_high | significant |
| --- | --- | --- | --- | --- |
| hf_vllm | 0.01048147483365404 | -0.04912049222469354 | 0.06553434230619987 | False |

## Conclusion
At least one claim pair has a directionally biased signed delta under prompt-cluster bootstrap.

A significant all-token mean supports directional path bias. Positive/negative advantage rows test association, not causality; differing group means can also reflect token-distribution differences.
