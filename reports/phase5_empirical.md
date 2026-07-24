# Phase 5 Executed Bugs: Empirical-Only

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- three altered model operations executed: PASS
- reference path self exact: PASS
- same single-sample shape for reference and altered path: PASS
- token and rollout alignment: PASS
- analytic legal bound: FAIL; certified bug classification disabled

## Delta Self Control
Every recomputed reference token matched exactly across two runs.

## External Validity
Executed on the step-5 T4 FP16 snapshot. Results are empirical anomaly sensitivity only.

## Results
| bug | rows | delta_mean | delta_p50 | delta_p99 | delta_max | clip_branch_forks | certified_bug_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logsumexp_reduction_off_by_one | 1024 | 2.8491694633584386 | 1.3583464175462723 | 13.126832340953861 | 18.58281135559082 | 313 | 0 |
| attention_mask_missing_column | 1024 | 0.19889389605451324 | 0.02564266324043274 | 2.266156016588207 | 15.70742642879486 | 104 | 0 |
| fp16_logits_intermediate | 1024 | 65480.72015238697 | 65481.645724751055 | 65485.390844535825 | 65486.082656145096 | 686 | 0 |

All rows retain region `unknown`; no empirical threshold is promoted to a legal bug boundary.
