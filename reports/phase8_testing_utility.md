# Fork Signal as an Empirical Testing Utility

## Objective

This batch addresses RQ5 by comparing decision-semantic fork alarms with fixed delta tolerances and delta-magnitude ranking on the three existing artificial altered operations.

## Dataset and Scope

- 39,936 legal eager/compile token comparisons.
- 3,072 rows from three deliberately altered operations.
- No analytic legal bound is available. The altered rows are empirical anomalies, not certified or historical real bugs.
- The five natural legal-path forks are counted as false alarms only for the narrow label "artificial injection present"; they remain real semantic-risk events.

## Result

| Method | Alarms | Precision | Recall |
|---|---:|---:|---:|
| fork signal | 1,108 | 99.55% | 35.90% |
| abs(delta) > 1e-4 | 5,195 | 56.96% | 96.32% |
| abs(delta) > 1e-3 | 4,237 | 66.96% | 92.35% |
| abs(delta) > 1e-2 | 2,945 | 90.90% | 87.14% |
| abs(delta) > 1e-1 | 2,329 | 100% | 75.81% |

Fork recall varies strongly by injection: 10.16% for the missing attention-mask column, 30.57% for the logsumexp reduction change and 66.99% for the FP16-logits intermediate.

Delta ranking is perfect for the first 2,000 inspected rows because these artificial injections often produce very large numerical changes. This benchmark therefore does not support a claim that fork ranking is universally superior to magnitude ranking.

## Interpretation

Fork signal offers a low-alarm, high-precision semantic filter, but it intentionally ignores mismatches that do not cross the measured clipping boundary. Fixed tolerance offers higher anomaly recall at a substantially larger review burden. The two signals serve different purposes and should be combined: delta ranking for broad implementation anomaly discovery, fork risk for prioritizing mismatches with immediate training semantics.

## Artifacts

- Result: `results/phase8_testing_utility.json`
- Implementation: `scripts/phase8_testing_utility.py`

## Next Decision

**REVISE.** Retain the testing application as a secondary contribution. Add real historical errors and less extreme injections before making comparative SE claims.
