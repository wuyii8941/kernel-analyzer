# Phase 9 Testing Utility

## Objective

Compare clipping-fork alarms with fixed absolute-delta thresholds and delta ranking for identifying artificial execution mutations.

## Label Contract

Only artificial mutation rows are positive. Legal eager/compile rows, including audited natural semantic forks, are negative for this mutation-classification task.
This is intentionally different from semantic-risk detection, where an audited natural fork is itself a positive event.

## Result

The evaluation contains `39936` legal-path rows and `7680` rows from `15` artificial mutations.

| Method | Alarms | Precision | Recall |
|---|---:|---:|---:|
| Fork signal | 425 | 0.9882 | 0.0547 |
| `abs_delta_gt_0.0001` | 7539 | 0.7034 | 0.6905 |
| `abs_delta_gt_0.001` | 5191 | 0.7303 | 0.4936 |
| `abs_delta_gt_0.01` | 1888 | 0.8581 | 0.2109 |
| `abs_delta_gt_0.1` | 576 | 1.0000 | 0.0750 |
| `abs_delta_gt_1` | 500 | 1.0000 | 0.0651 |

## Matched Alarm Budget

At `425` alarms, delta ranking has precision `1.0000` and recall `0.0553`. Fork signal has precision `0.9882` and recall `0.0547`.

On this catalog, delta ranking dominates fork signal for artificial-mutation classification at the matched alarm budget. The preregistered expectation that fork would have higher precision is not supported.

## Interpretation

Fork signal remains a tolerance-free indicator that a training decision changed, but that property does not make it a universal classifier of implementation mutations. It deliberately ignores mutations that alter numerical amplitude without crossing the frozen clipping boundary.

The five audited eager/compile natural forks count as false alarms under the mutation label contract, even though they are true semantic-risk events. Therefore this result evaluates testing utility for mutation identification, not the validity of the natural-fork existence claim.

## Artifacts

- `results/phase9_testing_utility.json`
- `results/phase9_mutations/all_mutation_rows.jsonl`
- `results/phase4_certificates.jsonl`
