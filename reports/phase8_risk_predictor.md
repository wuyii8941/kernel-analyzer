# Held-Out Fork-Risk Prediction

## Objective

This batch addresses RQ3: predict clipping-fork risk before executing the alternative path, using signed decision margin and a calibrated path-delta distribution.

## Protocol

The 39,936 applicable clipping tokens are grouped into 78 rollout groups. Each group is tested once under leave-one-rollout-out cross-fitting. All other groups are split into fit groups for signed-delta distributions and calibration groups for count scaling. Tokens from the test rollout never enter fit or calibration.

The conditional model stratifies signed delta by advantage sign and 16-token position bucket. It is compared with fixed tolerance, observed absolute delta, margin-only ranking, a global independence distribution, and an observed signed-crossing oracle. Absolute delta and the oracle require the alternative path and are retrospective controls, not deployable pre-execution predictors.

## Conditional Estimand

The reported fork rate is conditional on `(Qwen3-0.6B, frozen online GRPO checkpoints represented in the canonical scan, eps=0.2, Tesla T4 FP16 autocast, PyTorch 2.13 nightly, eager versus default Inductor compile with MATH-locked SDPA)`. It is not an unconditional model or hardware failure rate.

For a token with signed reference margin `m`, the pre-execution model estimates the probability that a signed path perturbation `D` crosses zero: `P(m <= 0 < m + D or m > 0 >= m + D)`. The empirical distribution of `D` comes only from non-test rollout groups. The test rollout contributes neither its deltas nor labels to fitting or calibration.

The earlier `3.758` count was an in-sample independence estimate used before the held-out protocol was frozen. The current report uses the leave-one-rollout-out estimates below (`1.919` global independence; `3.154` signed conditional calibrated) and does not treat `3.758` as held-out performance.

## Result

There are only 5 positive tokens in rollout groups 1, 3 and 4.

| Method | Average precision | Top-1% recall | Predicted fork count |
|---|---:|---:|---:|
| fixed `abs(delta)>1e-3` | 0.0036 | 0.20 | n/a |
| observed absolute delta | 0.0201 | 0.60 | n/a |
| margin only | 0.0900 | 1.00 | n/a |
| global independence distribution | 0.0585 | 1.00 | 1.919 |
| signed conditional calibrated | 0.0270 | 1.00 | 3.154 |
| observed signed crossing oracle | 1.0000 | 1.00 | n/a |

Observed fork count is 5. The signed conditional model improves count calibration over the global independence baseline but ranks positives worse than margin-only and the independence model. Its 10-bin ECE is `4.62e-5`, versus `7.72e-5` for independence; these small values are dominated by the extreme class imbalance and must not be read without the count error.

The fixed `1e-3` tolerance flags 1,400 tokens, finds 4 of 5 forks, and has precision `0.286%`. The top 1% (400 tokens) of margin, independence and signed-conditional risk each contains all five forks, with precision `1.25%`.

Rollout-cluster bootstrap intervals are wide: the signed conditional predicted-count interval is approximately `[2.304, 4.147]`, while the observed-count interval is `[0, 11]`. The latter reflects only three positive rollout clusters.

This is leave-one-rollout-out, not “leave one positive fork out.” Estimating a path-delta distribution from only the other four positive fork deltas would condition on the outcome and provide an unusably small, biased perturbation sample.

## Artifacts

- Predictions: `results/phase8_risk_predictions.jsonl`
- Summary: `results/phase8_risk_summary.json`
- Implementation: `scripts/phase8_risk_predictor.py`

## Interpretation

The risk signal is useful for budget allocation: all positives lie in the top 1% under margin-based pre-execution scores. The proposed conditional signed model is not yet superior for ranking. The current data support a narrower claim: signed distributions can improve expected-count calibration, while simple distance to the decision boundary remains the strongest tested ranking feature.

No global signed bias is assumed. Direction enters only through the crossing probability for each token's signed margin.

## Next Decision

**REVISE.** Retain the predictor and held-out protocol, but do not claim model superiority. More positive rollout groups, another checkpoint and BF16 hardware are required before selecting a final predictor.
