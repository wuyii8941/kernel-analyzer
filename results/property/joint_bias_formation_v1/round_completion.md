# Joint bias round completion

Status: **COMPLETE_WITH_CLAIM_BOUNDARIES**.

## What completed

- Live four-counterfactual consequence: **12/12** cases, all recurrence residuals exactly closed.
- Observed regimes: `FEEDBACK_SUSTAINED=11, MIXED_LOCAL_AND_FEEDBACK_PERSISTENCE=1`.
- Exact raw response replay: saved-P and Qwen3-VL SiLU, 32 states each.
- DeepSeek BF16 antithetic capture: 3/3 executed, but 3/3 reflected endpoints are not exactly representable and therefore remain `UNRESOLVED_REPRESENTABILITY`.
- Frozen Gemma NEW_IMPL held-out evidence consolidated.

## Main result

The screen-negative sample does not reveal a new persistent local-source family. Local increments are mostly diffusive, while persistent actual drift is usually carried by the closed-loop feedback term. This is a real trajectory effect, but it is not evidence that the sampled operator source itself has Flash-style persistent directionality.

The 16-step actual-amplification prefix preserves the side of the diffusive boundary in 12/12 cases and has Pearson correlation 0.826 with the 32-step value. This supports it as a triage feature, not yet as a universal classifier.

The held-out campaign confirms source-negative predictions on Gemma RMSNorm/attention, and independently finds Adam-moment feedback. Because the frozen feedback predictor abstained, the full three-factor predictor remains unresolved.

## Claim boundary

This round closes the requested execution gaps. It does **not** establish a universal property, does not label feedback-sustained drift as operator-local source bias, and does not impute exact `-epsilon` where BF16 representability forbids it.
