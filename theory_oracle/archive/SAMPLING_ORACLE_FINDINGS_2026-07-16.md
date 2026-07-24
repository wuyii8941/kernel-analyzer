# Sampling-distribution Oracle Findings — 2026-07-16

## Bottom line

At temperature 0.7, Qwen eager and compiled induce small but nonzero categorical-distribution discrepancies at the frozen final response positions. That nonnegative discrepancy scale reproduces on held-out sequence states. Signed top-1-probability and entropy shifts do not reproduce in direction.

Algorithmic sampling variability is much larger than the implementation-relative distribution shift. Therefore a single sampled-token mismatch is not a clean implementation Oracle. The primary sampling endpoint must compare the categorical laws, while RNG variability and finite-state uncertainty are reported separately.

The state populations, coupling diagnostics and sample budgets were frozen in `SAMPLING_ORACLE_CONTRACT_2026-07-16.md` before the formal outputs were read.

## Exact distribution comparison

| Endpoint | Discovery, 32 states | Confirmation, 32 held-out states |
|---|---:|---:|
| Mean total variation (TV), state-bootstrap 95% CI | 1.053e-3 [5.087e-4, 1.669e-3] | 6.593e-4 [2.164e-4, 1.201e-3] |
| Median TV | 4.54e-6 | 2.45e-5 |
| Maximum TV | 5.558e-3 | 5.475e-3 |
| States with TV above 1e-3 | 9/32 | 4/32 |
| Mean Jensen–Shannon divergence | 3.295e-6 | 2.043e-6 |
| Argmax disagreements | 0/32 | 0/32 |
| Mean eager-top-1 probability shift | -2.036e-4 | +1.292e-4 |
| Mean entropy shift, state-bootstrap 95% CI | +1.310e-4 [-7.508e-4, 1.017e-3] | -4.441e-4 [-1.036e-3, 5.207e-5] |

The TV endpoint is directional-free and reproduces qualitatively. Its distribution is highly state-conditioned: the median is orders of magnitude below the maximum, so the mean alone is incomplete. The signed endpoints change direction and their entropy intervals include zero. This does not support a persistent sampling-direction bias for the named state population.

All compiled-execution identity, single-graph, no-measurement-recompile and self-pair gates passed. Model execution was deterministic under the declared protocol.

## Algorithmic RNG is a different source of variability

| Endpoint | Discovery | Confirmation |
|---|---:|---:|
| Eager within-distribution independent-draw disagreement | 17.78% | 16.66% |
| Compiled within-distribution independent-draw disagreement | 17.77% | 16.65% |
| Cross-distribution independent-draw disagreement | 17.77% | 16.65% |
| Mean TV | 0.105% | 0.066% |
| Within-distribution disagreement / mean TV | about 169x | about 253x |
| Common-uniform inverse-CDF empirical disagreement | 0.400% | 0.188% |

Independent-draw disagreement is mainly a property of sampling from a non-degenerate categorical distribution. It remains large even if eager and compiled distributions are identical. Conversely, the minimum attainable paired mismatch over all couplings is exactly TV, not the mismatch rate of arbitrary independent or common-random-number streams.

The inverse-CDF common-uniform result is only a diagnostic for one coupling and vocabulary ordering. It is larger than TV because inverse-CDF coupling is not generally maximal. It cannot replace a distribution metric.

Four independent streams of 1024 draws per state and implementation were used to calibrate Monte Carlo behavior. Top-1 frequency errors were around 0.4–0.5 percentage points on average; this is finite-draw uncertainty, not model-execution variability and not implementation drift.

## Three quantities that must remain separate

1. **Implementation-relative distribution shift:** exact TV, JS/KL and probability-mass changes computed from frozen logits.
2. **Algorithmic RNG variability:** randomness of realized tokens conditional on one fixed categorical law.
3. **Sampling uncertainty:** uncertainty in population summaries because only 32 sequence states per split were observed, plus Monte Carlo error in diagnostic draw frequencies.

Calling all three “variance” would make the Oracle uninterpretable.

## Oracle consequence

For stochastic sampling, the defensible Oracle output is a distribution profile:

- exact state-conditioned distribution distances;
- signed probability/entropy changes only when their direction survives confirmation;
- tail and quantile information across states;
- a declared coupling only for paired-token diagnostics;
- separate RNG calibration and state-level confidence intervals.

No single sampled token is a correctness verdict. No correctness claim is available without a normative target distribution. Temperature, truncation policy and vocabulary event map are part of the Oracle definition.

## Kill-criterion audit

- The claim “a sampling fork directly measures compiler drift” is rejected: ordinary sampling disagreement dominates the implementation TV scale.
- The claim “the sampling shift has a persistent direction” is not supported: signed top-1 and entropy summaries reverse or include zero.
- The narrower claim “there is a small, state-conditioned implementation-relative categorical-law discrepancy” survives discovery/confirmation.
- Whether that discrepancy is operationally unacceptable remains unidentified because no independent distributional tolerance has been supplied.
