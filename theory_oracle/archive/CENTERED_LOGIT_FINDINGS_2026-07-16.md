# Centered-logit Diagnostic Findings — 2026-07-16

## Question

The earlier Qwen study confirmed a negative mean `compiled - eager` logit shift at the final checkpoint. This diagnostic asks whether that result is mainly a decision-invariant translation of the whole vocabulary vector or a relative-logit distortion.

The estimand and state banks were frozen in `ORACLE_NEXT_STAGE_CONTRACT_2026-07-16.md` before these outputs were read.

## Result

| Endpoint | Discovery, 32 sequences / 4092 token positions | Confirmation, 32 sequences / 4084 token positions |
|---|---:|---:|
| Mean signed raw-logit delta | -8.55e-4 | -9.43e-4 |
| Mean absolute common-mode shift | 3.532e-3 | 3.528e-3 |
| Raw mean absolute logit delta | 6.301e-3 | 6.336e-3 |
| Centered mean absolute logit delta | 5.261e-3 | 5.320e-3 |
| Common-mode energy share | 22.73% | 22.45% |
| Centered residual energy share | 77.27% | 77.55% |
| Maximum energy-decomposition closure error | 1.51e-15 | 2.23e-15 |
| Same-state repeat variability | 0 | 0 |

The signed raw shift is real for the named checkpoint/population, but it is not “just a common offset.” Roughly three quarters of delta energy remains after subtracting the per-token vocabulary mean. Conversely, roughly one quarter is invariant to softmax, argmax and top-k and must not be counted as decision-relevant distortion.

## Does centering produce a better semantic detector?

No stable improvement was observed. Descriptive token-level rank AUCs were computed only as a diagnostic; token positions remain nested within sequence states, so these are not independent-observation confidence claims.

| Split / event | Raw mean-abs delta | Centered mean-abs delta | Raw max-abs delta | Centered max-abs delta | Correct boundary distance/risk |
|---|---:|---:|---:|---:|---:|
| Discovery argmax | 0.677 | 0.605 | 0.688 | 0.638 | 0.996 |
| Confirmation argmax | 0.837 | 0.470 | 0.636 | 0.572 | 0.997–0.999 |
| Discovery top-5 set | 0.537 | 0.513 | 0.557 | 0.545 | 0.982–0.986 |
| Confirmation top-5 set | 0.504 | 0.567 | 0.538 | 0.555 | 0.985–0.987 |

Within the conservative boundary-risk set, neither raw nor centered discrepancy magnitude ranked event changes consistently well; most AUCs were near 0.5–0.6, with unstable exceptions caused by sparse argmax events.

The common-mode shift also had essentially zero rank association with target-token log-probability delta (Spearman approximately -0.012 and -0.030 in the two splits).

## Oracle consequence

Centering is retained because it removes a semantic invariance confound and gives a correct numerical decomposition. It is **not** promoted as a stronger detector.

For ranking events, the evidence favors:

1. event-specific boundary distance;
2. an implementation-specific perturbation bound or direction relative to that boundary;
3. the paired event outcome;
4. state-level uncertainty and same-state repeat variability.

The global raw shift and centered residual remain explanatory numerical endpoints. Neither replaces the event-specific boundary Oracle.

## Kill-criterion audit

The proposed centered metric fails the strong novelty criterion “adds stable event-ranking power beyond raw numerical delta.” It survives only the narrower conceptual criterion “separates decision-invariant translation from relative-coordinate distortion.” Any paper or tool claim must reflect that downgrade.

