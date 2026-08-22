# Three-factor experiment progress

This document records the execution state after the metric and replay audit. It is intentionally conservative: a missing raw vector or an unrepresentable antithetic arm is reported as unresolved, never reconstructed from a scalar trajectory summary.

## Completed

- `metric_binding.json` separates the Liger RN→SR screen (`A=0.9418938921`) from the historical, unbound `A≈2.315` number and from the Phi fixed-update propagation probe (`1.0104399627`).
- `replay_feasibility.json` audits which cases have the artifacts required for a generic even/odd response decomposition. At present, none is generically ready.
- Phi was recaptured on 16 states. The response-odd population is directionally biased (`cross_state_ratio=0.7747391992`), while response-even is centered (`-0.0056745431`). The reflected BF16 endpoint is not exactly representable (relative error `0.03856–0.07613`), so the run is `UNRESOLVED_REPRESENTABILITY`, not a causal intervention.
- Saved-P response vectors were streamed over 32 repair-trajectory steps. Both even and odd response populations are centered across that trajectory; this is a trajectory-conditioned response certificate, not a common-state formation verdict.
- Saved-P was then recaptured on the frozen 16+16 open-loop common-state split. Local, gradient, and effective-update layers are centered in both partitions; the offline zero-moment AdamW reflected-gradient even/odd response is also centered in both partitions. This is a bounded response measurement, not an exact source antithetic intervention.
- The existing Phi row-permutation intervention remains a case-level transport result: natural gradient cross-state ratio `0.67535` versus shuffled `0.10834`. It supports the importance of residual/transport pairing for this Phi boundary, but does not close a universal VJP decomposition.
- The first pre-registered screen-negative group (DeepSeek-8B, seq64) has now been captured with 32 open-loop states per exact case. Two cases are CENTERED in all three formation layers; one is `UNRESOLVED_INCONCLUSIVE` because its 16-state confirmation interval crosses the frozen margin. It is not a positive and is not called a safe negative.
- The v2.1 status vocabulary now distinguishes genuinely short populations (`UNRESOLVED_INSUFFICIENT_STATES`) from a complete population that is statistically inconclusive (`UNRESOLVED_INCONCLUSIVE`). The existing artifact was migrated without changing any Gram statistic.
- DeepSeek seq128 and Mamba seq64 were attempted on the host GPU but remained in AOT/Inductor warm-up and were interrupted without writing a formation artifact. They are `BLOCKED_AOT_WARMUP_INTERRUPTED`, not negative controls and not missing-data imputations.

## What this changes

The earlier zero-response Phi dry-run was an implementation error: it replayed BF16 MM instead of the frozen FP32 external reference. The corrected run reaches the target endpoint and exposes the representability boundary. No new strict persistent-bias case is promoted.

## Remaining blockers

1. Most cases do not retain raw endpoint residuals and exact `+epsilon/-epsilon` response arms; generic attribution therefore remains unresolved.
2. The Liger RN→SR result cannot be interpreted until the RN metric is reconciled with the historical `A≈2.315` artifact.
3. The 16-step prefix→32-step predictor backtest, the remaining screen-negative groups, live candidate/repair consequence reruns, and held-out confirmation remain open. The DeepSeek seq64 capture is formation-only and must not be reported as a persistence result.

The 12 screen-negative rows are now mechanically bound to eight exact model/shape plan files under `results/property/joint_bias_formation_v1/negative_consequence_plans/`; this binding is not a consequence result. The generic open-loop capture runner cannot be relabeled as a live optimizer trajectory runner.

The current scientific boundary is a partial, case-specific attribution map. It is not yet a cheap universal oracle.
