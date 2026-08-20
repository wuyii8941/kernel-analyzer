# Bias-risk oracle v1

## Decision

The validated object is a **fail-closed multi-witness risk oracle**, not a
single scalar and not a safety certificate. It reports `DIRECTIONAL_RISK` when
any independently defined sufficient witness fires; otherwise it reports
`ABSTAIN`.

The development regression recovered 6/6 strict formation positives
with zero false-safe decisions. A secondary retrospective lm-head positive was
detected while a real RMSNorm sign-changing control was not. Most importantly,
the frozen prospective moving-frame screen discovered one new case:
`deepseek8b_seq64_l35_attention_dv`. One of three promoted
candidates confirmed on 32 disjoint natural states; neither of the two
sign-changing controls fired.

## Witnesses

1. **Conditional event/source asymmetry.** At a fixed training condition,
   randomized or semantic-orbit repair samples have a non-centered complete
   local/effective response.
2. **Transported directional component.** Either the complete parameter-vector
   population has positive cross-state U-statistic energy, or the same-state
   coefficient `<delta_g, g_repair>/||g_repair||^2` has a nonzero confirmed
   mean. A calibration-frozen analytic/cross-fit projection is an optional
   instance of this channel.
3. **Response rectification.** Exact `+delta_g/-delta_g` perturbations are not
   mapped to opposite effective optimizer updates.

These are three operational witnesses for the two exact antithetic-symmetry
defects: event/pair-mass asymmetry and non-odd downstream response. They do not
use trajectory drift, T4, or SEUP as formation labels.

## New case

For DeepSeek layer-35 attention,

`O = P V`, `dV = P^T dO`, and `dW_v = dV^T H`.

Replacing only the compiled BF16 `dV` BMM output with the FP32-recomputed,
BF16-ABI reference changes the complete `v_proj.weight` gradient. On unseen
states, candidate minus repair has mean relative coefficient
-0.00369834, with 95% CI
[-0.00663691,
-0.000591755]. The absolute
gradient direction may rotate with the input; in the repair-gradient moving
frame, the candidate consistently contracts the update component.

This is a new **conditional bias-formation case**. It is not yet a complete
SEUP/trajectory accumulation case and does not imply that all attention BMMs
share the effect.

## Cost and generality

The numerical witnesses add no additional model F+B after a candidate/repair
trace exists: they reduce retained vectors to dot products, Grams, and an
offline optimizer response. For a grouped capture of `N` endpoints, the exact
current runner costs `N + 3` F+B arms per state (`N` repairs plus shared
candidate, reference, and sham), rather than `4N` isolated arms. Four states
are an inexpensive first screen; 16+16 is reserved for promoted cases.

The oracle is implementation-agnostic over any exact F+B boundary with a
matched repair and reachable parameter gradient. Missing repair, zero carrier
reach, discontinuous unsupported paths, or non-replicating evidence produces
`ABSTAIN`, never `SAFE`.

## Claim boundary

This evidence is sufficient to freeze the indicator as a research
risk-discovery oracle. It is not sufficient for certifying arbitrary unseen
kernels as safe or for claiming long-horizon training failure. Those require
coverage-specific repair availability and, for persistence claims, the
separate SEUP/consequence analysis.
