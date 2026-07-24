# Qwen3 operator-attribution transport findings v0.1

## Result

The frozen A/B/C transport campaign is valid and its fail-closed verdict is
`STATE_CONDITIONAL_DIRECTION` for both clipped gradients and parameter
updates.  All manifest hashes, transition snapshots, candidate identities,
scorer anchors, family call counts and intervention counts passed.

Every eager, compiled, SiLU-repair and cast-control arm was repeated in an
independent process.  All within-state repeats were exact.  Runtime variability
was therefore not detected under the declared deterministic protocol.

The frozen FP16-to-FP32 cast repair was exact-null at every state and endpoint.
This supports the intervention-mechanics control; it is not evidence that all
casts are harmless or that casts in general have no implementation effect.

## State-conditioned SiLU repair

The same predeclared repair replaced call 13 of the 27-call generated family
`triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_32` at every state.

| state | clipped-gradient repair | update repair | endpoint interpretation |
|---|---:|---:|---|
| A replay | exact-null | exact-null | eager and compiled endpoints were already exact at these two vector endpoints |
| B original | non-null; +0.1941% distance reduction | non-null; +0.0187% distance reduction | slightly closer to eager |
| C replay | non-null; -0.3155% distance reduction | non-null; -0.1783% distance reduction | slightly farther from eager in total distance |

The normalized projection onto the compiled-to-eager target was 0.00651 at B
and -0.000222 at C for clipped gradients.  For updates it was 0.02328 at B and
0.01747 at C, although the C repair still increased total eager distance due
to an orthogonal component.

The B/C repair-effect vector cosine was -0.0225 for clipped gradients and
-0.00104 for parameter updates.  Thus the non-null effects at B and C are
nearly orthogonal in parameter coordinates.  Shared existence does not imply a
shared effect vector or a stable global direction.

No arm changed the declared clip/AMP/skip semantic events.  The observed
operator effect is a continuous gradient/update impact, not a discrete-event
shift in these three states.

## Oracle interpretation

This campaign empirically separates the three objects that must not be merged:

- persistent/global effect: not established; a single stable direction across
  A/B/C is rejected;
- state-conditioned heterogeneity: established descriptively, because A is
  null while B/C are non-null and the B/C effect vectors are nearly orthogonal;
- within-state runtime variability: not detected, because every arm's repeats
  are exact.

With only three deliberately selected states, the across-state sample
variances are descriptive and are not population variance estimates.  The
state distribution has not been sampled densely enough to estimate prevalence
or confidence intervals.

## Causal and correctness boundary

The result is intervention-dependent operator attribution.  It demonstrates
that replacing one mapped generated call can alter full-step endpoints at some
states, while the same intervention is null at another state.  It does not
identify arithmetic root cause, necessity, sufficiency or the contribution of
all calls in the family.  There is no injection arm and no independent
specification or high-precision authority.  Eager is only the comparison
baseline, so “closer to eager” is not a correctness judgment.

Machine-readable evidence:
`results/operator_oracle/qwen3_operator_attribution_transport_v0_1/evaluation.json`.

Frozen contract and manifest:
`QWEN3_OPERATOR_ATTRIBUTION_TRANSPORT_CONTRACT_V0_1_2026-07-20.md` and
`QWEN3_OPERATOR_ATTRIBUTION_TRANSPORT_MANIFEST_V0_1.json`.
