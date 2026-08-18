# Bias case protocol

The project reports two separate verdicts.  They answer different scientific
questions and neither substitutes for the other.

## Flash-style case track

`FLASH_STYLE_CASE` asks whether one concrete natural forward/backward mechanism
causally produces a directional error that accumulates along a paired training
trajectory, following the logic of the FlashAttention reference case.

Required evidence is:

1. **Complete F+B** — exact forward, saved tensors, upstream cotangent, actual
   backward arithmetic and output edges are independently bound.
2. **Localized mechanism** — the numerical difference is finite, repeat-stable
   and assigned to a concrete arithmetic or implementation cause.
3. **Causal repair** — a type-compatible repair changes the declared endpoint
   and reachable gradients; a matched sham or exact non-target controls pass.
4. **Real carrier** — the effect reaches a real parameter-gradient or weight
   carrier, not merely an internal tensor.
5. **Paired trajectory** — candidate and repair are evaluated at the same
   current weights before each update; the mechanism remains active and live
   weights diverge directionally.

This track does not require one fixed error vector across unrelated natural
inputs.  A failure of the cross-state track cannot revoke a complete
Flash-style case.

## Generalizable-bias track

`GENERALIZABLE_BIAS` asks whether the concrete mechanism remains directional
outside one trajectory.  It requires independent confirmation states, all
declared carrier coordinates, exact repeats, and a distinct-cluster bootstrap
95% lower bound above zero.  Raw, reference-relative and analytic-factor
hypotheses remain frozen before values are observed and retain multiplicity
control.

A pass here is only cross-state evidence for that concrete mechanism.  Claims
across shapes, models, backends or operator families require their own held-out
splits and cannot be inferred from this verdict.

## Fail-closed classifications

- Local difference without repair: `NEEDS_CAUSAL_REPAIR_AND_TRAJECTORY`.
- Causal repair without paired carrier trajectory: `NEEDS_TRAJECTORY`.
- Complete Flash-style chain: `PASS_FLASH_STYLE_CASE`.
- Cross-state interval crossing zero: `FAIL_CROSS_STATE_NONCOHERENT`.
- Cross-state direction without a complete causal/trajectory case:
  `LOCAL_CROSS_STATE_DIRECTION_ONLY`.
- Composite root: counted separately and excluded from root-operator property
  labels.

The machine-readable dual-track rules are
`results/coverage/case_classification_protocol.json`.  The original protocol-v3
artifact remains frozen in `results/coverage/directional_bias_protocol.json`
for provenance.  The separation is a transparent post-measurement methodology
correction, not a prospective preregistration and not a license to promote
missing evidence.
