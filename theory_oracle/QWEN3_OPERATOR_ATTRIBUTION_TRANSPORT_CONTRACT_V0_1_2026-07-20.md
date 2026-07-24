# Qwen3 operator-attribution transport contract v0.1

## Question

Does a local operator repair effect observed at heldout-transport-B step 29
retain its existence, direction and endpoint relationship at two independently
replayed step-29 states, or was it conditional on the original state?

This is a transport test for intervention-dependent attribution.  It is not a
correctness or long-run-training test.

## Frozen states and treatments

States are A-replay, original B and C-replay.  They arise from three separately
seeded/data-offset GRPO trajectories under the same model, training and
implementation protocol.  They are deliberately selected independent states,
not a random sample from all training states.

Two treatments are frozen before A/C repair results exist:

1. `silu_middle`: replace zero-based call 13 of the 27-call family
   `triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_32` with the
   source-graph eager SiLU/value/backward operations.  At B this treatment was
   non-null and moved both endpoints slightly closer to eager.
2. `cast_up_control`: replace zero-based call 0 of the 84-call family
   `triton_poi_fused__to_copy_26`, mapped to the up-projection weight-gradient
   FP16-to-FP32 conversion.  At B this treatment was exact-null and serves as a
   negative intervention control.

For each newly captured state, eager, compiled, `silu_middle` and
`cast_up_control` execute twice in separate processes.  Repeat 1 retains full
clipped-gradient and parameter-update vectors; repeat 2 is a hash-level runtime
repeat.  Existing B baseline repeats are reused, and both B treatment repeats
are completed under the same executors.

## Validity gates

Every state must have a valid complete transition snapshot and a fresh
candidate-valid grad-state anchor.  Every arm must preserve the state and
scorer anchors, use the declared eager/compiled graph family, execute exactly
one backward, and satisfy the frozen family call/repair counts.  Within-arm
repeat equality is tested separately and is never silently pooled with state
heterogeneity.

The cast control is expected to remain exact-null because FP16-to-FP32
conversion is exact for represented values.  A non-null cast endpoint does not
prove cast bias; it triggers an `INDETERMINATE_INTERVENTION_CONTROL` verdict for
that state until local-output and intervention-mechanics controls explain it.

## Estimands

For each state and endpoint, let eager, compiled and repaired vectors be
`E_s`, `C_s` and `R_s`.

Report separately:

- implementation discrepancy magnitude: `||C_s - E_s||`;
- repair magnitude: `||R_s - C_s||`;
- repair/target cosine: alignment of `R_s-C_s` with `E_s-C_s`;
- normalized target projection:
  `dot(R_s-C_s, E_s-C_s) / ||E_s-C_s||^2`;
- fractional eager-distance reduction:
  `(||C_s-E_s|| - ||R_s-E_s||) / ||C_s-E_s||`;
- discrete clip/AMP/skip events and complete transition hashes.

Across A/B/C, report the average and state heterogeneity of the scalar target
projection and distance reduction, plus pairwise alignment of repair-effect
vectors in parameter coordinates.  With only three states these are descriptive
sample quantities, not population variance estimates or stable confidence
intervals.

Runtime variability is the within-state, same-arm repeat difference.  State
heterogeneity is the between-state variation in the intervention effect.  The
two must not be called one undifferentiated variance.

## Verdicts

- `TRANSPORTED_DIRECTION`: non-null SiLU repair has the same target-projection
  sign at all valid states and exact repeat controls.
- `STATE_CONDITIONAL_DIRECTION`: valid states have different signs or one state
  is exact-null while another is non-null.
- `TRANSPORTED_EXISTENCE_ONLY`: non-null in all states but direction/alignment
  is not stable.
- `INDETERMINATE_RUNTIME`: same-arm repeats differ beyond the declared exact
  deterministic protocol.
- `INDETERMINATE_INTERVENTION_CONTROL`: cast control is non-null.
- `INVALID`: snapshot, candidate identity or intervention-count gates fail.

No verdict grants necessity, sufficiency, root cause, population prevalence,
long-run harm or numerical correctness.
