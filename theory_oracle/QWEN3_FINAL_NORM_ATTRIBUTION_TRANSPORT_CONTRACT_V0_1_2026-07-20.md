# Qwen3 final-norm backward attribution transport contract v0.1

## Frozen question

Does the non-null heldout-B repair of the singleton final-RMSNorm backward
region remain non-null and retain its eager-relative direction at the
independently captured A and C step-29 states?

This is a mechanism-diversity extension of the SiLU transport campaign.  The
treatment is frozen before any A/C final-norm repair result is observed.

## Treatment and controls

The treatment is `final_norm_backward` from
`qwen3_backward_singleton_repair_v0_4.py`.  It replaces the single runtime call
of
`triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_slice_backward_sum_view_16`
with the declared eager FP32/FP16 tail-slice and RMSNorm-backward stages.

Existing valid eager/compiled baseline repeats and the A/B/C exact-null cast
control are reused.  The final-norm treatment runs twice in independent
processes at each state.  Repeat 1 retains complete clipped-gradient and update
vectors; repeat 2 is a hash-level runtime repeat.  The existing B repeat-1 is
reused, while B repeat-2 and both A/C repeats are new.

## Estimands and verdicts

For each endpoint and state, report repair nullness, repair magnitude,
repair/compiled-to-eager cosine, normalized target projection and fractional
eager-distance reduction.  Across states report descriptive variation and
pairwise repair-effect alignment.

- `TRANSPORTED_DIRECTION`: non-null in all states with the same target-projection sign;
- `STATE_CONDITIONAL_DIRECTION`: a state is null or projection signs differ;
- `TRANSPORTED_EXISTENCE_ONLY`: non-null in all states without a stable direction;
- `INDETERMINATE_RUNTIME`: within-state repeats differ;
- `INDETERMINATE_INTERVENTION_CONTROL`: a reused cast control is non-null;
- `INVALID`: any snapshot, baseline, candidate, treatment-count or artifact gate fails.

The result is intervention-dependent region attribution.  It is not a source-
operator decomposition, injection, root-cause, correctness, population or
long-run claim.
