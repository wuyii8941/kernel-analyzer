# Qwen3 final-norm attribution transport findings v0.1

## Result

The A/B/C final-norm transport campaign is valid and its fail-closed verdict
is `STATE_CONDITIONAL_DIRECTION` for clipped gradients and parameter updates.
All snapshot, candidate, scorer, singleton-call, repair-count, runtime-repeat,
manifest and cast-control gates passed.

The same singleton final-RMSNorm backward repair was exact-null at A and
non-null at B and C.  No arm changed the declared clip, AMP or optimizer-skip
events.

| state | clipped-gradient distance change | update distance change |
|---|---:|---:|
| A replay | exact-null | exact-null |
| B original | +0.1499% | +0.0576% |
| C replay | -0.5168% | -0.2466% |

Positive means the repaired endpoint moved closer to eager; negative means it
moved farther away.  Eager is a baseline, not a correctness authority.

At B and C the normalized compiled-to-eager target projections were both
positive: 0.00944 and 0.000648 for clipped gradients, and 0.03883 and 0.03368
for updates.  Nevertheless the C total distances increased, because the
repair also introduced a larger component orthogonal to the target direction.
Therefore target projection and total-distance change must be reported
separately.

The B/C repair-effect vector cosine was 0.000220 for clipped gradients and
-0.000408 for parameter updates.  The effects are essentially orthogonal in
parameter coordinates despite similar magnitudes and a shared positive target
projection.

## Relation to the SiLU transport result

Both a repeated activation/backward family and a singleton norm/reduction
region now show the same high-level structure: A is null, B/C are non-null,
within-state repeats are exact, and the non-null B/C effect vectors are nearly
orthogonal.  State-conditioned attribution is therefore not confined to the
one SiLU example.

This does not prove that all operator effects are state-conditioned.  The two
treatments were selected because they were non-null at B, and the three states
are deliberately selected rather than sampled from a target population.

## Evaluator amendment

The frozen evaluator initially failed on A because eager=compiled=repair made
two normalized ratios equal to 0/0.  A recorded mechanical amendment assigns
zero to normalized target projection and fractional distance reduction only
when both the target and repair effect are exactly null; genuinely undefined
zero-target/non-null cases remain missing.  Treatment selection, raw vectors,
metrics for B/C and the predeclared null-state verdict were not changed.

## Claim boundary

This is intervention-dependent attribution for one fused generated region.
It is not an independent contribution for each constituent `add`, `sum`,
`pow`, `rsqrt`, cast or view operator.  There is no injection, necessity,
sufficiency, population, correctness or long-run claim.

Machine-readable evidence:
`results/operator_oracle/qwen3_final_norm_attribution_transport_v0_1/evaluation.json`.
