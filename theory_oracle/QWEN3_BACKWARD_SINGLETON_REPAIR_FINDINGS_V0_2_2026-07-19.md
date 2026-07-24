# Qwen3 backward singleton-repair findings v0.2

## Verdict

The selected-state singleton denominator is now resolved as follows:

- 8 of 9 families have valid repair interventions;
- 3 are exact-null at both clipped-gradient and update endpoints;
- 5 are non-null;
- 1 is operator-level uninstantiated because its direct output is a
  compiler-private partial-reduction ABI;
- 0 have injection, population-transport, long-run or correctness credit.

All eight valid repairs preserve the frozen scorer, loss construction,
compiled graph identity, and complete training-step validity.

## Direction relative to eager

Percentages below are changes in L2 distance to eager; positive means closer
and negative means farther.  They are implementation-relative directions, not
correctness judgments.

| Treatment | Clipped gradient | Parameter update | Repair classification |
|---|---:|---:|---|
| tangent cast/view | 0 | 0 | exact null |
| embedding-gradient zero initialization | 0 | 0 | exact null |
| FP16-to-FP32 add | 0 | 0 | exact null |
| attention safe-softmax | about -0.0001% | about -0.0007% | non-null, extremely small |
| embedding norm/backward preparation | rounds to 0% | rounds to 0% | non-null at 1e-9-scale endpoint coordinates |
| final norm backward | about +0.150% | about +0.058% | non-null, closer at both endpoints |
| SiLU × multiply | about +0.099% | about -0.119% | non-null, endpoint direction reverses |
| SiLU × multiply backward | about -0.459% | about -0.345% | non-null, farther at both endpoints |

## Main theoretical finding

Operator attribution is endpoint-dependent.  The SiLU-multiply repair moves
the clipped gradient slightly toward eager while moving the AdamW update away.
Global gradient clipping and optimizer geometry therefore prevent a single
operator-distance score from serving as the complete Oracle.

Magnitude and structural change also differ.  The embedding-preparation repair
changes the final digest despite almost unchanged global norms.  Conversely,
the three exact-null repairs have identical per-tensor hashes, not merely small
norm differences.

## The unresolved singleton

Kernel 30 emits six partial reductions for each of three logical eager sums.
A selected call of repeated family 34 later combines them.  Eager specifies the
final sums but not this six-way internal representation, so the direct operator
interface has no unique eager target.  A joint kernel-30/sum-34 replacement is
a region intervention and is tracked separately rather than counted as an
operator repair.

## Nonclaims

Separate non-null repairs on adjacent families are not additive causal effects.
Exact-null selected-state results do not establish cross-state equivalence.
No result identifies necessity, sufficiency, root cause, mathematical truth, or
long-run training harm.
