# Qwen3 candidate kernel-15 direction findings v0.5

## Audited result

`results/operator_oracle/qwen3_candidate_kernel15_direction_v0_5/audit.json`
has status `VALID_ORIGINAL_CANDIDATE_KERNEL_DIRECTION_AUDIT`.  Eager and
candidate anchors, graph family, exact repeats, one-of-27 call replacement, no
repair-time compilation and final restoration all passed.

## Finding

All three fused residual-plus-input-RMSNorm repairs have deterministic nonzero
effects, but they do not have a common relation to the eager output:

| Call | Repair/eager-direction cosine | Change in distance to eager | Interpretation |
|---:|---:|---:|---|
| 0 | 0.490 | -0.00517 (-5.64%) | partly closes the discrepancy |
| 13 | 0.228 | +0.00918 (+10.02%) | moves farther despite positive projection |
| 26 | 0.063 | +0.00135 (+1.47%) | nearly orthogonal and slightly farther |

The signs are descriptive for this frozen state; eager is a baseline, not an
independent truth source.

## Consequence for the operator Oracle

`repair effect != 0` supports causal influence of the selected generated
invocation on the scorer observable.  It does **not** establish that the
invocation explains the implementation-relative discrepancy.  Even positive
directional alignment is insufficient: a repair vector can contain enough
orthogonal magnitude to increase total distance.

Therefore operator attribution needs at least two separate endpoints:

1. intervention impact: whether the observable changes;
2. discrepancy explanation: whether and how the intervention changes the
   eager--candidate contrast.

Neither endpoint alone provides constituent-operator attribution, population
generalization, sufficiency, or correctness.
