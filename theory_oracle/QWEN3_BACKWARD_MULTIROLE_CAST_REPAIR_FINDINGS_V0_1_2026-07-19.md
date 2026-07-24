# Qwen3 backward multi-role cast repair findings v0.1

## Result

The frozen three-role campaign is valid.  In every arm the family
`triton_poi_fused__to_copy_26` executed exactly 84 times, and exactly one of
the predeclared calls 0, 1 or 2 was replaced.  All candidate, scorer and
natural-transition gates passed.

Static call context resolves the first execution triple as FP16-to-FP32
weight-gradient conversions for the up projection, gate projection and down
projection.  Runtime shapes alone would not have separated up from gate because
both use `[3072, 1024]`; the down role uses `[1024, 3072]`.

All three selected role repairs were exact-null at both complete endpoints:
their clipped-gradient and AdamW-update tensor hash sets equal the unmodified
compiled baseline exactly.

## Interpretation

This campaign demonstrates two separate points:

1. one generated family can contain several semantic roles even when the
   kernel body is identical, so role mapping must precede attribution;
2. operator analysis must be able to report a strict zero effect rather than
   assuming every cast or floating-point operation contributes bias or noise.

The zero applies only to the selected first-executed triple.  It does not grant
equivalence across the remaining layer positions, and it is not a correctness
claim.  The family now has selected-state, selected-role repair evidence, not
full family coverage.

Machine-readable evaluation:
`results/operator_oracle/qwen3_backward_multirole_cast_repair_v0_1/evaluation.json`.
