# Qwen3 original-candidate layout/materialization kernel contracts v0.1

Three 28-call families are separate subjects:

1. `clone_6`: materialize non-contiguous query layout as contiguous FP32;
2. `...view_10`: reshape/transpose and repeat each of 8 value heads twice,
   materialize 16 heads, and convert FP16 to FP32;
3. `...view_11`: reshape attention BMM output, convert FP32 to FP16, transpose
   head and sequence axes, and materialize contiguous output.

For every family predeclare calls 0, 14 and 27.  Replacements reconstruct the
declared logical transformation from live source/destination shapes, check all
expected ranks and dimensions, and write the existing destination buffer.  The
families remain separate in results and coverage.

Require exact anchors and graph family, 28 calls with exactly one selected
repair, exact repeats, no backend recompilation and exact restoration.  Null and
either directional sign are admissible.  Claims are single-state,
generated-family, repair-only and implementation-relative; no primitive or
correctness attribution is implied.
