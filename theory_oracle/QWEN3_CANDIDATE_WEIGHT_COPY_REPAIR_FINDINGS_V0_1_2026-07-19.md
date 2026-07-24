# Qwen3 candidate weight-copy repair findings v0.1

All three generated copy families pass the generic fail-closed audit:

- q/o family `_to_copy_1`: 6/6 role-by-layer representatives are exact null;
- k/v family `_to_copy_3`: 6/6 representatives are exact null;
- gate/up/down family `_to_copy_13`: 9/9 representatives are exact null.

For every arm, the named live kernel executed the expected 56 or 84 calls,
exactly one selected call was replaced, repeats were bitwise exact, no backend
recompilation occurred, and restoring the kernel reproduced the candidate
anchor.  These are valid intervention nulls, not missing measurements.

The evidence supports that eager FP32-to-FP16 conversion and these generated
copy kernels agree at the selected state and role/layer representatives.  It
does not prove cross-state equivalence, cover other casts, or make the three
families interchangeable.  Population transport remains unvalidated.
