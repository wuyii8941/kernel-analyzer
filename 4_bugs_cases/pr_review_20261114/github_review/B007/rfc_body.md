Normalized linear-attention decode stores its running key sum Z in the activation dtype. Constant unit keys cause BF16 Z to stall at 256 steps and FP16 Z at 2048 while the FP32 KV state continues to grow, breaking constant preservation. Explicit FP32 initial Z avoids that failure.

Would the maintainers prefer FP32 Z by default for `normalize=True`, or an explicit state-precision policy? The attached implementation keeps cumsum/Z and denominator in FP32. Four new GPU regressions pass, including existing explicit-FP32-state controls; the PR draft contains the affected-path checks and resource-limited full-suite attempt.

Two interface details need agreement:

1. Returned Z becomes FP32, doubling that state's FP16/BF16 storage. Cache users must preserve it. Old lost increments cannot be recovered from a rounded cache.
2. The proposed helper returns attention output in its incoming `o.dtype`. The old helper could promote output to FP32 when given explicit FP32 Z. Should that promotion remain part of the interface, or should normalized output consistently follow activation dtype?

The signature and `(KV, Z)` tuple structure do not change. The proposal affects the normalized path, not the default unnormalized profile. Limited dense/packed helper timing and incremental allocation are included in the review package; no model performance or quality result is claimed.

This RFC follows the repository's requirement to agree precision changes before opening their implementation PR. Related #268 adds the state itself; this proposal concerns precision of repeated updates. The implementation patch is ready for review, with the dtype policy explicitly pending maintainer agreement.
