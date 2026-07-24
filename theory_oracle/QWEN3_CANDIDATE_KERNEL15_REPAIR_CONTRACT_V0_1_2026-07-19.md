# Qwen3 original-candidate kernel-15 repair contract v0.1

## Subject

The original whole-compiled Qwen3 forward contains 27 calls to
`triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15`.  Each call
fuses the previous decoder layer's attention/MLP residual accumulation with the
next layer's input RMSNorm and FP16 conversion.

Test calls 0, 13 and 26, corresponding to the early, middle and late repeated
fusion positions.  The treatment replaces only the selected generated
kernel invocation's `.run` with an eager PyTorch implementation that writes the
same preallocated output buffers.  Every other generated call, graph boundary,
fusion, layout and compiler specialization remains unchanged.

## Gates

- The unmodified compiled model reproduces the frozen candidate anchor twice.
- The observed Dynamo graph family is exact.
- The generated module and named kernel are resolved from the actual Inductor
  artifact, not the trace copy.
- Each repaired run observes 27 family calls and replaces exactly the selected
  one.
- Repaired repeats are exact.
- No backend compilation occurs during repair arms.
- Restoring the kernel object reproduces the candidate anchor.

## Claim

A passing contrast is an original-candidate-preserving **generated-kernel
invocation repair effect** at the selected state.  It is stronger than a
barrier-conditioned module effect but does not identify `add`, `mean`, `rsqrt`,
cast or RMSNorm individually.  It supplies no injection, population or
correctness claim.
