## Summary

Keep the KL backward temporary in at least FP32 and cast to the student dtype after upstream scaling and reduction. Currently it inherits the teacher dtype: uniform FP16 targets for a `(32, 32768)` mean loss can produce an all-zero FP32 student gradient instead of `-2**-35` per element.

Save the student dtype in the autograd context and promote the temporary with the student/target types. This changes storage and post-kernel scaling, not the kernel's existing FP32 target load/exp arithmetic; it does not establish FP64 kernel accuracy. The public signature and forward implementation stay the same. Extra temporary storage for low-precision targets is an explicit correctness trade-off.

## Testing Done

- The 144 new GPU parameterizations cover student/teacher dtype combinations, all reductions, both target conventions, and two upstream scales. Original source: 22 failures and 122 passes. Modified source: 144 passes.
- Full `test/transformers/test_kl_div.py` at the recorded current base: 248 passes.
- Repository-wide Ruff lint and format checks use the repository-pinned Ruff 0.14.11; results are in the evidence directory.
- FP64 references use the same physical targets, with zero absolute gradient tolerance. A representable small gradient cannot pass by disappearing.
- Bounded forward/backward timing and incremental allocation are recorded for FP16 and FP32 teachers. On `(32, 32768)`, the FP16-teacher incremental peak rises from 2,098,176 to 4,195,328 bytes. No performance gain is claimed.

```bash
python -m pytest test/transformers/test_kl_div.py
```

Hardware: NVIDIA RTX A6000; PyTorch 2.10.0+cu128; Triton 3.6.0. The full repository `make test`, convergence matrix, other hardware backends, and NCU profiling were not run. The change does not claim pretrained-model quality improvement.

Pinned base: [linkedin/Liger-Kernel@8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81](https://github.com/linkedin/Liger-Kernel/commit/8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.
