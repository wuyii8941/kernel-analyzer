## Summary

Prevent mean KL loss from erasing FP32 student gradients when the target is FP16. For a uniform target with shape `(32, 32768)`, the current backward returns zero instead of `-2**-35` per input coordinate.

Keep the backward temporary in at least FP32, apply upstream scaling and reduction, then cast to the student dtype. Forward behavior and public signatures are unchanged. The existing FP32 kernel arithmetic is unchanged; this does not add FP64 kernel accuracy.

## Testing Done

- Complete `test/transformers/test_kl_div.py`: **248 passed** on base `83746f4007d18a8ecc16786e9848cc1e5fc4eae0`.
- The 144 new parameterizations cover student/target dtypes, reductions, log targets and upstream scales. On the original implementation, 22 fail and 122 pass; all pass with the fix. References use the same physical targets in FP64 and gradient `atol=0`.
- `make checkstyle`: passed with Ruff 0.14.11.

```bash
python -m pytest test/transformers/test_kl_div.py
make checkstyle
```

For the FP16-target `(32, 32768)` workload, measured incremental temporary allocation increases from 2,098,176 to 4,195,328 bytes. This is a correctness/memory trade-off.

- Hardware Type: NVIDIA RTX A6000; PyTorch 2.10.0+cu128; Triton 3.6.0
- [ ] run `make test` to ensure correctness
- [x] run `make checkstyle` to ensure code style
- [ ] run `make test-convergence` to ensure convergence
