`L1Loss` currently returns a positive derivative when input equals target; PyTorch selects zero at this point. For example, a scalar bias added to a zero tensor has zero mean L1 loss but receives an approximately +1 bias gradient.

Return zero for exact equality and preserve the existing branches for other residuals, including their NaN behavior. Add six tests for exact and mixed residuals, mean/sum/unreduced losses, non-unit upstream gradients, and both input and target derivatives.

Validation on RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0:

- All six added GPU cases fail on the original source and pass with this change.
- The L1 selection of the existing p-loss tests reports 348 passes. Some existing FP16/no-AMP parameterizations return early; this is a pytest count, not 348 distinct GPU executions.
- Changed files pass basic Ruff correctness checks and `git diff --check`.
- A bounded `(256, 1, 28, 28)` mean-L1 forward/backward timing is included in the review evidence; it does not establish a performance improvement.

```bash
python -m pytest tests/test_p_loss_layers.py -k 'L1Loss and not SmoothL1Loss'
```

Related issue [#13](https://github.com/BobMcDear/attorch/issues/13) and PR [#19](https://github.com/BobMcDear/attorch/pull/19) concern SmoothL1/Huber thresholding; this change is confined to L1 equality. The complete p-loss matrix and model training were not rerun.

Pinned base: [BobMcDear/attorch@67c2772baf33913ef51c2d1064f746ca17b63749](https://github.com/BobMcDear/attorch/commit/67c2772baf33913ef51c2d1064f746ca17b63749) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.
