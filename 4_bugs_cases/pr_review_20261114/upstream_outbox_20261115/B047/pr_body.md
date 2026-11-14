`L1Loss` currently selects +1 at exact equality, whereas PyTorch selects zero. A scalar bias can therefore receive a +1 gradient when the mean L1 loss is already zero.

Return zero for exact equality and preserve the other branches, including their existing NaN behavior. Six regressions cover mean/sum/unreduced losses, all-zero and mixed residuals, non-unit cotangents, and input/target gradients.

Validation on `67c2772baf33913ef51c2d1064f746ca17b63749`: all six new regressions fail before the fix and pass afterward. The command below passes **354 parameterizations**, including all six additions. Some existing FP16/no-AMP cases return early; this count is not 354 distinct GPU executions.

```bash
python -m pytest tests/test_p_loss_layers.py -k 'l1_zero_residual_gradient or (L1Loss and not SmoothL1Loss)'
```

Changed files pass basic Ruff correctness and whitespace checks. Tested on NVIDIA RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

Related #13 and #19 address SmoothL1/Huber thresholding; this change only concerns the L1 zero-residual derivative.
