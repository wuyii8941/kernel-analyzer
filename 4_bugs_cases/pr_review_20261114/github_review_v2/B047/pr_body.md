`L1Loss` currently returns a positive derivative at exact equality, whereas PyTorch selects zero. A scalar bias added to a zero tensor can consequently receive an approximately +1 gradient despite zero mean L1 loss.

Return zero only for exact equality, preserving the other existing branches, including their NaN behavior. Six regressions cover mean/sum/unreduced losses, mixed residuals, non-unit cotangents and input/target gradients.

Validation: the [six new cases fail on the original source](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B047_before_v1/junit.xml) and [pass after the change](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B047_after_v1/junit.xml). The corrected command below [passes 354 parameterizations and includes all six new cases](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B047_corrected_public_command/junit.xml). Some existing FP16/no-AMP parameterizations return early, so this is a pytest count rather than 354 distinct kernel executions.

```bash
python -m pytest tests/test_p_loss_layers.py -k 'l1_zero_residual_gradient or (L1Loss and not SmoothL1Loss)'
```

Changed files pass basic Ruff correctness and whitespace checks. [Bounded same-device timing/allocation](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md) is provided without a speedup claim. Tested base: `BobMcDear/attorch@67c2772baf33913ef51c2d1064f746ca17b63749`, RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0. The complete p-loss matrix and model training are not claimed.

Related [issue #13](https://github.com/BobMcDear/attorch/issues/13) and [PR #19](https://github.com/BobMcDear/attorch/pull/19) concern SmoothL1/Huber thresholding; this patch changes only L1 equality.
