# L1Loss returns a positive gradient at exact equality

`attorch.L1Loss` disagrees with PyTorch's selected zero subgradient when prediction equals target.

```python
import torch
import attorch

for criterion in (torch.nn.L1Loss(), attorch.L1Loss()):
    target = torch.zeros((256, 1, 28, 28), device="cuda")
    bias = torch.zeros((), device="cuda", requires_grad=True)
    loss = criterion(target + bias, target)
    loss.backward()
    print(loss.item(), bias.grad.item())
```

Expected: zero loss and zero gradient for both. Original Attorch: zero loss but bias gradient approximately +1. The L1 branch uses `tl.where(0 <= diff, 1, -1)`; exact equality therefore receives +1 before normalization.

The attached patch returns zero for equality and retains all other existing branches. Six added GPU regressions fail before and pass after; the existing L1 selection reports 348 passes (including some existing early-return parameterizations). Related [#13](https://github.com/BobMcDear/attorch/issues/13)/PR [#19](https://github.com/BobMcDear/attorch/pull/19) concern SmoothL1/Huber, not this equality branch. No full-model quality claim is needed for this API mismatch.

Pinned base: [BobMcDear/attorch@67c2772baf33913ef51c2d1064f746ca17b63749](https://github.com/BobMcDear/attorch/commit/67c2772baf33913ef51c2d1064f746ca17b63749) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

Corrected validation command: `python -m pytest tests/test_p_loss_layers.py -k 'l1_zero_residual_gradient or (L1Loss and not SmoothL1Loss)'`. [Combined 354-case GPU record](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B047_corrected_public_command/junit.xml) includes all six added regressions.
