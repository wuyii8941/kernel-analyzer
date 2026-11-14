# FP16 targets can erase representable FP32 KL input gradients

The KL backward temporary inherits the target dtype. With the default mean reduction, small gradients can be lost in FP16 even when the student and its returned gradient are FP32.

```python
import torch
from liger_kernel.transformers.kl_div import LigerKLDIVLoss

target = torch.full((32, 32768), 1 / 32768, device="cuda", dtype=torch.float16)
for criterion in (torch.nn.KLDivLoss(), LigerKLDIVLoss()):
    student = torch.zeros(target.shape, device="cuda", dtype=torch.float32).log_softmax(-1).requires_grad_()
    criterion(student, target).backward()
    print(torch.count_nonzero(student.grad).item(), student.grad.min().item())
```

Expected: 1,048,576 nonzero entries, each `-2**-35`. Original Liger can return all zeros. All target entries exceed the log clamp; the derivative is exactly `-target / (B*V)`.

The proposed patch computes the backward temporary in at least FP32 and casts to the student dtype after scaling/reduction. New GPU matrix: original 22 failures/122 passes; patched 144 passes. The complete current KL file passes 248 tests. Low-precision targets require a larger temporary; details are in the PR draft and performance evidence. Related [#1535](https://github.com/linkedin/Liger-Kernel/issues/1535) concerns fused-linear KL forward logit precision and is not claimed resolved here.

Pinned base: [linkedin/Liger-Kernel@8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81](https://github.com/linkedin/Liger-Kernel/commit/8fbf19f47e391b6e7f5d60eb09fb2b0a49ddcf81) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

Evidence: [original mixed-dtype matrix](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B049_before_v1/junit.xml), [modified complete KL file](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/B049_latest_full/junit.xml), [temporary-memory cost](https://github.com/wuyii8941/kernel-analyzer/blob/3a4cbf1a0128dc615dbd6134768955d4aec58716/4_bugs_cases/pr_review_20261114/review_evidence_20261115/PERFORMANCE_v1.md).
