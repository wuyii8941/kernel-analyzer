When the student input is FP32 and the target is FP16, `LigerKLDIVLoss(reduction="mean")` can return all-zero input gradients even though the gradients are representable in FP32.

### Reproducer

```python
import torch
from liger_kernel.transformers.kl_div import LigerKLDIVLoss

student = torch.zeros((32, 32768), device="cuda", requires_grad=True)
teacher = torch.full(student.shape, 1 / 32768, device="cuda", dtype=torch.float16)
reference = student.detach().clone().requires_grad_()

LigerKLDIVLoss(reduction="mean")(student, teacher).backward()
torch.nn.KLDivLoss(reduction="mean")(reference, teacher).backward()

print("Liger gradient:", student.grad[0, 0].item())
print("PyTorch gradient:", reference.grad[0, 0].item())
print("Liger nonzero gradients:", student.grad.count_nonzero().item())
```

Observed on `83746f4007d18a8ecc16786e9848cc1e5fc4eae0`:

```text
Liger gradient: -0.0
PyTorch gradient: -2.9103830456733704e-11
Liger nonzero gradients: 0
```

Expected: each student gradient is `-2**-35`, matching PyTorch.

The backward temporary is allocated with the target dtype. Mean reduction can round the result to zero before it is returned to the FP32 student. Keeping the temporary in at least FP32 until scaling/reduction are complete fixes this reproducer.

Environment: NVIDIA RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

Proposed fix: #1625.
