`attorch.L1Loss` selects a positive derivative when prediction equals target, whereas PyTorch selects zero. This can move a shared bias away from an exact solution even when the loss is zero.

### Reproducer

```python
import torch
import attorch

for loss_fn in (attorch.L1Loss(reduction="mean"), torch.nn.L1Loss(reduction="mean")):
    bias = torch.zeros((), device="cuda", requires_grad=True)
    prediction = bias.expand(8).contiguous()
    loss = loss_fn(prediction, torch.zeros_like(prediction))
    loss.backward()
    print(type(loss_fn).__module__, "loss=", loss.item(), "bias.grad=", bias.grad.item())
```

Observed on `67c2772baf33913ef51c2d1064f746ca17b63749`:

```text
attorch.p_loss_layers loss= 0.0 bias.grad= 1.0
torch.nn.modules.loss loss= 0.0 bias.grad= 0.0
```

Expected: the bias gradient is zero for both implementations. The same convention should hold for the input and target gradients with `none`, `mean` and `sum` reductions.

The L1 backward branch currently uses `diff >= 0` to choose +1. Handling `diff == 0` separately restores PyTorch's zero-subgradient convention.

Environment: NVIDIA RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

Proposed fix: #22. This concerns L1 equality, separately from the SmoothL1/Huber reports in #13 and #19.
