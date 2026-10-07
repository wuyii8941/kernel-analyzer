### 🐛 Describe the bug

**`torch.compile`: the `scatter_add` decomposition of `max_pool2d_with_indices_backward` crashes (device-side assert) on windows whose forward index is `-1`, which Inductor's own forward produces for windows that sample only padding.**

```python
import torch, torch.nn.functional as F

x = torch.tensor([[[7.]], [[8.]]], device="cuda", requires_grad=True)   # length-1 axis
f = lambda x: F.max_pool1d(x, kernel_size=2, stride=1, padding=1, dilation=2)
y = torch.compile(f)(x)            # tensor([[[-inf]], [[-inf]]])  correct: the only window samples positions -1 and +1 (padding)
y.backward(torch.ones_like(y))     # Assertion `index out of bounds: 0 <= ...` failed -> CUDA error: device-side assert triggered
```

Expected: `x.grad == 0` (no input is in any window), which is what eager and torch 2.10 Inductor return. The arguments are legal (`padding <= ((kernel_size - 1) * dilation + 1) / 2`). It also crashes with `return_indices=True`, and for inputs where a window's real values are all `-inf` next to padding.

**Cause.** The global decomposition `torch/_decomp/decompositions.py::max_pool2d_with_indices_backward` (used by Inductor unless deterministic algorithms are enabled) does

```python
grad_input_flat = grad_input_flat.scatter_add(1, indices_flat, grad_output_flat)
```

with the indices from the forward. Inductor's forward stores `-1` for a window that has no real maximum, and the indirect-indexing bounds check then fires. Torch 2.10 had no such decomposition (the lowering compares indices window by window, so `-1` never matches and the gradient is 0). Masking invalid indices before the scatter (e.g. `indices_flat.clamp(min=0)` together with `grad_output_flat.masked_fill(indices_flat < 0, 0)`) would restore the previous behaviour.

### Versions

nightly 2.15.0.dev20260907+cu126 (CUDA); the decomposition body on main is identical as of 2026-10-07; torch 2.10.0 is not affected.
