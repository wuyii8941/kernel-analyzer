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

The same `-1` appears for a window whose real values are all `-inf` next to padding (e.g. `F.max_pool1d(x, 2, 2, 1,
ceil_mode=True, return_indices=True)` on `x = [-inf, -inf, 2., 3.]`: eager returns index `0` for the first window, Inductor `-1`); on
torch 2.10 that window's gradient is silently not routed, on nightly the backward crashes as above.

**Cause.** The global decomposition `torch/_decomp/decompositions.py::max_pool2d_with_indices_backward` (used by Inductor unless deterministic algorithms are enabled) does

```python
grad_input_flat = grad_input_flat.scatter_add(1, indices_flat, grad_output_flat)
```

with the indices from the forward. Inductor's forward stores `-1` for a window that has no real maximum, and the indirect-indexing bounds check then fires. Torch 2.10 had no such decomposition (the lowering compares indices window by window, so `-1` never matches and the gradient is 0). Masking invalid indices before the scatter (e.g. `indices_flat.clamp(min=0)` together with `grad_output_flat.masked_fill(indices_flat < 0, 0)`) would restore the previous behaviour.

### Versions

nightly 2.15.0.dev20260907+cu126 (CUDA); the decomposition body on main is identical as of 2026-10-07; torch 2.10.0 is not affected.

<!-- search record (for the submitter), 2026-10-07:
1. issues/PRs: "max_pool inductor device-side assert", "max_pool2d_with_indices_backward index out of bounds inductor", "max_pool -inf
   padding indices -1", "max_pool2d_with_indices_backward scatter_add decomposition": nothing about -1 indices; the decomposition came
   with the #167318 line (fixes #66042); #195124 (open) adds a non-overlapping fast path and does not handle -1.
2. tests: OpInfo max_pool samples have no padding-only window (see the B021 draft); test_torchinductor_opinfo runs
   max_pool2d_with_indices_backward on one sample only.
3. recent PRs: #195124 (open), #195123 (issue, redundant computation) - neither addresses invalid indices.
4. nightly: 2.15.0.dev20260907+cu126 (newest CUDA nightly runnable on driver 535) crashes; main's decomposition body is identical;
   torch 2.10.0 is not affected. -->
