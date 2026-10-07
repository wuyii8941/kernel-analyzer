### 🐛 Describe the bug

**`scatter_reduce` / `index_reduce` backward with `reduce="amax"/"amin"` and `include_self=False` counts the excluded `self` value in the tie split, so the source gradient is too small.**

When `include_self=False`, the reduced value at a target position does not depend on `self` there. But the backward counts `self` as one of the elements "equal to the result" whenever its (excluded) value happens to equal the result, divides the gradient by that count, and only afterwards zeroes `grad_self`. The gradient that should go to the source is partly lost.

```python
import torch

s   = torch.tensor([3., 5.], requires_grad=True)   # s[0] is excluded (include_self=False) and happens to equal the result
src = torch.tensor([3., 1.], requires_grad=True)   # position 0 receives 3 (the unique max) and 1
out = s.scatter_reduce(0, torch.tensor([0, 0]), src, "amax", include_self=False)
out.backward(torch.ones(2))
print(out)        # tensor([3., 5.])   correct
print(src.grad)   # tensor([0.5, 0.])  expected tensor([1., 0.])
print(s.grad)     # tensor([0., 1.])   correct
```

Here `out[0] = max(src[0], src[1]) = src[0]` with a unique maximiser, so the function is differentiable at this point and `d out[0] / d src[0] = 1` (a central difference through the forward gives exactly 1). No tie-breaking convention is involved: among the values that take part in the reduction there is no tie. The same happens for `amin` and for `Tensor.index_reduce`, on CPU and CUDA, and in compiled graphs (the decomposition uses the same formula).

**Where.** `torch/csrc/autograd/FunctionsManual.cpp`, `scatter_reduce_backward` and `index_reduce_backward`, amax/amin branch:

```cpp
Tensor self_is_result = (self == result).to(self.scalar_type());   // ignores include_self
Tensor src_is_result = (src == value).to(self.scalar_type());
Tensor N_to_distribute = self_is_result.scatter_add(dim, index, src_is_result);
Tensor grad_distributed = grad / N_to_distribute;
...
if (!include_self) {
  grad_self = grad_self.scatter(dim, index, 0);   // self is zeroed only afterwards; src was already divided by N incl. self
}
```

The `mean` branch of the same function already handles this (`N = include_self ? ones_like(grad) : zeros_like(grad)`). A fix would set `self_is_result` to 0 at the scattered positions when `include_self=False` before computing `N_to_distribute`.

**Why tests do not see it.** The OpInfo sample inputs for `scatter_reduce` / `index_reduce` use continuous random tensors, so an excluded `self` essentially never equals the result, and gradcheck never reaches this case. It shows up with integer-valued or repeated data (counts, one-hot style features, `-inf`/`0` initialised buffers combined with clamped values, etc.).

### Versions

torch 2.10.0+cu128 (CPU and CUDA); nightly 2.15.0.dev20261005+cpu and 2.15.0.dev20260907+cu126 reproduce; `FunctionsManual.cpp` on main unchanged as of 2026-10-07.
