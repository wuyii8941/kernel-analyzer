### 🐛 Describe the bug

**`F.gelu(approximate="tanh")` backward returns NaN for |x| >= ~1.85e19 in float32 / bfloat16 (eager CPU, eager CUDA and the Inductor decomposition).** The true derivative there is 1 (x > 0) or 0 (x < 0); the erf form and float64 are finite.

```python
import torch, torch.nn.functional as F

x = torch.tensor([1e10, 1.8e19, 1.9e19, 1e20, -1e20], requires_grad=True)
F.gelu(x, approximate="tanh").backward(torch.ones_like(x))
print(x.grad)                    # tensor([1., 1., nan, nan, nan])

x = torch.tensor([1e10, 1.8e19, 1.9e19, 1e20, -1e20], dtype=torch.float64, requires_grad=True)
F.gelu(x, approximate="tanh").backward(torch.ones_like(x))
print(x.grad)                    # tensor([1., 1., 1., 1., 0.], dtype=torch.float64)
```

Same result on CUDA, for bfloat16 inputs (opmath float32), and under `torch.compile`. The forward is correct.

**Cause.** `GeluBackwardCUDAKernelImpl` (`aten/src/ATen/native/cuda/ActivationGeluKernel.cu`), the CPU kernel (`aten/src/ATen/native/cpu/Activation.cpp`, scalar and vectorized paths) and the decomposition `torch/_decomp/decompositions.py::gelu_backward` compute

```
x_sq = x * x
tanh_derivative  = 1 - tanh_inner * tanh_inner        # exactly 0 for large |x|
inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)    # inf once x * x overflows (|x| >= 1.8447e19 in float32)
right_derivative = left * tanh_derivative * inner_derivative   # (0.5 * x * 0) * inf = NaN
```

Setting `right_derivative` to 0 where `tanh_derivative == 0` (or rewriting `inner_derivative` so it cannot overflow) would fix all three. The practical impact is small (activations of order 1e19), but the result is NaN where the derivative is finite, and it is the same in every backend, so comparing backends does not reveal it.

### Versions

torch 2.10.0+cu128 (CPU, CUDA, Inductor); nightly 2.15.0.dev20260907+cu126 and 2.15.0.dev20261005+cpu reproduce; the three formulas on main are unchanged as of 2026-10-07.

<!-- search record (for the submitter), 2026-10-07:
1. issues/PRs: "gelu tanh backward nan", "gelu approximate tanh nan gradient large", "gelu_backward tanh overflow", "GeluBackward tanh NaN":
   nothing about this; #189234 (open) fixes 1 + erf cancellation in the erf form, unrelated to the tanh-form backward.
2. tests: OpInfo nn.functional.gelu samples use make_tensor (|x| <= 9); no large-magnitude sample for the tanh form.
3. recent PRs touching ActivationGeluKernel.cu / Activation.cpp / gelu_backward decomposition: none changing the formula.
4. nightly: 2.15.0.dev20260907+cu126 (CUDA, Inductor) and 2.15.0.dev20261005+cpu reproduce. -->
