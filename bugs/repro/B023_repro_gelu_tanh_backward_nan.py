"""B023: gelu(approximate='tanh') backward returns NaN for |x| >= ~1.85e19 (float32 / bfloat16 opmath): x*x overflows to
inf inside inner_derivative while tanh_derivative is exactly 0, and 0 * inf = NaN.  The true derivative is 1 (x > 0) or 0
(x < 0); the erf form and float64 are finite.  Eager CPU (scalar and vectorised), eager CUDA and the Inductor
decomposition share the formula."""
import torch
import torch.nn.functional as F

print(torch.__version__)
xs = [1e10, 1.8e19, 1.9e19, 1e20, 1e30, -1e20]
for dev in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
    for dt in (torch.float32, torch.bfloat16, torch.float64):
        x = torch.tensor(xs, dtype=dt, device=dev, requires_grad=True)
        F.gelu(x, approximate="tanh").backward(torch.ones_like(x))
        print(f"{dev:4s} {str(dt)[6:]:9s} tanh-form grad {x.grad.float().tolist()}")
if torch.cuda.is_available():
    x = torch.tensor(xs, device="cuda", requires_grad=True)
    torch.compile(lambda t: F.gelu(t, approximate="tanh"))(x).sum().backward()
    print("inductor float32 tanh-form grad", x.grad.tolist())
x = torch.tensor(xs, requires_grad=True)
F.gelu(x).backward(torch.ones_like(x))
print("cpu float32 erf-form grad", x.grad.tolist())
