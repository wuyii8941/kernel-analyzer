# RAdam: capturable / torch.compile path computes rho_t in float32; the rectification decision flips for beta2 close to 1

### Summary

`torch.optim.RAdam` computes

```
rho_inf = 2 / (1 - beta2) - 1
rho_t   = rho_inf - 2 * step * beta2**step / (1 - beta2**step)
```

and takes the rectified (adaptive) step only when `rho_t > 5`. `rho_t` is the difference of two numbers close to
`rho_inf` (~2000 for beta2 = 0.999, while `rho_t` is ~5 in the first steps), i.e. a catastrophic cancellation.

* Non-capturable eager: `step = _get_value(step_t)` is a Python float, so this is evaluated in float64. Fine.
* `capturable=True`: `step = step_t` is a float32 tensor, so `beta2**step`, `1 - beta2**step` and `rho_t` are float32
  tensor ops; beta2 itself is rounded to float32 first (0.9999 -> 0.99989998, a 1.7e-4 relative error in 1 - beta2).
* `torch.compile(opt.step)`: `_get_value` returns the tensor while compiling, so the compiled step takes the same
  float32 path.

The relative error of `rho_t` is ~4e-3 for beta2 = 0.999, 3e-2 for 0.9995, 0.4 for 0.9999 and ~46x for 0.99999, so the
`rho_t > 5` decision flips in the very first steps — exactly the steps RAdam's rectification exists for.

### Reproduction

```python
import torch

def updates(beta2, mode, dtype, steps=10):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(48, 64, device="cuda").to(dtype))
    opt = torch.optim.RAdam([p], lr=1e-2, betas=(0.9, beta2), capturable=(mode == "capturable"))
    step = torch.compile(opt.step) if mode == "compiled" else opt.step
    g = torch.Generator(device="cuda").manual_seed(1)
    out = []
    for _ in range(steps):
        p.grad = torch.randn(p.shape, device="cuda", generator=g).to(dtype)
        before = p.detach().clone(); step(); out.append((p.detach() - before).double())
    return out

for beta2 in (0.999, 0.9995, 0.9999):
    ref = updates(beta2, "eager", torch.float64)
    for mode in ("eager", "capturable", "compiled"):
        d = updates(beta2, mode, torch.float32)
        print(beta2, mode, [f"{((a - b).norm() / b.norm()).item():.1e}" for a, b in zip(d, ref)])
```

Relative error of the parameter update against eager float64 (torch 2.10.0, RTX A6000):

| beta2 | steps where rho_t > 5 flips in float32 | capturable | compiled | eager fp32 (non-capturable) |
|---|---|---|---|---|
| 0.999 (default) | none | up to 6.0e-3 (step 7) | up to 5.7e-3 (step 6) | 2.3e-4 |
| 0.9995 | 5 | 0.99 at step 5 | 0.99 at step 5 | 3.3e-4 |
| 0.9999 | 2, 4, 5 | 0.99; 0.58 at step 6, 0.24 at step 10 | same | 7.6e-4 |
| 0.99999 | 1-5 | up to 13x | up to 18x | 2.3e-3 |

The same code is on main (`torch/optim/radam.py`, `_single_tensor_radam` and `_multi_tensor_radam`).

### Suggested fix

`rho_inf`, `rho_t` and the rectification factor depend only on beta2 and the step count. Compute them in float64 (a
float64 0-dim tensor in the capturable path), or at least evaluate `1 - beta2**t` as `-expm1(t * log1p(-(1 - beta2)))`
with `1 - beta2` taken from the Python float, and cast the resulting scalar factor to the parameter dtype.
