# Normalized linear-attention decode loses BF16/FP16 key-sum increments

In normalized decode, the running key sum Z is stored in the activation dtype. For constant unit Q/K/V, BF16 Z stalls at 256 and FP16 Z at 2048 while the FP32 KV state continues growing, so a constant value is no longer preserved.

```python
import torch
from fla.ops.linear_attn import fused_recurrent_linear_attn

for dtype, steps in [(torch.bfloat16, 264), (torch.float16, 2056)]:
    q = torch.ones((1, 1, 1, 64), device="cuda", dtype=dtype)
    state = None
    with torch.no_grad():
        for _ in range(steps):
            out, state = fused_recurrent_linear_attn(
                q, q, q, initial_state=state, output_final_state=True, normalize=True,
            )
    print(dtype, out.flatten()[0].item(), state[1].flatten()[0].item(), state[1].dtype)
```

Expected: output 1 and Z equal to the number of steps. Original Z remains 256/2048. Passing explicit FP32 initial Z avoids the numerical failure; it is included as a control in the regression.

The attached implementation retains Z/denominator in FP32. Its four new GPU regressions pass, including both FP32-state controls. The precision/interface decision should be discussed in the companion RFC first. Returned Z storage and the explicit-FP32-state output-promotion behavior change; the PR draft discloses both. Related [#268](https://github.com/fla-org/flash-linear-attention/issues/268) introduced Z state itself.

Pinned base: [fla-org/flash-linear-attention@2705ab8ee9a9086bcfa6d079858e4281e01215a1](https://github.com/fla-org/flash-linear-attention/commit/2705ab8ee9a9086bcfa6d079858e4281e01215a1) (`main`). Environment: RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.
