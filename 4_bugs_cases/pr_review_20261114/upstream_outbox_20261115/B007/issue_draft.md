With `normalize=True`, repeated one-token decoding stores the cumulative key sum `Z` in the activation dtype. For unit Q/K/V, BF16 Z stops growing at 256 and FP16 Z at 2048, while the FP32 KV state keeps growing. The normalized output then exceeds 1 even though all values are 1.

### Reproducer

```python
import torch
from fla.ops.linear_attn import fused_recurrent_linear_attn

@torch.no_grad()
def run(dtype, steps):
    qkv = torch.ones((1, 1, 1, 64), device="cuda", dtype=dtype)
    state = None
    for _ in range(steps):
        output, state = fused_recurrent_linear_attn(
            qkv, qkv, qkv,
            initial_state=state,
            output_final_state=True,
            normalize=True,
        )
    print(dtype, "steps=", steps,
          "KV=", state[0].flatten()[0].item(),
          "Z=", state[1].flatten()[0].item(),
          "output=", output.flatten()[0].item())

run(torch.bfloat16, 264)
run(torch.float16, 2056)
```

On main `2705ab8ee9a9086bcfa6d079858e4281e01215a1`:

```text
torch.bfloat16 steps=264   KV=264.0   Z=256.0   output=1.03125
torch.float16  steps=2056  KV=2056.0  Z=2048.0  output=1.00390625
```

Expected: Z equals the number of steps and the output remains 1. Initializing Z explicitly in FP32 avoids the lost increments.

Environment: NVIDIA RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

### Proposed change

Keep cumulative Z and the normalization denominator in FP32 across decode calls. This changes returned Z from FP16/BF16 to FP32, increasing its storage from 2 to 4 bytes per element.

Would FP32 Z by default be acceptable? There is also an output-dtype choice: explicit FP32 Z can currently promote the attention output to FP32; the candidate fix casts it back to the activation dtype. I will keep the implementation in draft while that API choice is discussed.
