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
