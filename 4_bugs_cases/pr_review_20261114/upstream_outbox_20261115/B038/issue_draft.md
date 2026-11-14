With dropout enabled, the variable-length attention backward can return a nonzero query gradient even when all keys are identical. For a fixed dropout mask, identical keys make the softmax probabilities independent of Q, so dQ should be zero up to numerical error.

### Reproducer

Save as `test_varlen_dropout_repro.py` and run `python -m pytest -q -s test_varlen_dropout_repro.py` on NVIDIA. It builds the same mask as this backward core and supplies the corresponding output/LSE directly, avoiding a separate forward implementation's RNG convention.

```python
import math

import torch
import triton
import triton.language as tl
from flag_gems.ops.flash_attention_backward import flash_attn_backward


@triton.jit
def make_mask(mask, seed, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(mask + i, tl.rand(seed, i) > 0.5, i < SIZE)


def test_varlen_dropout_query_gradient():
    m, n, d, seed = 128, 256, 32, 7331
    q = torch.zeros((m, 1, d), device="cuda", dtype=torch.float16)
    k = torch.full((n, 1, d), 0.125, device="cuda", dtype=q.dtype)
    v = torch.ones_like(k)
    dout = torch.full_like(q, 1 / q.numel())
    keep = torch.empty((m, n), device="cuda", dtype=torch.bool)
    make_mask[(triton.cdiv(keep.numel(), 256),)](
        keep, seed, keep.numel(), 256,
    )
    # All logits are equal. Apply exactly the backward core's fixed mask.
    out = (keep.float().sum(-1) * (2 / n))[:, None, None]
    out = out.expand_as(q).to(q.dtype).contiguous()
    lse = torch.full((m, 1), math.log(n), device="cuda", dtype=torch.float32)
    dq, _, _, _ = flash_attn_backward(
        dout, q, k, v, out, lse,
        cu_seq_q=torch.tensor([0, m], device="cuda", dtype=torch.int32),
        cu_seq_k=torch.tensor([0, n], device="cuda", dtype=torch.int32),
        max_seqlen_q=m, max_seqlen_k=n,
        softmax_scale=d**-0.5,
        is_dropout=True, dropout_p=0.5, rng_state=(seed, 0),
    )
    print("max |dQ|:", dq.abs().max().item())
    # Identical keys make the softmax probabilities independent of Q.
    torch.testing.assert_close(dq, torch.zeros_like(dq), rtol=0, atol=1e-6)
```

Observed on master `714859074155592820cac77d08a006f2c48eb4fc`: maximum absolute dQ is `0.0001729726791381836` and the assertion fails. With the proposed fix, it is `8.940696716308594e-07` and the assertion passes.

The outer factor in `dS = P * (dP_drop - D)` must be the undropped probability. Both varlen derivative loops currently overwrite that P after applying dropout. dV still needs the dropped probability, so the DKV loop needs a separate `P_drop`.

Environment: NVIDIA RTX A6000, PyTorch 2.10.0+cu128, Triton 3.6.0.

Proposed varlen-only fix: #9116. Related: #8969 fixes the dense path; #8748 concerns separate forward/backward RNG routing. This report does not claim to resolve that RNG integration issue.
