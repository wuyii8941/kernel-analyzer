"""Same-mask dropout-core timing; setup and mask construction are untimed."""
import importlib.util
import math
import sys
from pathlib import Path

import torch

import microbench


def tasks(_):
    source = Path(sys.modules['flag_gems'].__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        'dropout_regression', source/'tests/test_flash_attention_dropout_derivative.py',
    )
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    from flag_gems.ops.flash_attention_backward import flash_attn_backward
    for varlen in (False, True):
        m, n, h, d, seed = 128, 256, 2, 32, 7331
        q = torch.zeros(1, m, h, d, device='cuda', dtype=torch.float16)
        k = torch.full((1, n, h, d), .125, device='cuda', dtype=q.dtype)
        v, grad = torch.ones_like(k), torch.ones_like(q)/q.numel()
        keep = torch.empty(1, h, m, n, device='cuda', dtype=torch.bool)
        test._backward_mask[(math.ceil(keep.numel()/256),)](keep, seed, m, n, h, varlen, 256)
        out = (keep.float().sum(-1)*2/n).permute(0, 2, 1).unsqueeze(-1).expand_as(q).to(q.dtype).contiguous()
        lse = torch.full((1, h, m), math.log(n), device='cuda')
        args = (grad, q, k, v, out, lse)
        kwargs = dict(max_seqlen_q=m, max_seqlen_k=n, softmax_scale=d**-.5,
                      is_dropout=True, dropout_p=.5, rng_state=(seed, 0))
        if varlen:
            args = tuple(t.squeeze(0) for t in args[:5])+(lse.squeeze(0).T.contiguous(),)
            kwargs.update(cu_seq_q=torch.tensor([0, m], device='cuda', dtype=torch.int32),
                          cu_seq_k=torch.tensor([0, n], device='cuda', dtype=torch.int32))
        def step(args=args, kwargs=kwargs):
            return flash_attn_backward(*args, **kwargs)
        yield dict(name='dropout_backward_'+('varlen' if varlen else 'dense'),
                   q_shape=list(q.shape), k_shape=list(k.shape), mode='backward', dropout=.5), step


def measure(identifier, output):
    import flag_gems  # establish the source location before loading the regression
    microbench.tasks = tasks
    microbench.measure(identifier, output)
