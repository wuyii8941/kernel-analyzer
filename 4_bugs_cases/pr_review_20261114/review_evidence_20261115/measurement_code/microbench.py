"""Bounded same-device operator timing and incremental allocation, no models."""
import importlib
import json
import statistics
from pathlib import Path

import torch
import triton


def tasks(identifier):
    if identifier == 'B049':
        from liger_kernel.transformers.kl_div import LigerKLDIVLoss
        for dtype in (torch.float16, torch.float32):
            x = torch.zeros(32, 32768, device='cuda', requires_grad=True)
            target = torch.full_like(x, 1/32768, dtype=dtype)
            loss = LigerKLDIVLoss(reduction='mean')
            def step(x=x, target=target, loss=loss):
                x.grad = None
                result = loss(x, target)
                result.backward()
                return result
            yield dict(name='mean_kl_'+str(dtype), shape=list(x.shape), mode='forward_backward'), step
    elif identifier == 'B047':
        import attorch
        x = torch.ones(256, 1, 28, 28, device='cuda', requires_grad=True)
        y = torch.zeros_like(x)
        loss = attorch.L1Loss()
        def step():
            x.grad = None
            value = loss(x, y)
            value.backward()
            return value
        yield dict(name='mean_l1', shape=list(x.shape), mode='forward_backward'), step
    elif identifier == 'B054':
        import trident
        for strided in (False, True):
            x = torch.zeros(2, 32768, device='cuda', requires_grad=True)
            full = torch.zeros(2, 65536 if strided else 32768, dtype=torch.bool, device='cuda')
            mask = full[:, ::2] if strided else full
            mask[:, ::2] = True
            def step(x=x, mask=mask):
                x.grad = None
                out = trident.function.masked_softmax(x, mask, 1)
                out.backward(torch.ones_like(out))
                return out
            yield dict(name='masked_softmax_'+('strided_mask' if strided else 'contiguous'),
                       shape=list(x.shape), mode='forward_backward', before_strided_result_is_incorrect=strided), step
    elif identifier == 'B007':
        from fla.ops.linear_attn.utils import normalize_with_z_state
        for length, packed in [(1, False), (128, False), (128, True)]:
            q = torch.ones(1, length, 2, 64, device='cuda', dtype=torch.bfloat16)
            o = torch.ones_like(q)
            segments = torch.tensor([0, 64, 128], dtype=torch.int64, device='cuda') if packed else None
            initial = torch.zeros(2 if packed else 1, 1, 2, 64, device='cuda', dtype=q.dtype)
            def step(q=q, o=o, initial=initial, segments=segments):
                return normalize_with_z_state(o, q, q, .125, initial, False, segments)
            yield dict(name=f'normalization_T{length}_packed{packed}', shape=list(q.shape), mode='forward'), step
    elif identifier == 'B038':
        from flag_gems.ops.flash_attention_backward import flash_attn_backward
        for varlen in (False, True):
            m, n, h, d = 128, 256, 2, 32
            q = torch.zeros(1, m, h, d, device='cuda', dtype=torch.float16)
            k = torch.full((1, n, h, d), .125, device='cuda', dtype=q.dtype)
            v = torch.ones_like(k)
            grad = torch.ones_like(q)/q.numel()
            out = torch.ones_like(q)
            lse = torch.full((1, h, m), __import__('math').log(n), device='cuda')
            args = (grad, q, k, v, out, lse)
            kwargs = dict(max_seqlen_q=m, max_seqlen_k=n, softmax_scale=d**-.5)
            if varlen:
                args = tuple(t.squeeze(0) for t in args[:5])+(lse.squeeze(0).T.contiguous(),)
                kwargs.update(cu_seq_q=torch.tensor([0, m], device='cuda', dtype=torch.int32),
                              cu_seq_k=torch.tensor([0, n], device='cuda', dtype=torch.int32))
            def step(args=args, kwargs=kwargs): return flash_attn_backward(*args, **kwargs)
            yield dict(name='attention_backward_'+('varlen' if varlen else 'dense'),
                       q_shape=list(q.shape), k_shape=list(k.shape), mode='backward', dropout=0.), step


def measure(identifier, out):
    torch.manual_seed(73)
    rows = []
    for details, function in tasks(identifier):
        function()
        torch.cuda.synchronize()
        samples = [float(triton.testing.do_bench(function, warmup=20, rep=100)) for _ in range(3)]
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = function()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        rows.append(dict(**details, milliseconds=samples, median_ms=statistics.median(samples),
                         resident_bytes=allocated, peak_allocated_bytes=peak, incremental_peak_bytes=max(0, peak-allocated)))
        del result
    report = dict(case=identifier, torch=torch.__version__, triton=triton.__version__,
                  gpu=torch.cuda.get_device_name(), runs=rows,
                  scope='Three warm timing samples per workload in each isolated process; no statistical speedup claim. Incremental peak measured separately from timer allocation. No model or training quality benchmark.')
    with (Path(out)/'benchmark.json').open('x') as f: json.dump(report, f, indent=2)
    print(json.dumps(report))
