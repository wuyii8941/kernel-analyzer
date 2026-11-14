"""Bounded forward, backward, and full timings for the precision changes."""
import torch

import microbench


def tasks(identifier):
    if identifier == 'B007':
        from fla.ops.linear_attn.utils import normalize_with_z_state
        variants = [(1, False), (128, False), (128, True)]
        for length, packed in variants:
            q = torch.ones(1, length, 2, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            k = torch.ones_like(q, requires_grad=True)
            o = torch.ones_like(q, requires_grad=True)
            initial = torch.zeros(2 if packed else 1, 1, 2, 64, device='cuda', dtype=q.dtype,
                                  requires_grad=True)
            segments = torch.tensor([0, 64, 128], device='cuda', dtype=torch.int64) if packed else None
            def forward(o=o, q=q, k=k, initial=initial, segments=segments):
                return normalize_with_z_state(o, q, k, .125, initial, False, segments)
            inputs = (o, q, k, initial)
            result = forward()
            upstream = tuple(torch.ones_like(v) for v in result)
            for mode in ['forward', 'backward', 'full']:
                if mode == 'forward':
                    step = forward
                elif mode == 'backward':
                    def step(result=result, inputs=inputs, upstream=upstream):
                        return torch.autograd.grad(result, inputs, upstream, retain_graph=True)
                else:
                    def step(forward=forward, inputs=inputs, upstream=upstream):
                        return torch.autograd.grad(forward(), inputs, upstream)
                yield dict(name=f'normalization_T{length}_packed{packed}_{mode}',
                           shape=list(q.shape), dtype=str(q.dtype), mode=mode), step
    elif identifier == 'B049':
        from liger_kernel.transformers.kl_div import LigerKLDIVLoss
        for provider in ['liger', 'torch']:
            for dtype in [torch.float16, torch.float32]:
                x = torch.randn(32, 32768, device='cuda').log_softmax(-1).detach().requires_grad_()
                target = torch.full_like(x, 1/32768, dtype=dtype)
                criterion = LigerKLDIVLoss(reduction='mean') if provider == 'liger' else torch.nn.KLDivLoss(reduction='mean')
                def forward(x=x, target=target, criterion=criterion):
                    return criterion(x, target)
                value = forward()
                for mode in ['forward', 'backward', 'full']:
                    if mode == 'forward':
                        step = forward
                    elif mode == 'backward':
                        def step(value=value, x=x):
                            return torch.autograd.grad(value, x, retain_graph=True)
                    else:
                        def step(forward=forward, x=x):
                            return torch.autograd.grad(forward(), x)
                    yield dict(name=f'{provider}_mean_kl_{dtype}_{mode}', provider=provider,
                               shape=list(x.shape), target_dtype=str(dtype), mode=mode), step
    else:
        raise ValueError(identifier)


def measure(identifier, output):
    microbench.tasks = tasks
    microbench.measure(identifier, output)
