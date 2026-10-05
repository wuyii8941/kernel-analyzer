"""B016: Inductor fuses a scatter_add / index_add into a 1-element (or size-1-along-dim) buffer with a consumer of that
buffer into one kernel, and the consumer reads the buffer before the scatter's atomic adds land.

    python bugs/repro/B016_repro_inductor_scatter_size1_fusion.py [cuda|cpu]
"""
import os
import sys

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
print("torch", torch.__version__, dev)


def count(idx):  # number of entries per bucket, 1 bucket
    return torch.zeros(1, device=idx.device).scatter_add(0, idx, torch.ones(idx.shape, device=idx.device)).clamp(min=1)


def count_index_add(idx):
    return torch.zeros(1, device=idx.device).index_add(0, idx, torch.ones(idx.shape, device=idx.device)) + 0.5


def count_2_buckets(idx):  # control: 2 buckets
    return torch.zeros(2, device=idx.device).scatter_add(0, idx, torch.ones(idx.shape, device=idx.device)).clamp(min=1)


def mean_pool(x, batch, size):  # the scatter-mean of torch_geometric.utils.scatter (global_mean_pool)
    count = x.new_zeros(size).scatter_add_(0, batch, x.new_ones(batch.numel())).clamp(min=1)
    out = x.new_zeros(size, x.shape[1]).scatter_add_(0, batch[:, None].expand_as(x), x)
    return out / count[:, None]


def scatter_mean_backward(x, src, idx):  # gradient of scatter_reduce(mean, include_self) for 1-element self
    x = x.detach().requires_grad_(True)
    src = src.detach().requires_grad_(True)
    y = torch.compile(lambda a, b: a.scatter_reduce(0, idx, b, "mean", include_self=True))(x, src)
    return torch.autograd.grad(y.sum(), [x, src])


idx = torch.zeros(5, dtype=torch.long, device=dev)
for fn in (count, count_index_add, count_2_buckets):
    torch._dynamo.reset()
    try:
        c = torch.compile(fn)(idx).tolist()
    except Exception as e:  # noqa: BLE001
        c = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
    print(f"{fn.__name__:18s} eager {fn(idx).tolist()}  compiled {c}")

torch.manual_seed(0)
x = torch.randn(7, 4, device=dev)
for size, batch in ((1, torch.zeros(7, dtype=torch.long, device=dev)),
                    (2, torch.tensor([0, 0, 1, 1, 1, 0, 1], device=dev))):
    torch._dynamo.reset()
    try:
        err = (mean_pool(x, batch, size) - torch.compile(mean_pool)(x, batch, size)).abs().max().item()
        print(f"mean_pool num_graphs={size}: max |eager - compiled| = {err:.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"mean_pool num_graphs={size}: {type(e).__name__}: {str(e).splitlines()[0][:60]}")

torch._dynamo.reset()
try:
    g = scatter_mean_backward(torch.tensor([1.5], device=dev), torch.tensor([2.0], device=dev),
                              torch.zeros(1, dtype=torch.long, device=dev))
    print("scatter_reduce(mean, include_self) backward, 1 element: eager [0.5, 0.5], compiled",
          [t.item() for t in g])
except Exception as e:  # noqa: BLE001
    print("scatter_reduce backward:", type(e).__name__, str(e).splitlines()[0][:60])
