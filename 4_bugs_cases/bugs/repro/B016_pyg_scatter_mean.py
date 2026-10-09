"""B016 on PyG's scatter-mean: torch_geometric.utils.scatter(reduce='mean') (the code under global_mean_pool and
mean aggregation), taken verbatim from pyg-team/pytorch_geometric master (torch_geometric/utils/_scatter.py, the
'mean' branch and its `broadcast` helper), compiled with torch.compile for one segment (one graph in the batch).

    python bugs/repro/B016_pyg_scatter_mean.py
"""
import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch  # noqa: E402
from torch import Tensor  # noqa: E402


def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:  # torch_geometric/utils/_scatter.py
    dim = ref.dim() + dim if dim < 0 else dim
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


def scatter_mean(src: Tensor, index: Tensor, dim: int = 0, dim_size: int = None) -> Tensor:
    # torch_geometric.utils.scatter, reduce == 'mean' branch
    dim = src.dim() + dim if dim < 0 else dim
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]
    count = src.new_zeros(dim_size)
    count.scatter_add_(0, index, src.new_ones(src.size(dim)))
    count = count.clamp(min=1)
    index = broadcast(index, src, dim)
    out = src.new_zeros(size).scatter_add_(dim, index, src)
    return out / broadcast(count, out, dim)


def global_mean_pool(x, batch, size):  # torch_geometric.nn.pool.glob.global_mean_pool with a batch vector
    return scatter_mean(x, batch, dim=-2, dim_size=size)


torch.manual_seed(0)
print("torch", torch.__version__)
for num_nodes in (7, 30, 300):
    x = torch.randn(num_nodes, 16, device="cuda")
    for num_graphs in (1, 2):
        batch = torch.zeros(num_nodes, dtype=torch.long, device="cuda") if num_graphs == 1 else \
            (torch.arange(num_nodes, device="cuda") >= num_nodes // 2).long()
        torch._dynamo.reset()
        ref = global_mean_pool(x, batch, num_graphs)
        out = torch.compile(global_mean_pool)(x, batch, num_graphs)
        ratio = (out / ref)[0].mean().item()
        print(f"nodes={num_nodes:4d} graphs={num_graphs}: max|eager-compiled| {(out - ref).abs().max().item():.2e}, "
              f"compiled/eager for graph 0 = {ratio:.4f}")
