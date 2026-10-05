"""Inductor scatter into a one-element target (B016): the scatter index simplifies to a constant, the atomic store
loses its mask, and consumers can be fused ahead of the atomics.  OpCase from tool_spec_cases_inductor (torch.compile,
spec = eager float64 of the same function)."""

import torch

from tool_spec_cases_inductor import OpCase, rn

N, F = 30, 16


def _count(idx):
    return torch.zeros(1, device=idx.device, dtype=torch.float32).scatter_add(
        0, idx, torch.ones(idx.shape, device=idx.device, dtype=torch.float32))


def _count_f64_safe(idx):  # same in float64 for the spec (OpCase casts floating inputs only)
    return torch.zeros(1, device=idx.device, dtype=torch.float64).scatter_add(
        0, idx, torch.ones(idx.shape, device=idx.device, dtype=torch.float64))


def _mean_pool(x, batch):
    count = x.new_zeros(1).scatter_add_(0, batch, x.new_ones(batch.numel())).clamp(min=1)
    out = x.new_zeros(1, x.shape[1]).scatter_add_(0, batch[:, None].expand_as(x), x)
    return out / count[:, None]


def _mean_pool2(x, batch):
    count = x.new_zeros(2).scatter_add_(0, batch, x.new_ones(batch.numel())).clamp(min=1)
    out = x.new_zeros(2, x.shape[1]).scatter_add_(0, batch[:, None].expand_as(x), x)
    return out / count[:, None]


def _scaled_count(x, idx):  # count used by a consumer, as a float input scales it
    c = x.new_zeros(1).scatter_add(0, idx, torch.ones_like(x))
    return x.sum() / c


CASES = [
    OpCase("sc1_mean_pool_one_graph", lambda x, batch: _mean_pool(x, batch),
           lambda g: {"x": rn(g, N, F), "batch": torch.zeros(N, dtype=torch.long)},
           doc="scatter-mean pooling (torch_geometric.utils.scatter, reduce='mean') with one segment"),
    OpCase("sc1_mean_pool_two_graphs", lambda x, batch: _mean_pool2(x, batch),
           lambda g: {"x": rn(g, N, F), "batch": (torch.arange(N) >= N // 3).long()},
           doc="control: two segments"),
    OpCase("sc1_scaled_count", lambda x, idx: _scaled_count(x, idx),
           lambda g: {"x": rn(g, N), "idx": torch.zeros(N, dtype=torch.long)},
           doc="sum(x) / count, count by scatter_add of ones into one bucket"),
]


# strict enclosures of f (plan WP2): exact rational arithmetic on the float32 inputs instead of float64 eager
class StrictOpCase(OpCase):
    spec_bound = "strict enclosure: exact rational arithmetic on the float32 inputs, rounded outward to float64"

    def __init__(self, name, fn, make, strict, doc=""):
        super().__init__(name, fn, make, doc=doc)
        self.strict = strict

    def spec(self, inp):
        return {"out": self.strict(inp)}


def _strict_mean_pool(inp):
    from strict_specs import scatter_mean_one_segment
    return scatter_mean_one_segment(inp["x"].cpu().numpy())


def _strict_scaled_count(inp):
    from strict_specs import scaled_count
    return scaled_count(inp["x"].cpu().numpy())


CASES += [
    StrictOpCase("sc1_mean_pool_one_graph_strict", lambda x, batch: _mean_pool(x, batch),
                 lambda g: {"x": rn(g, N, F), "batch": torch.zeros(N, dtype=torch.long)}, _strict_mean_pool,
                 doc="scatter-mean of one segment = column means, exact"),
    StrictOpCase("sc1_scaled_count_strict", lambda x, idx: _scaled_count(x, idx),
                 lambda g: {"x": rn(g, N), "idx": torch.zeros(N, dtype=torch.long)}, _strict_scaled_count,
                 doc="sum(x) / N, exact"),
]
