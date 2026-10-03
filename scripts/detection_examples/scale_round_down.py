"""Known positive for the vector mean: y = mul_rd(x, 0.1) rounds every product down, so E[e] < 0."""

import torch

from scripts import mutation_kernels as mk

N = 2048


def make_inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return {"x": torch.randn(N, device="cuda", generator=g), "y": torch.empty(N, device="cuda")}


def run(t):
    mk.m_scale[(N // 256,)](t["x"], t["y"], 0.1, N, BLOCK=256, MUT=1)
