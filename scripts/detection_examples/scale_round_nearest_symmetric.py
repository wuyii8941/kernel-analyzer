"""Vector-mean null by symmetry: y = RN(0.1 x) with x symmetric about 0, so E[e] = 0 exactly
(RN(-z) = -RN(z)).  The relative bias of multiplying by 0.1f (about +2.3e-9) is a true alignment effect."""

import torch

from scripts import mutation_kernels as mk

N = 2048


def make_inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return {"x": torch.randn(N, device="cuda", generator=g), "y": torch.empty(N, device="cuda")}


def run(t):
    mk.m_scale[(N // 256,)](t["x"], t["y"], 0.1, N, BLOCK=256, MUT=0)
