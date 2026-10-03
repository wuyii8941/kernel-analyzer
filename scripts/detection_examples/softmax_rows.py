"""Softmax over rows of 3 x N(0, 1) logits (the tool's earlier localization found the exp node biased)."""

import torch

from scripts import mutation_kernels as mk


def make_inputs(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(16, 200, device="cuda", generator=g) * 3.0
    return {"x": x, "y": torch.empty_like(x)}


def run(t):
    mk.m_softmax[(16,)](t["x"], t["y"], 200, 200, BLOCK=256, MUT=0)
