"""Binding example for `kernel-analyzer check` (mode B: with a screening reference).

A binding gives one independent draw of the inputs per seed and the call. The optional float64 reference does not
provide contract review or kernel/task binding, so the report cannot certify a semantic error.
"""
import numpy as np
import torch

from kernel_analyzer.check import f64_point_spec
from scripts.reference_eval_kernels import softmax_rows

NAME = "softmax_rows"
IMPLEMENTATION = "row softmax Triton kernel (scripts/reference_eval_kernels.py)"
SPECIFICATION = "screening reference: softmax over the last axis, float64"
SPEC_BOUND = "heuristic float64 band; not a rigorous enclosure"


def make_inputs(seed):
    g = torch.Generator().manual_seed(seed)
    return {"x": (2 * torch.randn(8, 300, generator=g)).cuda()}


def run(inp):
    x = inp["x"]
    y = torch.empty_like(x)
    softmax_rows[(x.shape[0],)](x, y, x.shape[1], x.stride(0), BLOCK=512)
    return {"y": y}


def spec(inp):
    return {"y": f64_point_spec(torch.softmax(inp["x"].double(), -1).cpu().numpy())}
