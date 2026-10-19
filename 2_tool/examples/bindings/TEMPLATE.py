"""Binding template for `kernel-analyzer check` (see docs/binding_guide.md).

Fill in NAME, IMPLEMENTATION, make_inputs and run. Leave spec out for mode A (e_num only). An optional spec is an
external numerical reference for screening; this binding interface does not certify it as the task ground truth.
"""
import torch

from kernel_analyzer.check import f64_point_spec  # noqa: F401  (for spec)

NAME = "template"
IMPLEMENTATION = "library / kernel / configuration"


def setup():
    """Optional: compile and warm up (outside the measurement)."""


def make_inputs(seed):
    """One independent draw of the inputs; the same seed must give the same inputs."""
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(64, 256, generator=g).cuda()}


def run(inputs):
    """Call the implementation under test; return {output name: tensor written by its Triton kernel(s)}."""
    raise NotImplementedError("call the kernel here")


# def spec(inputs):
#     """Optional screening reference in float64; not a rigorous enclosure or semantic certificate."""
#     return {"y": f64_point_spec(math_in_float64(inputs["x"].double()).cpu().numpy())}
