#!/usr/bin/env python
"""Inspect whether an Inductor backend artifact exposes its generated module."""

import json
import sys

import torch
from torch import nn
from torch._dynamo.backends.registry import lookup_backend


class Subject(nn.Module):
    def forward(self, x):
        return torch.rsqrt((x.float().pow(2).mean(-1, keepdim=True) + 1e-6)) * x


captured = {}
inductor = lookup_backend("inductor")


def backend(gm, example_inputs):
    artifact = inductor(gm, example_inputs)
    captured["artifact"] = artifact
    return artifact


subject = torch.compile(Subject().cuda(), backend=backend)
subject(torch.randn(4, 32, device="cuda", dtype=torch.float16))
artifact = captured["artifact"]
current = getattr(artifact, "current_callable", None)
owner = getattr(current, "__self__", None)
module_name = owner.__class__.__module__ if owner is not None else None
module = sys.modules.get(module_name) if module_name else None
kernel_names = []
if module is not None:
    kernel_names = sorted(name for name, value in vars(module).items() if hasattr(value, "run"))
print(json.dumps({
    "artifact_type": f"{type(artifact).__module__}.{type(artifact).__name__}",
    "has_current_callable": current is not None,
    "current_callable_type": f"{type(current).__module__}.{type(current).__name__}" if current is not None else None,
    "has_bound_owner": owner is not None,
    "owner_type": f"{type(owner).__module__}.{type(owner).__name__}" if owner is not None else None,
    "generated_module_name": module_name,
    "generated_module_resolved": module is not None,
    "kernel_names": kernel_names,
    "cache_key": getattr(artifact, "cache_key", None),
}, indent=2, sort_keys=True))
