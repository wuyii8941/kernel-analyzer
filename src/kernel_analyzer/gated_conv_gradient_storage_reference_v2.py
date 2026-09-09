"""Current storage decoder for the reviewed gated-convolution gradient formula.

The original adapter remains frozen for historical result reproduction.  This
version pins the current mathematical reference instead of changing that old
adapter's dependency hash after the fact.
"""

import hashlib
from pathlib import Path

import torch

from kernel_analyzer.dense_pointer_view import dense_pointer_view
from kernel_analyzer.gated_conv_gradient_reference import reference as original_reference


for filename, digest in {
    "gated_conv_gradient_reference.py": "6374d5c040f05cbaf6850a4dcc774685e87a1f58419fa0181b717d99625ed83c",
    "dense_pointer_view.py": "14710c101077ad50e7347dba10c67b52ec66a8cb8a0de063632973698085d7bd",
}.items():
    if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != digest:
        raise ValueError("Current storage-reference dependency changed: " + filename)


def reference(metadata, candidate, contract):
    channels, steps, padding = (contract[key] for key in ("channels", "steps", "padding"))
    shapes = {
        "in_out_ptr0": (channels, steps + padding),
        "in_ptr0": (channels, steps),
        "in_ptr1": (steps, channels),
        "in_ptr2": (channels,),
        "in_ptr3": (channels, steps),
        "in_ptr4": (channels, steps),
    }
    pointers = metadata.get("runtime_pointers", {})
    decoded = {}
    for name, shape in shapes.items():
        value = pointers.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError("Missing pointer snapshot: " + name)
        decoded[name] = dense_pointer_view(value, shape)
    return original_reference(dict(metadata, runtime_pointers=decoded), candidate, contract)
