"""Versioned storage decoder; unchanged gated convolution mathematics.

Retains the original checker and formula while interpreting dense transposed
snapshots in the pointer order used by the reviewed Triton program.
"""
import torch
import hashlib
from pathlib import Path
from kernel_analyzer.gated_conv_gradient_reference import check_source, reference as original_reference
from kernel_analyzer.dense_pointer_view import dense_pointer_view

# The existing family launcher pins this adapter. Pin its reused dependencies
# here as well, so a queued job cannot silently acquire another decoder/formula.
for filename, digest in {
    'gated_conv_gradient_reference.py': 'ff2304a7838d00eb0a0324e8fdad7edd8f7ea4487133dadd8b4f64c9a53f8839',
    'dense_pointer_view.py': '14710c101077ad50e7347dba10c67b52ec66a8cb8a0de063632973698085d7bd',
}.items():
    if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen storage-reference dependency changed: '+filename)


def reference(metadata, candidate, contract):
    channels, steps, padding = (contract[k] for k in ('channels', 'steps', 'padding'))
    shapes = {'in_out_ptr0': (channels, steps+padding), 'in_ptr0': (channels, steps),
              'in_ptr1': (steps, channels), 'in_ptr2': (channels,),
              'in_ptr3': (channels, steps), 'in_ptr4': (channels, steps)}
    pointers = metadata.get('runtime_pointers', {})
    decoded = {}
    for name, shape in shapes.items():
        value = pointers.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError('Missing pointer snapshot: '+name)
        decoded[name] = dense_pointer_view(value, shape)
    # Existing checks retain dtype, device, alias and output requirements.
    return original_reference(dict(metadata, runtime_pointers=decoded), candidate, contract)
