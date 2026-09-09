"""Registry adapter for the reviewed in-place selected-logit NLL backward."""

import torch

from kernel_analyzer.selected_nll_runtime import reference_write, snapshot_inputs
from kernel_analyzer.selected_nll_source import check_source


def reference(metadata, candidate, contract):
    if metadata.get("formal_pointer") != contract.get("output_pointer"):
        raise ValueError("Selected NLL output pointer differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported selected NLL input/output alias")
    if not isinstance(candidate, torch.Tensor):
        raise ValueError("Selected NLL candidate must be a tensor")
    snapshot = snapshot_inputs(metadata.get("runtime_pointers") or {}, contract)
    return reference_write(snapshot).to(candidate.dtype).reshape(candidate.shape)
