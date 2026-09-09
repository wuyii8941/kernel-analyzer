"""Registry adapter for the reviewed in-place softcapped NLL backward."""

import torch

from kernel_analyzer.softcapped_nll_reference import softcapped_nll_backward
from kernel_analyzer.softcapped_nll_runtime import snapshot_inputs
from kernel_analyzer.softcapped_nll_source import check_source


def reference(metadata, candidate, contract):
    if metadata.get("formal_pointer") != contract.get("output_pointer"):
        raise ValueError("Softcapped NLL output pointer differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported softcapped NLL input/output alias")
    if not isinstance(candidate, torch.Tensor):
        raise ValueError("Softcapped NLL candidate must be a tensor")
    snapshot = snapshot_inputs(metadata.get("runtime_pointers") or {}, contract)
    return softcapped_nll_backward(**snapshot).to(candidate.dtype).reshape(candidate.shape)
