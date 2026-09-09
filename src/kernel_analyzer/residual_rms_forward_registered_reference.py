"""Registry adapter for the reviewed residual-add/RMSNorm normalized output.

The source writes the residual and inverse RMS in addition to the normalized
tensor. The general registry binds only ``out_ptr0``; specialized capture can
still inspect all three writes without changing this family definition.
"""

from kernel_analyzer.residual_rms_forward_reference import select_output
from kernel_analyzer.residual_rms_forward_source import check_source as _check_source


def check_source(source, symbol):
    contract = _check_source(source, symbol)
    return dict(contract, output_pointer="out_ptr0")


def reference(metadata, candidate, contract):
    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("Residual RMS symbol differs")
    formal = metadata.get("formal_pointer")
    if formal != contract.get("output_pointer") or formal != "out_ptr0":
        raise ValueError("Residual RMS normalized-output boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported residual RMS input/output alias")
    return select_output(
        metadata.get("runtime_pointers") or {},
        candidate,
        formal_pointer=formal,
        rows=contract["rows"],
        width=contract["width"],
        epsilon=contract["epsilon"],
    )
