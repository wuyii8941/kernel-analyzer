"""Registry adapter for the reviewed grouped causal softmax probability output.

The generated region has four writes. The shared registry deliberately binds
only the probability output; the other writes remain available through the
specialized multi-output capture and are not silently treated as equivalent.
"""

from kernel_analyzer.grouped_causal_softmax_reference import select_output
from kernel_analyzer.grouped_causal_softmax_source import check_source as _check_source


def check_source(source, symbol):
    contract = _check_source(source, symbol)
    return dict(contract, output_pointer="out_ptr2")


def reference(metadata, candidate, contract):
    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("Grouped causal softmax symbol differs")
    formal = metadata.get("formal_pointer")
    if formal != contract.get("output_pointer") or formal != "out_ptr2":
        raise ValueError("Grouped causal softmax probability boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported grouped causal softmax input/output alias")
    return select_output(
        metadata.get("runtime_pointers") or {},
        candidate,
        formal_pointer=formal,
        rows=contract["rows"],
        width=contract["width"],
        scale=contract["scale"],
    )
