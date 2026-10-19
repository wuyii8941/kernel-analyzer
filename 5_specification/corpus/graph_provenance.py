"""Record the compiler's complete FX graph without evaluating its operations.

These records supply task-binding evidence; they are not an independent reference.
"""
import torch


def value_metadata(value):
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "dtype": str(value.dtype),
                "shape": [str(x) for x in value.shape],
                "stride": [str(x) for x in value.stride()],
                "storage_offset": str(value.storage_offset())}
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [value_metadata(x) for x in value]}
    if value is None or type(value) in (bool, int, float, str):
        return {"kind": type(value).__name__, "value": repr(value)}
    return {"kind": "unsupported", "type": type(value).__name__}


def snapshot_graph(module):
    """Keep exact FX argument expressions, node order and tuple-valued metadata.

    Expressions are diagnostic text, never executable input to the reference.
    Missing metadata stays missing; symbolic dimensions are not converted to ints.
    """
    nodes = []
    for node in module.graph.nodes:
        record = {"name": node.name, "op": node.op, "expression": node.format_node(),
                  "inputs": [x.name for x in node.all_input_nodes]}
        if "val" in node.meta:
            record["value"] = value_metadata(node.meta["val"])
        nodes.append(record)
    return {"nodes": nodes, "semantic_verdict": "not_established",
            "source": "same GraphLowering as the generated kernels"}
