"""Extract a mathematical region with explicit, provenance-bound input values."""
from dynamic_three_way import complete_fragment, fx_text, fx_text_refs, tensor_corresponds


def build_task_region(task_graph, inputs, outputs):
    """inputs maps task node names to captured pre-state views.

    The caller establishes each node's provenance and graph ownership. This
    routine checks completeness and metadata; it never chooses nodes by shape.
    """
    names = [n["name"] for n in task_graph["nodes"]]
    if len(set(names)) != len(names) or not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("task nodes and requested outputs must be explicit and unique")
    if not set(inputs) <= set(names) or not set(outputs) <= set(names):
        raise ValueError("region boundary references an absent task node")
    if set(inputs) & set(outputs):
        raise ValueError("computed outputs must be distinct from boundary inputs")
    # Exact names supplied here are already bound by compiler origins. These
    # synthetic parameter labels are internal to graph extraction only.
    params = {name: {"role": "input", "buffer": name, "origin_node": name, "origins": [name]}
              for name in inputs}
    seeds = [{"name": name, "kind": "placeholder", "meta": ""} for name in outputs]
    fragment, _ = complete_fragment(seeds, task_graph, params, {name: name for name in inputs})
    nodes = {n["name"]: n for n in fragment}
    placeholders = {n["name"] for n in fragment if n["kind"] == "placeholder"}
    if placeholders != set(inputs):
        raise ValueError(f"region inputs incomplete or unused: required={sorted(placeholders)}, supplied={sorted(inputs)}")
    refs = {r for n in fragment if n["kind"] == "call" for r in fx_text_refs(n)}
    if refs - set(nodes):
        raise ValueError(f"region has unresolved task dependencies: {sorted(refs - set(nodes))}")
    for name, arg in inputs.items():
        if not tensor_corresponds(nodes[name], arg):
            raise ValueError(f"region input type or shape mismatch: {name}")
        if arg.before is None or arg.window is not None:
            raise ValueError(f"complete region input pre-state missing: {name}")
    graph = fx_text.to_graph(fragment)
    graph["outputs"] = list(outputs)
    return graph
