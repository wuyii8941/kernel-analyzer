"""Check the explicit real formula before selecting the independent evaluator.

Accepts one declared graph form; a different legal task is unsupported here,
not mathematically wrong. Node names and output ordering do not select semantics.
"""
from kernel_spec.num import Invalid, NotEstablished


def _expected(rows, classes):
    nodes = []

    def op(name, *args, **kwargs):
        node = {"name": f"s{len(nodes)}", "op": name, "args": list(args), "kwargs": kwargs}
        nodes.append(node)
        return {"ref": node["name"]}

    labels = op("aten.reshape", op("aten.slice", {"ref": "$labels"}, 1, 1, 9223372036854775807), [-1])
    valid = op("aten.ne", labels, -100)
    count = op("prims.convert_element_type", op("aten.sum", valid), "torch.float32")
    x = op("aten.reshape", op("prims.convert_element_type",
           op("aten.reshape", {"ref": "$logits"}, [1, rows, classes]), "torch.float32"), [-1, classes])
    pair = op("prims.prepare_softmax_online", x, 1)
    maximum = op("operator.getitem", pair, 0)
    log_mass = op("aten.log", op("operator.getitem", pair, 1))
    log_probs = op("aten.sub", op("aten.sub", x, maximum), log_mass)
    selected = op("aten.where", valid, labels, op("aten.full", [], 0, dtype="torch.int64"))
    losses = op("aten.neg", op("aten.squeeze", op("aten.gather", log_probs, 1,
                op("aten.unsqueeze", selected, 1)), 1))
    loss = op("aten.div", op("aten.sum", op("aten.where", valid, losses,
              op("aten.full", [], 0.0, dtype="torch.float32"))), count)
    return {"nodes": nodes}, {"mean_loss": loss["ref"], "maxima": maximum["ref"],
                               "log_mass": log_mass["ref"], "count": count["ref"]}


def _expressions(graph, inputs, outputs):
    nodes = {n["name"]: n for n in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]) or set(nodes) & set(inputs):
        raise Invalid("duplicate or shadowed graph names")
    seen, visiting, cache = set(), set(), {}

    def value(x):
        if isinstance(x, dict) and set(x) == {"ref"}:
            return expand(x["ref"])
        if isinstance(x, list):
            return ("list", tuple(value(v) for v in x))
        if x is None or type(x) in (str, bool, int, float):
            return (type(x).__name__, x)
        raise NotEstablished("unsupported graph argument form")

    def expand(name):
        if name in inputs:
            return ("input", inputs[name])
        if name in cache:
            return cache[name]
        if name in visiting or name not in nodes:
            raise Invalid("cyclic or unresolved task dependency")
        n = nodes[name]
        if set(n) - {"name", "op", "args", "kwargs"}:
            raise NotEstablished("additional node semantics require review")
        visiting.add(name)
        result = ("call", n["op"], tuple(value(x) for x in n.get("args", [])),
                  tuple(sorted((key, value(x)) for key, x in n.get("kwargs", {}).items())))
        visiting.remove(name)
        seen.add(name)
        cache[name] = result
        return result

    result = {name: expand(name) for name in outputs}
    if seen != set(nodes):
        raise NotEstablished("task has additional nodes outside the declared outputs")
    return result


def require_region_formula(graph, *, logits_input, labels_input, logits_shape):
    """Return output-role -> actual node name, or reject without judging the task.

    Input type/layout/provenance checks remain the caller's obligations. The
    FP32 count conversion is exact only within the supported row limit.
    """
    if set(graph) - {"inputs", "nodes", "outputs"} or graph.get("inputs"):
        raise NotEstablished("additional graph declarations require review")
    if (len(logits_shape) != 2 or any(type(n) is not int or n <= 0 for n in logits_shape)
            or logits_shape[0] > 2**24 or logits_input == labels_input):
        raise NotEstablished("unsupported input shape or count conversion")
    outputs = graph.get("outputs", [])
    if len(outputs) != 4 or not all(isinstance(n, str) for n in outputs) or len(set(outputs)) != 4:
        raise NotEstablished("all four distinct task outputs are required")
    actual = _expressions(graph, {logits_input: "logits", labels_input: "labels"}, outputs)
    expected, roles = _expected(*logits_shape)
    formulas = _expressions(expected, {"$logits": "logits", "$labels": "labels"}, list(roles.values()))
    mapping = {}
    for role, node in roles.items():
        matches = [name for name, expression in actual.items() if expression == formulas[node]]
        if len(matches) != 1:
            raise NotEstablished(f"selected task does not establish the {role} formula")
        mapping[role] = matches[0]
    return mapping
