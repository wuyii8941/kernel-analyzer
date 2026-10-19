"""Conservative inspection of a generated wrapper's consecutive launch region.

This records compiler structure, not a proof about arbitrary Python helper code.
It neither executes the wrapper nor derives the task from kernel arithmetic.
"""
import ast


def check_launch_region(source, kernels):
    base = {"binding_status": "not_established", "semantic_verdict": "not_established"}

    def refuse(reason):
        return dict(base, status="not_established", reason=reason)

    if not kernels or len(set(kernels)) != len(kernels):
        return refuse("region requires distinct, explicit kernel names")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return refuse("wrapper syntax is invalid")

    def launch(node):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            return None
        call = node.value
        if isinstance(call.func, ast.Attribute) and call.func.attr == "run" and isinstance(call.func.value, ast.Name):
            return call.func.value.id
        return None

    occurrences, blocks = {}, []
    for node in ast.walk(tree):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(x, ast.stmt) for x in value):
                blocks.append(value)
                for i, statement in enumerate(value):
                    name = launch(statement)
                    if name in kernels:
                        occurrences.setdefault(name, []).append((value, i))
    if any(len(occurrences.get(k, [])) != 1 for k in kernels):
        return refuse("selected launch sites are absent or ambiguous")
    selected = [occurrences[k][0] for k in kernels]
    block = selected[0][0]
    positions = [i for _, i in selected]
    if any(b is not block for b, _ in selected) or positions != sorted(positions):
        return refuse("launches do not share one statement block in the requested order")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    owner = block[positions[0]]
    while owner in parents and not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
        owner = parents[owner]
    if not isinstance(owner, ast.FunctionDef):
        return refuse("launch region has no ordinary wrapper function")

    def plain(node):
        return (isinstance(node, (ast.Name, ast.Constant))
                or isinstance(node, (ast.Tuple, ast.List)) and all(plain(x) for x in node.elts)
                or isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch")

    calls, housekeeping = [], []
    for statement in block[positions[0]:positions[-1] + 1]:
        name = launch(statement)
        if name in kernels:
            call = statement.value
            if not all(plain(x) for x in call.args) or not all(k.arg == "stream" and plain(k.value) for k in call.keywords):
                return refuse("launch arguments contain unsupported evaluation")
            calls.append({"kernel": name, "line": statement.lineno,
                          "args": [ast.unparse(x) for x in call.args],
                          "stream": next((ast.unparse(k.value) for k in call.keywords if k.arg == "stream"), None)})
            continue
        safe = isinstance(statement, ast.Delete) and all(isinstance(x, ast.Name) for x in statement.targets)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            value = statement.value
            safe = isinstance(value, ast.Name)
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                safe = (value.func.id in {"empty_strided_cuda", "reinterpret_tensor", "get_raw_stream"}
                        and not value.keywords and all(plain(x) for x in value.args))
        if not safe:
            return refuse(f"intervening statement at line {statement.lineno} requires an explicit effect mapping")
        housekeeping.append({"line": statement.lineno, "statement": ast.unparse(statement)})
    return dict(base, status="wrapper launch region checked", calls=calls, housekeeping=housekeeping,
                wrapper_function_line=owner.lineno,
                limitation="compiler wrapper structure only; helper semantics, captures and task outputs require separate checks")


def map_return_values(source, task_graph, region):
    """Pair explicit positional returns of the same compiled graph and wrapper.

    The caller must check graph ownership. No deduplication: repeated returns
    and tensor views remain separate positions, and ambiguity is rejected.
    """
    if region.get("status") != "wrapper launch region checked":
        raise ValueError("wrapper launch region has not been checked")
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.lineno == region["wrapper_function_line"]]
    if len(functions) != 1:
        raise ValueError("wrapper function is not unique")
    # Reject nested definitions/control-flow returns rather than guessing a path.
    owner = functions[0]
    returns = [n for n in ast.walk(owner) if isinstance(n, ast.Return)]
    if len(returns) != 1 or returns[0] not in owner.body or not isinstance(returns[0].value, ast.Tuple):
        raise ValueError("wrapper requires a single unconditional tuple return")
    last_launch = region["calls"][-1]["line"]
    for statement in ast.walk(owner):
        if isinstance(statement, ast.stmt) and last_launch < statement.lineno < returns[0].lineno:
            if not (isinstance(statement, ast.Delete) and all(isinstance(t, ast.Name) for t in statement.targets)):
                raise ValueError("work or buffer reassignment after the region requires an explicit mapping")
    outputs = [n for n in task_graph["nodes"] if n["op"] == "output"]
    if len(outputs) != 1:
        raise ValueError("task graph output is not unique")
    parsed = ast.parse(outputs[0]["expression"])
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Return) or not isinstance(parsed.body[0].value, ast.Tuple):
        raise ValueError("task graph requires an explicit tuple return")
    graph_values, wrapper_values = parsed.body[0].value.elts, returns[0].value.elts
    if len(graph_values) != len(wrapper_values):
        raise ValueError("graph and wrapper return counts disagree")
    names = {n["name"] for n in task_graph["nodes"]}
    result = []
    for i, (graph_value, value) in enumerate(zip(graph_values, wrapper_values)):
        if isinstance(graph_value, ast.Constant) and graph_value.value is None:
            if not isinstance(value, ast.Constant) or value.value is not None:
                raise ValueError("absent task output and wrapper return disagree")
            result.append({"position": i, "constant": None})
            continue
        if not isinstance(graph_value, ast.Name) or graph_value.id not in names:
            raise ValueError("task output is not a recorded node")
        view = None
        if isinstance(value, ast.Call):
            if (not isinstance(value.func, ast.Name) or value.func.id != "reinterpret_tensor"
                    or len(value.args) != 4 or value.keywords):
                raise ValueError("unsupported return expression")
            shape, stride, offset = [ast.literal_eval(x) for x in value.args[1:]]
            if (not isinstance(shape, tuple) or not isinstance(stride, tuple) or len(shape) != len(stride)
                    or any(type(x) is not int or x < 0 for x in shape + stride)
                    or type(offset) is not int or offset != 0):
                raise ValueError("return view requires literal layout and zero offset")
            view = {"shape": list(shape), "stride": list(stride), "offset": offset}
            value = value.args[0]
        if not isinstance(value, ast.Name):
            raise ValueError("return buffer is not explicit")
        result.append({"position": i, "node": graph_value.id, "buffer": value.id, "view": view})
    return result


def check_local_temporary(source, region, buffer_names):
    """Check a restricted local allocation/alias/delete lifetime in the wrapper.

    Selected kernels' memory effects must still be checked separately: this
    cannot prove absence of pointer escapes inside device code.
    """
    base = {"binding_status": "not_established", "semantic_verdict": "not_established"}

    def refuse(reason):
        return dict(base, status="not_established", reason=reason)

    if region.get("status") != "wrapper launch region checked" or not buffer_names:
        return refuse("checked region and explicit buffer names required")
    tree = ast.parse(source)
    owners = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.lineno == region["wrapper_function_line"]]
    if len(owners) != 1:
        return refuse("wrapper function is not unique")
    owner = owners[0]
    blocks = [value for n in ast.walk(owner) for _, value in ast.iter_fields(n)
              if isinstance(value, list) and any(isinstance(s, ast.stmt) and s.lineno == region["calls"][0]["line"]
                                                for s in value)]
    if len(blocks) != 1:
        return refuse("launch statement block is ambiguous")
    block = blocks[0]
    aliases = set(buffer_names)
    changed = True
    while changed:
        changed = False
        for n in ast.walk(owner):
            if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and isinstance(n.value, ast.Name) and n.value.id in aliases and n.targets[0].id not in aliases):
                aliases.add(n.targets[0].id)
                changed = True
    parents = {child: n for n in ast.walk(owner) for child in ast.iter_child_nodes(n)}
    for n in ast.walk(owner):
        if isinstance(n, ast.Name) and n.id in aliases:
            statement = n
            while statement in parents and not isinstance(statement, ast.stmt):
                statement = parents[statement]
            if statement not in block:
                return refuse("temporary is referenced outside the launch statement block")
    live, created, events, allocations = set(), set(), [], 0
    call_lines = {c["line"] for c in region["calls"]}
    for statement in block:
        refs = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name) and n.id in aliases}
        if not refs:
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target, value = statement.targets[0].id, statement.value
            if target not in aliases or target in created:
                return refuse("temporary name is reassigned or escapes")
            if isinstance(value, ast.Name) and value.id in live:
                live.add(target)
                created.add(target)
                events.append({"line": statement.lineno, "alias": target, "source": value.id})
                continue
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "empty_strided_cuda" and not value.keywords and len(value.args) == 3
                    and all(isinstance(x, ast.Tuple) and all(isinstance(y, ast.Constant) and type(y.value) is int
                                                           and y.value >= 0 for y in x.elts) for x in value.args[:2])
                    and isinstance(value.args[2], ast.Attribute) and isinstance(value.args[2].value, ast.Name)
                    and value.args[2].value.id == "torch"):
                allocations += 1
                if allocations != 1:
                    return refuse("temporary names do not share one local allocation")
                live.add(target)
                created.add(target)
                events.append({"line": statement.lineno, "allocate": target})
                continue
        if isinstance(statement, ast.Delete) and all(isinstance(t, ast.Name) for t in statement.targets):
            if not refs <= live:
                return refuse("temporary is deleted before its lifetime begins")
            live -= refs
            events.append({"line": statement.lineno, "delete": sorted(refs)})
            continue
        if statement.lineno in call_lines and isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            direct = {a.id for a in statement.value.args if isinstance(a, ast.Name)}
            if refs <= live and refs <= direct:
                events.append({"line": statement.lineno, "use": sorted(refs)})
                continue
        return refuse(f"temporary use at line {statement.lineno} is not covered")
    if allocations != 1 or live or created != aliases:
        return refuse("temporary allocation and all alias lifetimes are not closed")
    return dict(base, status="wrapper local temporary checked", aliases=sorted(aliases), events=events,
                limitation="wrapper lifetime only; device pointer escapes and memory effects remain separate obligations")
