"""Source contract for a pure Inductor Triton embedding lookup.

This checker accepts only the complete, pointwise lookup body.  Fused
normalization, loss, or embedding-gradient kernels are intentionally outside
this family even when their generated name contains ``embedding``.
"""

import ast
import hashlib


BODY = """
xnumel = {elements}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = tl.full([XBLOCK], True, tl.int1)[:]
x1 = xindex // {width}
x0 = (xindex % {width})
x2 = xindex
tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
tl.device_assert((0 <= tmp0) & (tmp0 < {vocab}), "index out of bounds: 0 <= tmp0 < {vocab}")
tmp2 = tl.load(in_ptr1 + (x0 + {width}*tmp0), None).to(tl.float32)
tl.store(out_ptr0 + (x2), tmp2, None)
"""


def _literal_assignment(function, name):
    values = [
        node.value for node in function.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    if len(values) != 1:
        raise ValueError("Missing or ambiguous assignment: " + name)
    return ast.literal_eval(values[0])


def _integer_from_compare(function):
    values = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare):
            continue
        if (isinstance(node.left, ast.Name) and node.left.id == "tmp0"
                and len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt)
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)):
            values.append(node.comparators[0].value)
    values = [value for value in values if type(value) is int and value > 0]
    if len(set(values)) != 1:
        raise ValueError("Unique embedding vocabulary bound required")
    return values[0]


def _width_from_floor_div(function):
    values = []
    for node in ast.walk(function):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv)
                and isinstance(node.left, ast.Name) and node.left.id == "xindex"
                and isinstance(node.right, ast.Constant)):
            values.append(node.right.value)
    values = [value for value in values if type(value) is int and value > 0]
    if len(set(values)) != 1:
        raise ValueError("Unique embedding width required")
    return values[0]


def check_source(source, symbol):
    outer = ast.parse(source)
    assignments = [
        node for node in outer.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == symbol for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Missing or ambiguous kernel")
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "triton" or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError("Literal Triton source required")
    functions = [
        node for node in ast.parse(call.args[1].value).body
        if isinstance(node, ast.FunctionDef) and node.name == symbol
    ]
    if len(functions) != 1:
        raise ValueError("Missing or ambiguous Triton function")
    function = functions[0]
    names = ["in_ptr0", "in_ptr1", "out_ptr0", "xnumel", "XBLOCK"]
    if ([argument.arg for argument in function.args.args] != names
            or function.args.defaults or function.args.posonlyargs
            or function.args.kwonlyargs or function.args.vararg or function.args.kwarg):
        raise ValueError("Unexpected embedding argument ABI")
    elements = _literal_assignment(function, "xnumel")
    width = _width_from_floor_div(function)
    vocab = _integer_from_compare(function)
    if (type(elements) is not int or elements <= 0 or elements % width
            or elements >= 2**31 or vocab >= 2**31):
        raise ValueError("Invalid embedding dimensions")
    expected = ast.parse(BODY.format(elements=elements, width=width, vocab=vocab)).body
    if [ast.dump(node) for node in function.body] != [ast.dump(node) for node in expected]:
        raise ValueError("Embedding arithmetic, bounds, loads, stores, or indexing differs")

    signatures = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg != "triton_meta" or not isinstance(keyword.value, ast.Dict):
                continue
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if isinstance(key, ast.Constant) and key.value == "signature":
                    signatures.append(ast.literal_eval(value))
    expected_signature = {
        "in_ptr0": "*i64", "in_ptr1": "*bf16", "out_ptr0": "*bf16", "xnumel": "i32"
    }
    if signatures != [expected_signature]:
        raise ValueError("Embedding storage types differ")
    return {
        "symbol": symbol,
        "elements": elements,
        "tokens": elements // width,
        "width": width,
        "vocabulary_size": vocab,
        "input_pointers": ["in_ptr0", "in_ptr1"],
        "output_pointer": "out_ptr0",
        "output_pointers": ["out_ptr0"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "function_ast_sha256": hashlib.sha256(ast.dump(function).encode()).hexdigest(),
        "runtime_binding_complete": False,
        "proof_scope": "PURE_EMBEDDING_LOOKUP_SOURCE_ONLY_NOT_RUNTIME_OR_BIAS_PROOF",
    }
