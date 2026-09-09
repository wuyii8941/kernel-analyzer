"""Strict same-input reference for generated tanh-GELU product kernels.

This family covers ``gelu_tanh(x) * multiplier`` in two reviewed layouts:
two contiguous inputs, or a contiguous input multiplied by one selected block
from packed storage. In-place variants are deliberately not accepted because
their pre-call snapshot contract differs.
"""

import ast
import hashlib
import math

from kernel_analyzer.dense_pointer_view import dense_pointer_view


def _body(elements, *, width=None, stride=None, offset=None):
    if width is None:
        indexing = "x0 = xindex"
        first_address = second_address = output_address = "x0"
    else:
        indexing = f"x2 = xindex\nx0 = (xindex % {width})\nx1 = xindex // {width}"
        first_address = output_address = "x2"
        selected = f"x0 + {stride}*x1"
        second_address = selected if offset == 0 else f"{offset} + {selected}"
    return ast.parse(f'''
xnumel = {elements}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = tl.full([XBLOCK], True, tl.int1)[:]
{indexing}
tmp0 = tl.load(in_ptr0 + ({first_address}), None).to(tl.float32)
tmp16 = tl.load(in_ptr1 + ({second_address}), None).to(tl.float32)
tmp1 = tmp0.to(tl.float32)
tmp2 = tl.full([1], 0.5, tl.float32)
tmp3 = tmp1 * tmp2
tmp4 = tmp1 * tmp1
tmp5 = tmp4 * tmp1
tmp6 = tl.full([1], 0.044715, tl.float32)
tmp7 = tmp5 * tmp6
tmp8 = tmp1 + tmp7
tmp9 = tl.full([1], 0.7978845608028654, tl.float32)
tmp10 = tmp8 * tmp9
tmp11 = libdevice.tanh(tmp10)
tmp12 = tl.full([1], 1.0, tl.float32)
tmp13 = tmp11 + tmp12
tmp14 = tmp3 * tmp13
tmp15 = tmp14.to(tl.float32)
tmp17 = tmp15 * tmp16
tl.store(out_ptr0 + ({output_address}), tmp17, None)
''').body


def _integer(node, message):
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        raise ValueError(message)
    return node.value


def _function(source, symbol):
    assignments = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == symbol
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Missing or ambiguous forward GELU source")
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "triton" or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError("Literal Triton forward GELU source required")
    functions = [
        node for node in ast.parse(call.args[1].value).body
        if isinstance(node, ast.FunctionDef) and node.name == symbol
    ]
    if len(functions) != 1:
        raise ValueError("Missing or ambiguous forward GELU function")
    return functions[0]


def _check_signature(function):
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
    expected = dict(in_ptr0="*bf16", in_ptr1="*bf16", out_ptr0="*bf16", xnumel="i32")
    if signatures != [expected]:
        raise ValueError("Forward GELU storage types differ")


def _selected_layout(values):
    try:
        width = _integer(values["x0"].right, "Literal GELU width required")
        if not (isinstance(values["x0"].op, ast.Mod)
                and isinstance(values["x1"], ast.BinOp)
                and isinstance(values["x1"].op, ast.FloorDiv)):
            raise ValueError("Forward GELU selection indexing differs")
        load = values["tmp16"]
        if not (isinstance(load, ast.Call) and isinstance(load.func, ast.Attribute)
                and load.func.attr == "to"):
            raise ValueError("Forward GELU multiplier load differs")
        address = load.func.value.args[0].right
        if (not isinstance(address, ast.BinOp) or not isinstance(address.op, ast.Add)
                or not isinstance(address.right, ast.BinOp)
                or not isinstance(address.right.op, ast.Mult)):
            raise ValueError("Forward GELU selected address differs")
        stride = _integer(address.right.left, "Literal GELU stride required")
        if isinstance(address.left, ast.Name) and address.left.id == "x0":
            offset = 0
        elif (isinstance(address.left, ast.BinOp) and isinstance(address.left.op, ast.Add)
              and isinstance(address.left.right, ast.Name) and address.left.right.id == "x0"):
            offset = _integer(address.left.left, "Literal GELU offset required")
        else:
            raise ValueError("Forward GELU selected address differs")
    except (KeyError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Forward GELU selected layout is not reviewed") from exc
    return width, stride, offset


def check_source(source, symbol):
    function = _function(source, symbol)
    if ([arg.arg for arg in function.args.args]
            != ["in_ptr0", "in_ptr1", "out_ptr0", "xnumel", "XBLOCK"]
            or function.args.defaults or function.args.posonlyargs
            or function.args.kwonlyargs or function.args.vararg or function.args.kwarg):
        raise ValueError("Forward GELU signature differs")
    _check_signature(function)
    values = {
        target.id: statement.value
        for statement in function.body if isinstance(statement, ast.Assign)
        for target in statement.targets if isinstance(target, ast.Name)
    }
    try:
        elements = _integer(values["xnumel"], "Literal GELU extent required")
    except KeyError as exc:
        raise ValueError("Forward GELU extent is missing") from exc
    if elements <= 0:
        raise ValueError("Invalid forward GELU extent")
    if "x2" in values:
        width, stride, offset = _selected_layout(values)
        if (width <= 0 or stride < width or offset < 0 or offset + width > stride
                or elements % width):
            raise ValueError("Invalid forward GELU selected layout")
        layout = "PACKED_SELECTION"
        expected = _body(elements, width=width, stride=stride, offset=offset)
    else:
        width = stride = offset = None
        layout = "CONTIGUOUS"
        expected = _body(elements)
    if [ast.dump(node) for node in function.body] != [ast.dump(node) for node in expected]:
        raise ValueError("Forward GELU arithmetic or addressing differs")
    return dict(
        family="GELU_FORWARD_PRODUCT_V1", symbol=symbol, elements=elements,
        layout=layout, width=width, stride=stride, offset=offset,
        output_pointer="out_ptr0",
        reference_variant="TANH_GELU_PRODUCT_FP32_BF16_WRITE",
        mathematical_expression="tanh-approximate GELU(input) * multiplier",
        function_ast_sha256=hashlib.sha256(ast.dump(function).encode()).hexdigest(),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_binding_complete=False,
    )


def reference(metadata, candidate, contract):
    import torch

    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("Forward GELU symbol differs")
    if metadata.get("formal_pointer") != contract.get("output_pointer"):
        raise ValueError("Forward GELU output boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Forward GELU input/output alias is unsupported")
    pointers = metadata.get("runtime_pointers") or {}
    if set(pointers) != {"in_ptr0", "in_ptr1", "out_ptr0"}:
        raise ValueError("Unexpected forward GELU pointer set")
    elements = contract["elements"]
    first, second, output = (pointers[name] for name in ("in_ptr0", "in_ptr1", "out_ptr0"))
    expected_second = (elements if contract["layout"] == "CONTIGUOUS"
                       else (elements // contract["width"]) * contract["stride"])
    for name, value, size in (("in_ptr0", first, elements),
                              ("in_ptr1", second, expected_second),
                              ("out_ptr0", output, elements)):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16
                or value.device != candidate.device or value.numel() != size):
            raise ValueError("Forward GELU pointer differs: " + name)
        if name != "out_ptr0" and not torch.isfinite(value).all():
            raise ValueError("Nonfinite forward GELU input: " + name)
    if (candidate.dtype != torch.bfloat16 or candidate.numel() != elements
            or not candidate.is_contiguous()):
        raise ValueError("Forward GELU candidate representation differs")
    x = dense_pointer_view(first, (elements,)).detach().float().clone()
    if contract["layout"] == "CONTIGUOUS":
        multiplier = dense_pointer_view(second, (elements,)).detach().float().clone()
    else:
        rows = elements // contract["width"]
        packed = dense_pointer_view(second, (rows, contract["stride"]))
        start = contract["offset"]
        multiplier = packed[:, start:start + contract["width"]].reshape(-1).float().clone()
    argument = (x + 0.044715 * (x * x * x)) * math.sqrt(2 / math.pi)
    result = (0.5 * x * (1.0 + torch.tanh(argument))) * multiplier
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite forward GELU reference arithmetic")
    return result.to(candidate.dtype).reshape(candidate.shape)
