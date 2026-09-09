"""Strict reference for one generated RMS normalization forward family.

The reviewed kernel stores the per-row mean-square plus epsilon as FP32 and
writes the normalized BF16 tensor. The common registry binds only the latter;
the auxiliary statistic remains a distinct output boundary.
"""

import ast
import hashlib
import math

from kernel_analyzer.dense_pointer_view import dense_pointer_view


def _body(rows, width, epsilon):
    return ast.parse(f'''
xnumel = {rows}
r0_numel = {width}
R0_BLOCK: tl.constexpr = {width}
rnumel = r0_numel
RBLOCK: tl.constexpr = R0_BLOCK
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = xindex < xnumel
r0_index = tl.arange(0, R0_BLOCK)[None, :]
r0_offset = 0
r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
roffset = r0_offset
rindex = r0_index
r0_1 = r0_index
x0 = xindex
tmp0 = tl.load(in_ptr0 + (r0_1 + {width}*x0), xmask, other=0.0).to(tl.float32)
tmp1 = tmp0.to(tl.float32)
tmp2 = tmp1 * tmp1
tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
tmp5 = tl.where(xmask, tmp3, 0)
tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
tmp7 = tl.full([1, 1], {float(width)!r}, tl.float32)
tmp8 = tmp6 / tmp7
tmp9 = tl.full([1, 1], {epsilon!r}, tl.float32)
tmp10 = tmp8 + tmp9
tmp11 = tl.full([1, 1], -0.5, tl.float32)
tmp12 = libdevice.pow(tmp10, tmp11)
tmp13 = tmp1 * tmp12
tmp14 = tmp13.to(tl.float32)
tl.store(in_out_ptr0 + x0, tmp10, xmask)
tl.store(out_ptr0 + (r0_1 + {width}*x0), tmp14, xmask)
''').body


def _function(source, symbol):
    assignments = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == symbol
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Missing or ambiguous RMS forward source")
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "triton" or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError("Literal Triton RMS forward source required")
    functions = [
        node for node in ast.parse(call.args[1].value).body
        if isinstance(node, ast.FunctionDef) and node.name == symbol
    ]
    if len(functions) != 1:
        raise ValueError("Missing or ambiguous RMS forward function")
    return functions[0]


def _assigned(function, name):
    values = [
        node.value for node in function.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)
    ]
    if len(values) != 1:
        raise ValueError("Missing or ambiguous RMS forward value: " + name)
    return values[0]


def check_source(source, symbol):
    function = _function(source, symbol)
    expected_args = ["in_out_ptr0", "in_ptr0", "out_ptr0", "xnumel",
                     "r0_numel", "XBLOCK"]
    if ([arg.arg for arg in function.args.args] != expected_args
            or function.args.defaults or function.args.posonlyargs
            or function.args.kwonlyargs or function.args.vararg or function.args.kwarg):
        raise ValueError("RMS forward signature differs")
    try:
        rows = ast.literal_eval(_assigned(function, "xnumel"))
        width = ast.literal_eval(_assigned(function, "r0_numel"))
        epsilon_call = _assigned(function, "tmp9")
        epsilon = ast.literal_eval(epsilon_call.args[1])
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Literal RMS forward dimensions and epsilon required") from exc
    if (type(rows) is not int or type(width) is not int or min(rows, width) <= 0
            or type(epsilon) not in (int, float) or isinstance(epsilon, bool)
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("Invalid RMS forward dimensions or epsilon")
    if [ast.dump(node) for node in function.body] != [
            ast.dump(node) for node in _body(rows, width, epsilon)]:
        raise ValueError("RMS forward arithmetic, reduction, or addressing differs")
    signatures = []
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == "triton_meta" and isinstance(keyword.value, ast.Dict):
                for key, value in zip(keyword.value.keys, keyword.value.values):
                    if isinstance(key, ast.Constant) and key.value == "signature":
                        signatures.append(ast.literal_eval(value))
    expected_signature = dict(in_out_ptr0="*fp32", in_ptr0="*bf16",
                              out_ptr0="*bf16", xnumel="i32", r0_numel="i32")
    if signatures != [expected_signature]:
        raise ValueError("RMS forward storage types differ")
    return dict(
        family="RMS_FORWARD_NORMALIZED_V1", symbol=symbol, rows=rows, width=width,
        epsilon=epsilon, output_pointer="out_ptr0",
        reference_variant="FP32_ROW_RMS_BF16_WRITE",
        mathematical_expression="input / sqrt(mean(input**2) + epsilon)",
        function_ast_sha256=hashlib.sha256(ast.dump(function).encode()).hexdigest(),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_binding_complete=False,
    )


def reference(metadata, candidate, contract):
    import torch

    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("RMS forward symbol differs")
    if metadata.get("formal_pointer") != "out_ptr0":
        raise ValueError("RMS forward normalized-output boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("RMS forward input/output alias is unsupported")
    pointers = metadata.get("runtime_pointers") or {}
    if set(pointers) != {"in_out_ptr0", "in_ptr0", "out_ptr0"}:
        raise ValueError("Unexpected RMS forward pointer set")
    rows, width = contract["rows"], contract["width"]
    source_value = pointers["in_ptr0"]
    statistic = pointers["in_out_ptr0"]
    output = pointers["out_ptr0"]
    expected = (("in_ptr0", source_value, torch.bfloat16, rows * width),
                ("in_out_ptr0", statistic, torch.float32, rows),
                ("out_ptr0", output, torch.bfloat16, rows * width))
    for name, value, dtype, size in expected:
        if (not isinstance(value, torch.Tensor) or value.dtype != dtype
                or value.device != candidate.device or value.numel() != size):
            raise ValueError("RMS forward pointer differs: " + name)
    if not torch.isfinite(source_value).all():
        raise ValueError("Nonfinite RMS forward input")
    if (candidate.dtype != torch.bfloat16 or candidate.numel() != rows * width
            or not candidate.is_contiguous()):
        raise ValueError("RMS forward candidate representation differs")
    x = dense_pointer_view(source_value, (rows, width)).detach().float().clone()
    mean_square = x.square().sum(dim=1, keepdim=True) / float(width)
    result = x * torch.pow(mean_square + contract["epsilon"], -0.5)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite RMS forward reference arithmetic")
    return result.to(torch.bfloat16).reshape(candidate.shape)
