"""Reviewed Ministral attention position-scaling Triton source contract.

The generated kernel implements the model expression

    1 + beta * log(1 + floor(position / original_context_length))

and stores the result in BF16.  This checker establishes source eligibility;
it is not runtime evidence and does not claim that the result is biased.
"""

from __future__ import annotations

import ast
import hashlib


def _literal_triton_function(source: str, symbol: str) -> ast.FunctionDef:
    assignments = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == symbol
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Unique position-scaling source assignment required")
    call = assignments[0].value
    if (not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "triton"
            or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError("Literal Triton source required")
    functions = [
        node for node in ast.parse(call.args[1].value).body
        if isinstance(node, ast.FunctionDef) and node.name == symbol
    ]
    if len(functions) != 1:
        raise ValueError("Unique position-scaling function required")
    return functions[0]


def _expected_body(elements: int) -> list[ast.stmt]:
    body = f"""xnumel = {elements}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = xindex < xnumel
x0 = xindex
tmp0 = x0
tmp1 = tmp0.to(tl.float32)
tmp2 = tl.full([1], 6.103515625e-05, tl.float32)
tmp3 = tmp1 * tmp2
tmp4 = libdevice.floor(tmp3)
tmp5 = tl.full([1], 1.0, tl.float32)
tmp6 = tmp4 + tmp5
tmp7 = tl_math.log(tmp6)
tmp8 = tl.full([1], 0.1, tl.float32)
tmp9 = tmp7 * tmp8
tmp10 = tmp9 + tmp5
tmp11 = tmp10.to(tl.float32)
tl.store(out_ptr0 + (x0), tmp11, xmask)
"""
    return ast.parse(body).body


def _expected_fused_rotary_body() -> list[ast.stmt]:
    """Exact high-position Ministral query rotary/scaling kernel body.

    This is deliberately not a loose token-pattern match.  The emitted kernel
    fuses RoPE and the position-dependent query scaling, so accepting it as the
    old standalone scaling output would assign the wrong mathematical boundary.
    """
    return ast.parse("""xnumel = 524288
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = tl.full([XBLOCK], True, tl.int1)[:]
x0 = (xindex % 128)
x1 = ((xindex // 128) % 128)
x2 = xindex // 16384
x4 = xindex
tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 4096*x1), None).to(tl.float32)
tmp1 = tl.load(in_ptr1 + ((x4 % 64)), None, eviction_policy='evict_last')
tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
tmp3 = tmp2.to(tl.float32)
tmp4 = tmp1 * tmp3
tmp5 = tl_math.cos(tmp4)
tmp6 = tl.full([1], 1.0, tl.float32)
tmp7 = tmp5 * tmp6
tmp8 = tmp7.to(tl.float32)
tmp9 = tmp0 * tmp8
tmp10 = x0
tmp11 = tl.full([1], 0, tl.int64)
tmp12 = tmp10 >= tmp11
tmp13 = tl.full([1], 64, tl.int64)
tmp14 = tmp10 < tmp13
tmp15 = tl.load(in_ptr0 + (64 + 128*x2 + 4096*x1 + (x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
tmp16 = -tmp15
tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
tmp18 = tl.where(tmp14, tmp16, tmp17)
tmp19 = tmp10 >= tmp13
tmp20 = tl.full([1], 128, tl.int64)
tmp21 = tmp10 < tmp20
tmp22 = tl.load(in_ptr0 + (128*x2 + 4096*x1 + ((-64) + x0)), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
tmp23 = tl.where(tmp14, tmp18, tmp22)
tmp24 = tl_math.sin(tmp4)
tmp25 = tmp24 * tmp6
tmp26 = tmp25.to(tl.float32)
tmp27 = tmp23 * tmp26
tmp28 = tmp9 + tmp27
tmp29 = tl.full([1], 6.103515625e-05, tl.float32)
tmp30 = tmp3 * tmp29
tmp31 = libdevice.floor(tmp30)
tmp32 = tmp31 + tmp6
tmp33 = tl_math.log(tmp32)
tmp34 = tl.full([1], 0.1, tl.float32)
tmp35 = tmp33 * tmp34
tmp36 = tmp35 + tmp6
tmp37 = tmp36.to(tl.float32)
tmp38 = tmp28 * tmp37
tl.store(out_ptr0 + (x4), tmp38, None)
tl.store(out_ptr1 + (x0 + 128*x2 + 4096*x1), tmp38, None)
""").body


def _signatures(fn: ast.FunctionDef) -> list[dict]:
    found = []
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg != "triton_meta" or not isinstance(keyword.value, ast.Dict):
                continue
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if isinstance(key, ast.Constant) and key.value == "signature":
                    found.append(ast.literal_eval(value))
    return found


def check_source(source: str, symbol: str) -> dict:
    fn = _literal_triton_function(source, symbol)
    arguments = [argument.arg for argument in fn.args.args]
    fused = arguments == [
        "in_ptr0", "in_ptr1", "in_ptr2", "out_ptr0", "out_ptr1",
        "xnumel", "XBLOCK",
    ]
    standalone = arguments == ["out_ptr0", "xnumel", "XBLOCK"]
    if (not (standalone or fused)
            or fn.args.defaults or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg):
        raise ValueError("Position-scaling pointer arguments differ")
    first = fn.body[0] if fn.body else None
    if (not isinstance(first, ast.Assign)
            or len(first.targets) != 1
            or not isinstance(first.targets[0], ast.Name)
            or first.targets[0].id != "xnumel"
            or not isinstance(first.value, ast.Constant)
            or type(first.value.value) is not int
            or first.value.value <= 0):
        raise ValueError("Static positive position count required")
    elements = int(first.value.value)
    expected = _expected_fused_rotary_body() if fused else _expected_body(elements)
    if [ast.dump(node) for node in fn.body] != [ast.dump(node) for node in expected]:
        raise ValueError("Position-scaling arithmetic or addressing differs")
    signatures = _signatures(fn)
    if fused:
        allowed = [{
            "in_ptr0": "*bf16", "in_ptr1": "*fp32", "in_ptr2": "*i64",
            "out_ptr0": "*bf16", "out_ptr1": "*bf16", "xnumel": "i32",
        }]
    else:
        allowed = [
            {"out_ptr0": "*bf16", "xnumel": "i32"},
            {"out_ptr0": "*bf16", "xnumel": "i32", "XBLOCK": "constexpr"},
        ]
    if len(signatures) != 1 or signatures[0] not in allowed:
        raise ValueError("Position-scaling storage precision differs")

    semantic = ast.Module(body=[ast.FunctionDef(
        name=fn.name, args=fn.args, body=fn.body, decorator_list=[],
        returns=fn.returns, type_comment=fn.type_comment,
    )], type_ignores=[])
    return {
        "symbol": symbol,
        "elements": elements,
        "original_context_length": 16384,
        "beta": 0.1,
        "layout": "FUSED_ROTARY_QUERY_SCALING" if fused else "STANDALONE_SCALING",
        "output_pointer": "out_ptr0",
        "output_pointers": ["out_ptr0", "out_ptr1"] if fused else ["out_ptr0"],
        "output_dtype": "bfloat16",
        "function_ast_sha256": hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
        "function_semantic_ast_sha256": hashlib.sha256(
            ast.dump(semantic).encode()).hexdigest(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "reference": ("MINISTRAL_FUSED_ROTARY_POSITION_SCALING_FP32_BF16_WRITE"
                      if fused else "MINISTRAL_POSITION_ATTENTION_SCALING_FP32_BF16_WRITE"),
        "runtime_binding_complete": False,
    }
