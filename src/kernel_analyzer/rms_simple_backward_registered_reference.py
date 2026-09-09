"""Strict reference for a generated in-place RMS backward family."""
import ast
import hashlib
from kernel_analyzer.dense_pointer_view import dense_pointer_view


def _body(rows, width):
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
tmp0 = tl.load(in_out_ptr0 + (r0_1 + {width}*x0), xmask, other=0.0).to(tl.float32)
tmp2 = tl.load(in_ptr0 + (r0_1 + {width}*x0), xmask, other=0.0).to(tl.float32)
tmp9 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
tmp1 = tmp0.to(tl.float32)
tmp3 = tmp2.to(tl.float32)
tmp4 = tmp1 * tmp3
tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
tmp7 = tl.where(xmask, tmp5, 0)
tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
tmp10 = tl.full([1, 1], -0.5, tl.float32)
tmp11 = libdevice.pow(tmp9, tmp10)
tmp12 = tmp1 * tmp11
tmp13 = tl.full([1, 1], -1.5, tl.float32)
tmp14 = libdevice.pow(tmp9, tmp13)
tmp15 = tmp14 * tmp10
tmp16 = tmp8 * tmp15
tmp17 = tl.full([1, 1], {1.0 / width!r}, tl.float32)
tmp18 = tmp16 * tmp17
tmp19 = tl.full([1, 1], 2.0, tl.float32)
tmp20 = tmp3 * tmp19
tmp21 = tmp18 * tmp20
tmp22 = tmp12 + tmp21
tmp23 = tmp22.to(tl.float32)
tl.store(in_out_ptr0 + (r0_1 + {width}*x0), tmp23, xmask)
''').body


def _function(source, symbol):
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Missing or ambiguous simple RMS backward source')
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != 'triton' or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton simple RMS backward source required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Missing or ambiguous simple RMS backward function')
    return functions[0]


def check_source(source, symbol):
    function = _function(source, symbol)
    expected_args = ['in_out_ptr0', 'in_ptr0', 'in_ptr1', 'xnumel',
                     'r0_numel', 'XBLOCK']
    if ([a.arg for a in function.args.args] != expected_args or function.args.defaults
            or function.args.posonlyargs or function.args.kwonlyargs
            or function.args.vararg or function.args.kwarg):
        raise ValueError('Simple RMS backward signature differs')
    try:
        rows = ast.literal_eval(function.body[0].value)
        width = ast.literal_eval(function.body[1].value)
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError('Literal simple RMS backward dimensions required') from exc
    if (type(rows) is not int or type(width) is not int or min(rows, width) <= 0
            or width & (width - 1) or width > 2**24):
        raise ValueError('Invalid simple RMS backward dimensions')
    if [ast.dump(n) for n in function.body] != [ast.dump(n) for n in _body(rows, width)]:
        raise ValueError('Simple RMS backward arithmetic, reduction, or addressing differs')
    signatures = []
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == 'triton_meta' and isinstance(keyword.value, ast.Dict):
                    for key, value in zip(keyword.value.keys, keyword.value.values):
                        if isinstance(key, ast.Constant) and key.value == 'signature':
                            signatures.append(ast.literal_eval(value))
    expected = dict(in_out_ptr0='*bf16', in_ptr0='*bf16', in_ptr1='*fp32',
                    xnumel='i32', r0_numel='i32')
    if signatures != [expected]:
        raise ValueError('Simple RMS backward storage types differ')
    return dict(family='RMS_SIMPLE_BACKWARD_V1', symbol=symbol, rows=rows, width=width,
                output_pointer='in_out_ptr0', reference_variant='FP32_NATIVE_BF16_WRITE',
                mathematical_expression=('g*s**-0.5 - x*sum(g*x)*s**-1.5/width; '
                                         's is saved mean-square plus epsilon'),
                function_ast_sha256=hashlib.sha256(ast.dump(function).encode()).hexdigest(),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                runtime_binding_complete=False)


def reference(metadata, candidate, contract):
    import torch
    if metadata.get('symbol') != contract.get('symbol'):
        raise ValueError('Simple RMS backward symbol differs')
    if metadata.get('formal_pointer') != 'in_out_ptr0':
        raise ValueError('Simple RMS backward output boundary differs')
    if metadata.get('input_output_storage_aliases'):
        raise ValueError('Unexpected simple RMS backward argument alias')
    pointers = metadata.get('runtime_pointers') or {}
    if set(pointers) != {'in_out_ptr0', 'in_ptr0', 'in_ptr1'}:
        raise ValueError('Unexpected simple RMS backward pointer set')
    rows, width = contract['rows'], contract['width']
    gradient, value, saved = (pointers[n] for n in ('in_out_ptr0', 'in_ptr0', 'in_ptr1'))
    expected = (('in_out_ptr0', gradient, torch.bfloat16, rows * width),
                ('in_ptr0', value, torch.bfloat16, rows * width),
                ('in_ptr1', saved, torch.float32, rows))
    for name, tensor, dtype, size in expected:
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype
                or tensor.device != candidate.device or tensor.numel() != size
                or not torch.isfinite(tensor).all()):
            raise ValueError('Simple RMS backward pointer differs: ' + name)
    if (candidate.dtype != torch.bfloat16 or candidate.numel() != rows * width
            or not candidate.is_contiguous()):
        raise ValueError('Simple RMS backward candidate representation differs')
    g = dense_pointer_view(gradient, (rows, width)).detach().float().clone()
    x = dense_pointer_view(value, (rows, width)).detach().float().clone()
    s = dense_pointer_view(saved, (rows, 1)).detach().float().clone()
    inner = (g * x).sum(dim=1, keepdim=True)
    result = g * torch.pow(s, -0.5)
    result = result + ((inner * -0.5) * torch.pow(s, -1.5) / width) * (x * 2.0)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite simple RMS backward reference arithmetic')
    return result.to(torch.bfloat16).reshape(candidate.shape)
