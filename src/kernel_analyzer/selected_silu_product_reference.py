"""Same-input reference for a selected-time gradient times SiLU activation.

This is NOT the derivative of SiLU. Source validation proves the declared
real-arithmetic expression and addressing, not a nonzero bias or loss effect.
"""
import ast
import hashlib


BODY = '''
xnumel = {elements}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = xindex < xnumel
x0 = xindex
tmp2 = tl.load(in_ptr0 + ({step} + {length} * x0), xmask, eviction_policy='evict_last').to(tl.float32)
tmp3 = tl.load(in_ptr1 + ({offset} + x0), xmask)
tmp0 = tl.full([1], 0, tl.int32)
tmp1 = tmp0 == tmp0
tmp4 = -tmp3
tmp5 = libdevice.exp(tmp4)
tmp6 = tl.full([1], 1.0, tl.float32)
tmp7 = tmp5 + tmp6
tmp8 = tmp3 / tmp7
tmp9 = tmp8.to(tl.float32)
tmp10 = tmp2 * tmp9
tmp11 = tl.full([1], 0.0, tl.float32)
tmp12 = tl.where(tmp1, tmp10, tmp11)
tl.store(out_ptr0 + x0, tmp12, xmask)
'''


def integer(node):
    if not isinstance(node, ast.Constant) or type(node.value) is not int:
        raise ValueError('Expected literal integer')
    return node.value


def check_source(source, symbol):
    assignments = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Kernel definition absent or ambiguous')
    call = assignments[0].value
    if not (isinstance(call, ast.Call) and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
        raise ValueError('Literal kernel source required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Kernel function absent or ambiguous')
    fn = functions[0]
    if ([a.arg for a in fn.args.args] != ['in_ptr0', 'in_ptr1', 'out_ptr0', 'xnumel', 'XBLOCK']
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults):
        raise ValueError('Unexpected pointer signature')
    try:
        elements = integer(fn.body[0].value)
        # Extract literals only; the subsequent full AST comparison validates
        # every operator, mask, pointer, cast, and store, including these paths.
        address = fn.body[5].value.func.value.args[0].right
        zero_step = isinstance(address, ast.BinOp) and isinstance(address.op, ast.Mult)
        if zero_step:
            step, offset = 0, 0
            length = integer(address.left)
        else:
            step = integer(address.left)
            length = integer(address.right.left)
            offset = integer(fn.body[6].value.args[0].right.left)
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError('Unexpected selected-time addressing') from exc
    if not (0 < elements < 2**31 and 0 < length < 2**31 and 0 <= step < length
            and offset == step * elements and elements * length < 2**31):
        raise ValueError('Invalid dimensions or inconsistent time selection')
    expected_text = BODY.format(elements=elements, step=step, length=length, offset=offset)
    if zero_step:
        expected_text = expected_text.replace(f'(0 + {length} * x0)', f'{length} * x0')
        expected_text = expected_text.replace('(0 + x0)', 'x0')
    expected = ast.parse(expected_text).body
    if [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in expected]:
        raise ValueError('Selected SiLU product expression or addressing differs')
    return dict(family='SELECTED_SILU_PRODUCT_V1', symbol=symbol, elements=elements,
                sequence_length=length, selected_step=step, output_pointer='out_ptr0',
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                mathematical_expression='gradient[channel,time] * silu(activation[time,channel])',
                proof_scope='Real expression and selected-time addressing; not SiLU derivative or bias proof')


def physical_flat(value):
    """Read pointer order, including dense transposed saved activations.

    reshape(-1) alone would silently reorder a transposed tensor, unlike the
    original kernel's linear pointer loads. Reject holes and overlap first.
    """
    stride = 1
    for actual, size in sorted((s,n) for n,s in zip(value.shape,value.stride()) if n>1):
        if actual != stride:
            raise ValueError('Input layout is not dense nonoverlapping storage')
        stride *= size
    if value.numel() <= 0:
        raise ValueError('Empty input layout')
    return value.as_strided((value.numel(),),(1,))


def reference(metadata, candidate, contract):
    import torch
    if metadata.get('input_output_storage_aliases') != []:
        raise ValueError('Nonaliasing read inputs were not established')
    channels, length, step = (contract[k] for k in ('elements', 'sequence_length', 'selected_step'))
    pointers = metadata.get('runtime_pointers', {})
    values = []
    for name in ('in_ptr0', 'in_ptr1'):
        value = pointers.get(name)
        if (not isinstance(value, torch.Tensor)
                or value.numel() != channels * length or value.device != candidate.device
                or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError('Input layout or representation differs: ' + name)
        values.append(physical_flat(value).float())
    if (not candidate.is_contiguous() or candidate.numel() != channels
            or candidate.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('Output layout or representation differs')
    gradient = values[0].reshape(channels, length)[:, step]
    activation = values[1].reshape(length, channels)[step]
    return (gradient * torch.nn.functional.silu(activation)).to(candidate.dtype).reshape(candidate.shape)
