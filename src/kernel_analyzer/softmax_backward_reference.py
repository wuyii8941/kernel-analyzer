"""Same-input reference for a checked fused softmax backward template.

The saved row maximum and denominator are inputs, not recomputed from a
different forward graph. This identifies one local implementation comparison,
not a proof of nonzero rounding bias.
"""
import ast
import hashlib
import math


BODY = '''
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
tmp2 = tl.load(in_out_ptr0 + (r0_1 + {width}*x0), xmask, other=0.0).to(tl.float32)
tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
tmp1 = tmp0.to(tl.float32)
tmp3 = tmp2.to(tl.float32)
tmp5 = tmp3 - tmp4
tmp6 = libdevice.exp(tmp5)
tmp8 = (tmp6 / tmp7)
tmp9 = tmp1 * tmp8
tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
tmp12 = tl.where(xmask, tmp10, 0)
tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
tmp14 = -tmp8
tmp15 = tl.fma(tmp14, tmp13, tmp9)
tmp16 = tmp15.to(tl.float32)
tmp17 = tl.full([1, 1], {scale}, tl.float32)
tmp18 = tmp16 * tmp17
tl.store(in_out_ptr0 + (r0_1 + {width}*x0), tmp18, xmask)
'''


def check_source(source, symbol):
    matches = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
            call = node.value
            if not (isinstance(call, ast.Call) and len(call.args) >= 2 and isinstance(call.args[1], ast.Constant)
                    and isinstance(call.args[1].value, str)):
                raise ValueError('Expected literal kernel definition')
            matches.extend(n for n in ast.parse(call.args[1].value).body if isinstance(n, ast.FunctionDef) and n.name == symbol)
    if len(matches) != 1:
        raise ValueError('Kernel definition absent or ambiguous')
    fn = matches[0]
    dimensions = []
    for node, name in zip(fn.body[:2], ('xnumel', 'r0_numel')):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name and isinstance(node.value, ast.Constant)
                and type(node.value.value) is int and node.value.value > 0):
            raise ValueError('Unknown row dimensions')
        dimensions.append(node.value.value)
    if len(dimensions) != 2:
        raise ValueError('Missing dimensions')
    rows, width = dimensions
    if width & (width - 1):
        raise ValueError('Unmasked reduction requires power-of-two width')
    scales = [node.value.args[1].value for node in fn.body
              if isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'tmp17'
              and isinstance(node.value, ast.Call) and len(node.value.args) >= 2
              and isinstance(node.value.args[1], ast.Constant)]
    if len(scales) != 1 or type(scales[0]) not in (int, float) or not math.isfinite(scales[0]) or scales[0] <= 0:
        raise ValueError('Invalid declared backward scale')
    scale = scales[0]
    expected = ast.parse(BODY.format(rows=rows, width=width, scale=repr(scale))).body
    if [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in expected]:
        raise ValueError('Softmax backward expression or indexing differs')
    return {'family': 'SAVED_NORMALIZER_SOFTMAX_BACKWARD_V1', 'symbol': symbol,
            'rows': rows, 'width': width, 'scale': scale, 'output_pointer': 'in_out_ptr0',
            'function_ast_sha256': hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'mathematical_expression': 'scale * (g*p - p*sum(g*p)); p=exp(scores-saved_max)/saved_denominator',
            'proof_scope': 'Real formula with identical saved normalizers; not nonzero-bias or loss proof'}


def evaluate(gradient, scores, saved_max, saved_denominator, scale):
    probability = (scores - saved_max).exp() / saved_denominator
    product = gradient * probability
    return (product - probability * product.sum(-1, keepdim=True)) * scale


def reference(metadata, candidate, contract):
    import torch
    if metadata.get('input_output_storage_aliases') != []:
        raise ValueError('Nonaliasing read inputs were not established')
    rows, width = contract['rows'], contract['width']
    pointers = metadata.get('runtime_pointers', {})
    values = []
    for name, size in (('in_ptr0', rows * width), ('in_out_ptr0', rows * width),
                       ('in_ptr1', rows), ('in_ptr2', rows)):
        value = pointers.get(name)
        if not isinstance(value, torch.Tensor) or not value.is_contiguous() or value.numel() != size:
            raise ValueError('Runtime input layout differs: ' + name)
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Unsupported representation: ' + name)
        if name in ('in_ptr1', 'in_ptr2') and value.dtype != torch.float32:
            raise ValueError('Saved normalizers must be FP32')
        values.append(value.float())
    if candidate.dtype not in (torch.float16, torch.bfloat16, torch.float32) or candidate.numel() != rows * width or not candidate.is_contiguous():
        raise ValueError('Runtime output layout differs')
    gradient, scores, maximum, denominator = values
    if not bool(torch.isfinite(maximum).all()) or not bool((torch.isfinite(denominator) & (denominator > 0)).all()):
        raise ValueError('Undefined saved softmax normalizers')
    # The literal coefficient is a FP32 constant in the actual kernel.
    scale = torch.tensor(contract['scale'], dtype=torch.float32, device=scores.device)
    return evaluate(gradient.reshape(rows, width), scores.reshape(rows, width),
                    maximum.reshape(rows, 1), denominator.reshape(rows, 1), scale).to(candidate.dtype).reshape(candidate.shape)
