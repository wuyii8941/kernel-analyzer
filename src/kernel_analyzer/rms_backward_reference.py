"""Common-input reference for a source-checked fused RMSNorm backward family.

The formula includes an incoming gradient accumulator. It establishes the
real-arithmetic mapping, not a proof that its rounding error has nonzero mean.
"""
import ast
import hashlib


BODY = '''
xnumel = {rows}
r0_numel = {width}
rnumel = r0_numel
RBLOCK: tl.constexpr = R0_BLOCK
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = xindex < xnumel
r0_base = tl.arange(0, R0_BLOCK)[None, :]
rbase = r0_base
x0 = xindex
_tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = _tmp10 + tmp9
    _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
tmp10 = tl.sum(_tmp10, 1)[:, None]
tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp12 = tl.load(in_out_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.load(in_ptr3 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tl.full([1, 1], -0.5, tl.float32)
    tmp22 = tmp10 * tmp21
    tmp23 = tmp19 * tmp19
    tmp24 = tmp23 * tmp19
    tmp25 = tmp22 * tmp24
    tmp26 = tl.full([1, 1], {inverse_width}, tl.float32)
    tmp27 = tmp25 * tmp26
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.full([1, 1], 2.0, tl.float32)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp27 * tmp31
    tmp33 = tmp20 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp12 + tmp34
    tl.store(in_out_ptr0 + (r0_1 + {width}*x0), tmp35, r0_mask & xmask)
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
    # The recognized coefficient is exactly reciprocal width in binary FP32.
    if width & (width - 1) or width > 2**24:
        raise ValueError('Width outside exact coefficient contract')
    expected = ast.parse(BODY.format(rows=rows, width=width, inverse_width=1. / width)).body
    if [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in expected]:
        raise ValueError('Fused RMS backward expression or indexing differs')
    return {'family': 'FUSED_RMS_BACKWARD_ACCUMULATE_V1', 'symbol': symbol,
            'rows': rows, 'width': width, 'output_pointer': 'in_out_ptr0',
            'function_ast_sha256': hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'mathematical_expression': 'a + g*r - x*sum(g*x)*r**3/width; g=(p0+p1)*w',
            'proof_scope': 'Real arithmetic under declared row layout and nonaliasing read inputs; not nonzero mean or training loss proof'}


def evaluate(a, p0, p1, w, x, inverse_rms):
    """Real formula evaluated with the supplied tensor dtype (also testable in FP64)."""
    g = (p0 + p1) * w
    inner = (g * x).sum(-1, keepdim=True)
    # Preserve the reference's explicit operation grouping instead of hiding
    # which rounding choices differ from a fused Triton implementation.
    correction = ((inner * -.5) * ((inverse_rms * inverse_rms) * inverse_rms)) * (1. / x.shape[-1])
    return a + (g * inverse_rms + correction * (x * 2.))


def reference(metadata, candidate, contract):
    import torch
    if metadata.get('input_output_storage_aliases') != []:
        raise ValueError('Nonaliasing read inputs were not established before invocation')
    rows, width = contract['rows'], contract['width']
    pointers = metadata.get('runtime_pointers', {})
    values = []
    for name, size in (('in_out_ptr0', rows * width), ('in_ptr0', rows * width),
                       ('in_ptr1', rows * width), ('in_ptr2', width),
                       ('in_ptr3', rows * width), ('in_ptr4', rows)):
        value = pointers.get(name)
        if not isinstance(value, torch.Tensor) or not value.is_contiguous() or value.numel() != size:
            raise ValueError('Runtime input layout differs: ' + name)
        if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('Unsupported input representation: ' + name)
        values.append(value.float())
    if pointers['in_ptr4'].dtype != torch.float32:
        raise ValueError('Saved inverse RMS must be FP32 for this contract')
    if candidate.dtype not in (torch.float16, torch.bfloat16, torch.float32) or candidate.numel() != rows * width or not candidate.is_contiguous():
        raise ValueError('Runtime output layout differs')
    a, p0, p1, w, x, r = values
    return evaluate(a.reshape(rows, width), p0.reshape(rows, width), p1.reshape(rows, width),
                    w.reshape(1, width), x.reshape(rows, width), r.reshape(rows, 1)).to(candidate.dtype).reshape(candidate.shape)
