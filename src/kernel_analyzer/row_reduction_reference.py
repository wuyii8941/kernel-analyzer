"""Source-checked family reference for Inductor contiguous row square sums.

This recognizes a bounded code template, not arbitrary Triton semantics. The
identity concerns the real arithmetic expression; floating-point bias is an
experimental question. No case names, model names or observed effects enter.
"""
from __future__ import annotations

import ast
import hashlib


PREFIX = '''
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
_tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = _tmp4 + tmp3
    _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
tmp4 = tl.sum(_tmp4, 1)[:, None]
tl.store(out_ptr0 + (x0), tmp4, xmask)
'''

VARIANTS = ('FP32_NATIVE', 'FP32_REVERSE_FEATURE_ORDER', 'FP64_EVALUATION')


def check_source(source: str, symbol: str) -> dict:
    matches = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
            raise ValueError('Kernel source is not an embedded literal')
        matches.extend(n for n in ast.parse(call.args[1].value).body
                       if isinstance(n, ast.FunctionDef) and n.name == symbol)
    if len(matches) != 1:
        raise ValueError('Kernel identity absent or ambiguous')
    fn = matches[0]
    dimensions = []
    for node, name in zip(fn.body[:2], ('xnumel', 'r0_numel')):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
                and isinstance(node.value, ast.Constant) and type(node.value.value) is int
                and node.value.value > 0):
            raise ValueError('Unknown row reduction dimensions')
        dimensions.append(node.value.value)
    if len(dimensions) != 2:
        raise ValueError('Missing dimensions')
    rows, width = dimensions
    expected = ast.parse(PREFIX.format(rows=rows, width=width)).body
    if [ast.dump(n) for n in fn.body[:len(expected)]] != [ast.dump(n) for n in expected]:
        raise ValueError('Row square-sum expression or indexing differs')
    if any(isinstance(n, ast.Name) and n.id == 'out_ptr0'
           for later in fn.body[len(expected):] for n in ast.walk(later)):
        raise ValueError('Output is used after the recognized store')
    return {'family': 'CONTIGUOUS_ROW_SQUARE_SUM_V1', 'symbol': symbol,
            'rows': rows, 'width': width, 'input_pointer': 'in_ptr0', 'output_pointer': 'out_ptr0',
            'function_ast_sha256': hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'mathematical_expression': 'sum_j float32(input[row,j])**2',
            'proof_scope': 'Each unmasked feature occurs once; disjoint row addresses; zero masked padding; not a nonzero-bias proof',
            'legal_variants': list(VARIANTS)}


def reference(metadata, candidate, contract, *, variant='FP32_NATIVE'):
    import torch
    if variant not in VARIANTS:
        raise ValueError('Undeclared row reduction variant')
    source = metadata.get('runtime_pointers', {}).get('in_ptr0')
    rows, width = contract['rows'], contract['width']
    if (not isinstance(source, torch.Tensor) or source.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or source.numel() != rows * width or not source.is_contiguous()):
        raise ValueError('Input does not match checked row addresses')
    if candidate.dtype != torch.float32 or candidate.numel() != rows or not candidate.is_contiguous():
        raise ValueError('Output does not match checked row addresses')
    x = source.reshape(rows, width)
    if variant == 'FP64_EVALUATION':
        # Input load conversion remains FP32 as declared by the candidate.
        result = x.float().double().square().sum(-1).float()
    else:
        x = x.float()
        if variant == 'FP32_REVERSE_FEATURE_ORDER':
            x = x.flip(-1)
        result = x.square().sum(-1)
    return result.reshape(candidate.shape)
