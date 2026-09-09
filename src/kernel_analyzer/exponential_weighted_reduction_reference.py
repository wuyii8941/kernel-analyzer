"""Reviewed reference for one Inductor exponential weighted reduction.

The supported Triton template computes, for every output row ``x``,

    sum_j source[x, j] * -exp(log_weight[x // rows_per_group, j]).

The checker accepts only the complete, audited function body.  It is therefore
an adapter for this mathematical family, not a name-based guess about arbitrary
generated kernels.
"""
from __future__ import annotations

import ast
import hashlib


BODY = '''
xnumel = {rows}
r0_numel = {width}
R0_BLOCK: tl.constexpr = {width}
rnumel = r0_numel
RBLOCK: tl.constexpr = R0_BLOCK
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
r0_index = tl.arange(0, R0_BLOCK)[None, :]
r0_offset = 0
r0_mask = r0_index < r0_numel
roffset = r0_offset
rindex = r0_index
r0_2 = r0_index
x3 = xindex
x1 = xindex // {rows_per_group}
tmp0 = tl.load(in_ptr0 + (r0_2 + {width}*x3), r0_mask, other=0.0)
tmp1 = tl.load(in_ptr1 + (r0_2 + {width}*x1), r0_mask, eviction_policy='evict_last', other=0.0)
tmp2 = libdevice.exp(tmp1)
tmp3 = -tmp2
tmp4 = tmp0 * tmp3
tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
tmp7 = tl.where(r0_mask, tmp5, 0)
tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
tl.store(out_ptr0 + (x3), tmp8, None)
'''

VARIANTS = ('FP32_NATIVE', 'FP32_REVERSE_COMPONENT_ORDER', 'FP64_EVALUATION')


def _embedded_function(source: str, symbol: str) -> ast.FunctionDef:
    matches = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not any(
                isinstance(target, ast.Name) and target.id == symbol for target in node.targets):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and len(call.args) >= 2
                and isinstance(call.args[1], ast.Constant)
                and isinstance(call.args[1].value, str)):
            raise ValueError('Kernel source is not an embedded literal')
        matches.extend(item for item in ast.parse(call.args[1].value).body
                       if isinstance(item, ast.FunctionDef) and item.name == symbol)
    if len(matches) != 1:
        raise ValueError('Kernel identity absent or ambiguous')
    return matches[0]


def _positive_integer_assignment(node: ast.stmt, name: str) -> int:
    if not (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name
            and isinstance(node.value, ast.Constant) and type(node.value.value) is int
            and node.value.value > 0):
        raise ValueError('Unknown exponential reduction dimensions')
    return node.value.value


def check_source(source: str, symbol: str) -> dict:
    fn = _embedded_function(source, symbol)
    if len(fn.body) < 17:
        raise ValueError('Incomplete exponential weighted reduction')
    rows = _positive_integer_assignment(fn.body[0], 'xnumel')
    width = _positive_integer_assignment(fn.body[1], 'r0_numel')

    group_assignment = fn.body[15]
    if not (isinstance(group_assignment, ast.Assign)
            and len(group_assignment.targets) == 1
            and isinstance(group_assignment.targets[0], ast.Name)
            and group_assignment.targets[0].id == 'x1'
            and isinstance(group_assignment.value, ast.BinOp)
            and isinstance(group_assignment.value.op, ast.FloorDiv)
            and isinstance(group_assignment.value.left, ast.Name)
            and group_assignment.value.left.id == 'xindex'
            and isinstance(group_assignment.value.right, ast.Constant)
            and type(group_assignment.value.right.value) is int
            and group_assignment.value.right.value > 0):
        raise ValueError('Unknown exponential weight grouping')
    rows_per_group = group_assignment.value.right.value
    if rows % rows_per_group:
        raise ValueError('Output rows do not divide into complete groups')

    expected = ast.parse(BODY.format(
        rows=rows, width=width, rows_per_group=rows_per_group)).body
    if len(fn.body) != len(expected) or [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in expected]:
        raise ValueError('Exponential weighted reduction expression or indexing differs')
    # The generated decorator contains the logical CUDA device index.  That
    # index may legitimately differ when the same audited graph is replayed on
    # another GPU, so it is not part of this family's mathematical identity.
    # Keep the complete function hash for provenance, but use arguments plus
    # the already fully checked body as the stable runtime identity.
    semantic_tree = ast.Module(body=[ast.FunctionDef(
        name=fn.name,
        args=fn.args,
        body=fn.body,
        decorator_list=[],
        returns=fn.returns,
        type_comment=fn.type_comment,
    )], type_ignores=[])
    return {
        'family': 'EXPONENTIAL_WEIGHTED_REDUCTION_V1',
        'symbol': symbol,
        'rows': rows,
        'width': width,
        'rows_per_group': rows_per_group,
        'group_count': rows // rows_per_group,
        'input_pointer': 'in_ptr0',
        'log_weight_pointer': 'in_ptr1',
        'output_pointer': 'out_ptr0',
        'function_ast_sha256': hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
        'function_semantic_ast_sha256': hashlib.sha256(
            ast.dump(semantic_tree).encode()).hexdigest(),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'mathematical_expression': 'sum_j source[x,j] * -exp(log_weight[x//rows_per_group,j])',
        'proof_scope': (
            'Complete audited FP32 Triton body and addresses; establishes the real-valued '
            'expression and common-input reference, not zero numerical bias'),
        'legal_variants': list(VARIANTS),
    }


def reference(metadata, candidate, contract, *, variant='FP32_NATIVE'):
    import torch

    if variant not in VARIANTS:
        raise ValueError('Undeclared exponential weighted reduction variant')
    pointers = metadata.get('runtime_pointers', {})
    source = pointers.get(contract['input_pointer'])
    log_weight = pointers.get(contract['log_weight_pointer'])
    rows = contract['rows']
    width = contract['width']
    group_count = contract['group_count']
    rows_per_group = contract['rows_per_group']
    if (not isinstance(source, torch.Tensor) or source.dtype != torch.float32
            or source.numel() != rows * width or not source.is_contiguous()):
        raise ValueError('Source input does not match checked FP32 row addresses')
    if (not isinstance(log_weight, torch.Tensor) or log_weight.dtype != torch.float32
            or log_weight.numel() != group_count * width or not log_weight.is_contiguous()):
        raise ValueError('Log-weight input does not match checked FP32 grouped addresses')
    if candidate.dtype != torch.float32 or candidate.numel() != rows or not candidate.is_contiguous():
        raise ValueError('Candidate output does not match checked FP32 row addresses')

    x = source.reshape(rows, width)
    group = torch.arange(rows, device=source.device).div(rows_per_group, rounding_mode='floor')
    w = log_weight.reshape(group_count, width).index_select(0, group)
    if variant == 'FP64_EVALUATION':
        result = (x.double() * -w.double().exp()).sum(-1).float()
    else:
        terms = x * -w.exp()
        if variant == 'FP32_REVERSE_COMPONENT_ORDER':
            terms = terms.flip(-1)
        result = terms.sum(-1)
    return result.reshape(candidate.shape)
