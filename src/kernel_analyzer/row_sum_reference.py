"""Audited FP32 row sum with BF16 write; no inference from fused kernel names."""
import ast
import hashlib

BODY = '''
xnumel = {rows}
r0_numel = {width}
R0_BLOCK: tl.constexpr = {block}
rnumel = r0_numel
RBLOCK: tl.constexpr = R0_BLOCK
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = xindex < xnumel
r0_index = tl.arange(0, R0_BLOCK)[None, :]
r0_offset = 0
r0_mask = r0_index < r0_numel
roffset = r0_offset
rindex = r0_index
r0_1 = r0_index
x0 = xindex
tmp0 = tl.load(in_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, other=0.0)
tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
tl.store(out_ptr0 + x0, tmp4, xmask)
'''


def check_source(source, symbol):
    assignments = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Ambiguous or missing definition')
    call = assignments[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'triton' and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton source required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Missing or ambiguous function')
    fn = functions[0]
    if ([a.arg for a in fn.args.args] != ['in_ptr0','out_ptr0','xnumel','r0_numel','XBLOCK']
            or fn.args.defaults or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected function signature')
    try:
        rows, width, block = [n.value.value for n in fn.body[:3]]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('Literal dimensions required') from exc
    if (any(type(v) is not int or v <= 0 for v in (rows,width,block))
            or block != 1 << (width-1).bit_length() or rows*width >= 2**31):
        raise ValueError('Invalid reduction dimensions')
    expected = ast.parse(BODY.format(rows=rows,width=width,block=block)).body
    if [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in expected]:
        raise ValueError('Reduction expression, addressing or padding differs')
    signatures = []
    for decorator in fn.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == 'triton_meta' and isinstance(keyword.value, ast.Dict):
                    for key,value in zip(keyword.value.keys,keyword.value.values):
                        if isinstance(key,ast.Constant) and key.value == 'signature':
                            signatures.append(ast.literal_eval(value))
    if signatures != [dict(in_ptr0='*fp32',out_ptr0='*bf16',xnumel='i32',r0_numel='i32')]:
        raise ValueError('Storage types differ')
    return dict(family='CONTIGUOUS_FP32_ROW_SUM_BF16_WRITE_V1',symbol=symbol,
                rows=rows,width=width,output_pointer='out_ptr0',
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                mathematical_expression='sum_j input[row,j], then BF16 store',
                proof_scope='One occurrence per element and zero padding; not a proof of numerical bias')


def reference(metadata, candidate, contract):
    import torch
    x = metadata.get('runtime_pointers',{}).get('in_ptr0')
    rows,width = contract['rows'],contract['width']
    if (metadata.get('input_output_storage_aliases') != []
            or not isinstance(x,torch.Tensor) or x.dtype != torch.float32
            or x.numel() != rows*width or not x.is_contiguous()
            or candidate.dtype != torch.bfloat16 or candidate.numel() != rows
            or not candidate.is_contiguous() or candidate.device != x.device):
        raise ValueError('Runtime storage contract differs')
    if not torch.isfinite(x).all():
        raise ValueError('Finite input required')
    return x.reshape(rows,width).sum(-1).to(candidate.dtype).reshape(candidate.shape)
