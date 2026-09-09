"""Audited common-input softplus bias-gradient reduction, not a bias theorem.

This family is distinct from SiLU. For z[t,c]=input[t,c]+bias[c], it computes
sum_t (g1[c,t]+g2[c,t])*softplus'(z[t,c]), with beta=1 and threshold=20.
The independent FP32 reference uses sigmoid for the derivative. It does not
recompute earlier forward states or change the parameter/optimizer protocol.
"""
import ast
import hashlib


BODY = '''
xnumel = {channels}
r0_numel = {steps}
R0_BLOCK: tl.constexpr = {steps}
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
tmp0 = tl.load(in_ptr0 + (x0 + {channels} * r0_1), xmask, other=0.0).to(tl.float32)
tmp1 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last').to(tl.float32)
tmp8 = tl.load(in_ptr2 + (r0_1 + {steps} * x0), xmask, other=0.0)
tmp10 = tl.load(in_ptr3 + (r0_1 + {steps} * x0), xmask, other=0.0)
tmp2 = tmp0 + tmp1
tmp3 = tmp2.to(tl.float32)
tmp4 = tl.full([1, 1], 1.0, tl.float32)
tmp5 = tmp3 * tmp4
tmp6 = tl.full([1, 1], 20.0, tl.float32)
tmp7 = tmp5 > tmp6
tmp9 = tmp8.to(tl.float32)
tmp11 = tmp10.to(tl.float32)
tmp12 = tmp9 + tmp11
tmp13 = tmp12.to(tl.float32)
tmp14 = libdevice.exp(tmp5)
tmp15 = tmp13 * tmp14
tmp16 = tmp14 + tmp4
tmp17 = tmp15 / tmp16
tmp18 = tl.where(tmp7, tmp13, tmp17)
tmp19 = tmp18.to(tl.float32)
tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
tmp22 = tl.where(xmask, tmp20, 0)
tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
'''
EXTRA_STORE = 'tl.store(out_ptr1 + (x0 + {channels} * r0_1), tmp19, xmask)\n'
FINAL_STORE = 'tl.store(out_ptr0 + x0, tmp23, xmask)\n'


def check_source(source, symbol):
    assignments = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Kernel definition absent or ambiguous')
    call = assignments[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'triton' and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton definition required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Kernel function absent or ambiguous')
    fn = functions[0]
    base = ['in_ptr0', 'in_ptr1', 'in_ptr2', 'in_ptr3', 'out_ptr0']
    names = [a.arg for a in fn.args.args]
    extra = names == base + ['out_ptr1', 'xnumel', 'r0_numel', 'XBLOCK']
    if (names != base + ['xnumel', 'r0_numel', 'XBLOCK'] and not extra
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.defaults
            or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected softplus reduction signature')
    try:
        channels, steps = [n.value.value for n in fn.body[:2]]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError('Missing literal dimensions') from exc
    if (type(channels) is not int or type(steps) is not int or channels <= 0
            or steps <= 0 or steps & (steps-1) or channels*steps >= 2**31):
        raise ValueError('Invalid dimensions or unmasked reduction padding')
    expected = (BODY + (EXTRA_STORE if extra else '') + FINAL_STORE).format(
        channels=channels, steps=steps)
    if [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in ast.parse(expected).body]:
        raise ValueError('Softplus derivative, reduction or address expression differs')
    signatures = []
    for decorator in fn.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == 'triton_meta' and isinstance(keyword.value, ast.Dict):
                    for key, value in zip(keyword.value.keys, keyword.value.values):
                        if isinstance(key, ast.Constant) and key.value == 'signature':
                            try:
                                signatures.append(ast.literal_eval(value))
                            except (ValueError, TypeError) as exc:
                                raise ValueError('Literal pointer types required') from exc
    expected_types = {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32',
                      'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}
    if extra:
        expected_types['out_ptr1'] = '*bf16'
    if signatures != [expected_types]:
        raise ValueError('Softplus storage/compute types differ')
    return dict(family='SOFTPLUS_BIAS_GRADIENT_REDUCTION_V1', symbol=symbol,
        channels=channels, steps=steps, output_pointer='out_ptr0', extra_output_present=extra,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
        mathematical_expression='sum_t (g1[c,t]+g2[c,t])*where(x[t,c]+b[c]>20,1,sigmoid(x[t,c]+b[c]))',
        reference_numeric_choice='FP32 sigmoid derivative and sum, then BF16 output; saved inputs unchanged',
        proof_scope='Real softplus bias VJP with beta=1, threshold=20; not a nonzero-bias or loss proof',
        unmeasured_outputs=['out_ptr1'] if extra else [])


def evaluate(x, bias, g1, g2):
    import torch
    z = x.T + bias[:, None]
    return ((g1+g2)*torch.where(z > 20, torch.ones_like(z), torch.sigmoid(z))).sum(-1)


def reference(metadata, candidate, contract):
    import torch
    if metadata.get('input_output_storage_aliases') != []:
        raise ValueError('Nonaliasing inputs were not established')
    channels, steps = contract['channels'], contract['steps']
    if (candidate.dtype != torch.bfloat16 or candidate.numel() != channels
            or not candidate.is_contiguous()):
        raise ValueError('Output layout or dtype differs')
    values = []
    for name, size, dtype in (('in_ptr0', steps*channels, torch.bfloat16),
                              ('in_ptr1', channels, torch.bfloat16),
                              ('in_ptr2', steps*channels, torch.float32),
                              ('in_ptr3', steps*channels, torch.float32)):
        value = metadata.get('runtime_pointers', {}).get(name)
        if (not isinstance(value, torch.Tensor) or value.numel() != size
                or not value.is_contiguous() or value.device != candidate.device or value.dtype != dtype):
            raise ValueError('Input layout or dtype differs: '+name)
        if not torch.isfinite(value).all():
            raise ValueError('Finite-input reference contract violated: '+name)
        values.append(value.float())
    x, bias, g1, g2 = values
    return evaluate(x.reshape(steps, channels), bias.reshape(channels),
                    g1.reshape(channels, steps), g2.reshape(channels, steps)).to(candidate.dtype).reshape(candidate.shape)
