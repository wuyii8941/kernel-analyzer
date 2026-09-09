"""Checked gated-SiLU backward, replacing only the gate-gradient output.

The other output of the fused kernel remains the candidate output. This is a
local-output substitution, not replacement of the whole fused computation.
"""
import ast
import hashlib


BODY = '''
xnumel = {elements}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = tl.full([XBLOCK], True, tl.int1)[:]
x0 = xindex
tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
tmp10 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
tmp2 = tmp1.to(tl.float32)
tmp3 = -tmp2
tmp4 = libdevice.exp(tmp3)
tmp5 = tl.full([1], 1.0, tl.float32)
tmp6 = tmp4 + tmp5
tmp7 = (tmp2 / tmp6)
tmp8 = tmp7.to(tl.float32)
tmp9 = tmp0 * tmp8
tmp11 = tmp0 * tmp10
tmp12 = tmp11.to(tl.float32)
tmp13 = (tmp5 / tmp6)
tmp14 = tmp13 * tmp5
tmp15 = tmp12 * tmp14
tmp16 = tmp5 - tmp14
tmp17 = tmp2 * tmp16
tmp18 = tmp17 + tmp5
tmp19 = tmp15 * tmp18
tmp20 = tmp19.to(tl.float32)
tl.store(out_ptr0 + (x0), tmp9, None)
tl.store(in_out_ptr0 + (x0), tmp20, None)
'''


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
    if ([a.arg for a in fn.args.args] != ['in_out_ptr0', 'in_ptr0', 'in_ptr1', 'out_ptr0', 'xnumel', 'XBLOCK']
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults):
        raise ValueError('Unexpected pointer signature')
    first = fn.body[0] if fn.body else None
    if not (isinstance(first, ast.Assign) and isinstance(first.value, ast.Constant)
            and type(first.value.value) is int and 0 < first.value.value < 2**31):
        raise ValueError('Invalid element count')
    elements = first.value.value
    expected = ast.parse(BODY.format(elements=elements)).body
    if [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in expected]:
        raise ValueError('Gated SiLU backward expression or indexing differs')
    return dict(family='GATED_SILU_BACKWARD_GATE_OUTPUT_V1', symbol=symbol, elements=elements,
                output_pointer='in_out_ptr0', unchanged_output_pointer='out_ptr0',
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                mathematical_expression='dy * up * sigmoid(gate) * (1 + gate*(1-sigmoid(gate)))',
                proof_scope='Real derivative of up*silu(gate); single gate-gradient output; no nonzero-bias or loss proof')


def evaluate(gradient, gate, up):
    sigmoid = gate.sigmoid()
    return gradient * up * sigmoid * (1 + gate * (1 - sigmoid))


def reference(metadata, candidate, contract):
    import torch
    if metadata.get('input_output_storage_aliases') != []:
        raise ValueError('Nonaliasing read inputs were not established')
    values = []
    for name in ('in_ptr0', 'in_ptr1', 'in_out_ptr0'):
        value = metadata.get('runtime_pointers', {}).get(name)
        if (not isinstance(value, torch.Tensor) or not value.is_contiguous()
                or value.numel() != contract['elements'] or value.device != candidate.device
                or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError('Input layout or representation differs: ' + name)
        values.append(value.float().reshape(-1))
    if (candidate.numel() != contract['elements'] or not candidate.is_contiguous()
            or candidate.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError('Output layout or representation differs')
    return evaluate(*values).to(candidate.dtype).reshape(candidate.shape)
