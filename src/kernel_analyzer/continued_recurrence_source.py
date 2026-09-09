"""Arithmetic template for recurrence segments receiving a prior FP32 state.

This is a body check only: storage signatures and runtime binding remain
required before this family may enter training measurement.
"""
import ast
import hashlib


def continued_body(*, channels=1536, width=16, outputs=60, time_start):
    values = (channels, width, outputs, time_start)
    if any(not isinstance(v, int) or isinstance(v, bool) for v in values):
        raise ValueError('Integer dimensions required')
    if min(channels, width, outputs) < 1 or time_start < outputs:
        raise ValueError('Invalid descending segment')
    lines = [f'xnumel = {channels * width}',
             'xoffset = tl.program_id(0) * XBLOCK',
             'xindex = xoffset + tl.arange(0, XBLOCK)[:]',
             'xmask = tl.full([XBLOCK], True, tl.int1)[:]',
             'x2 = xindex', f'x1 = xindex // {width}', f'x0 = xindex % {width}',
             'tmp0 = tl.load(in_ptr0 + x2, None)',
             'tmp1 = tl.load(in_ptr1 + x2, None)']
    def load(tmp, pointer, offset):
        lines.append(f'tmp{tmp} = tl.load(in_ptr{pointer} + ({offset}), None, eviction_policy="evict_last").to(tl.float32)')
    for j in range(outputs):
        load(4 if j == 0 else 6 + 18*j, 2, f'{(time_start-j)*channels} + x1')
        if j == 0:
            load(5, 3, 'x1')
        load(18+18*j, 4+2*j, 'x1')
        load(20+18*j, 5+2*j, 'x0')
    lines += ['tmp2 = libdevice.exp(tmp1)', 'tmp3 = -tmp2']
    for j in range(outputs):
        start = 6 if j == 0 else 7+18*j
        time = 4 if j == 0 else 6+18*j
        lines += [f'tmp{start} = tmp{time} + tmp5',
                  f'tmp{start+1} = tmp{start}.to(tl.float32)']
        if j == 0:
            lines.append('tmp8 = tl.full([1], 20.0, tl.float32)')
        b = 9+18*j
        x = start+1
        lines += [f'tmp{b} = tmp{x} > tmp8',
                  f'tmp{b+1} = libdevice.exp(tmp{x})',
                  f'tmp{b+2} = libdevice.log1p(tmp{b+1})',
                  f'tmp{b+3} = tl.where(tmp{b}, tmp{x}, tmp{b+2})',
                  f'tmp{b+4} = tmp{b+3}.to(tl.float32)',
                  f'tmp{b+5} = tmp{b+4}.to(tl.float32)',
                  f'tmp{b+6} = tmp3 * tmp{b+5}',
                  f'tmp{b+7} = libdevice.exp(tmp{b+6})',
                  f'tmp{b+8} = tmp{0 if j == 0 else 5+18*j} * tmp{b+7}',
                  f'tmp{b+10} = tmp{b+9}.to(tl.float32)',
                  f'tmp{b+12} = tmp{b+11}.to(tl.float32)',
                  f'tmp{b+13} = tmp{b+10} * tmp{b+12}',
                  f'tmp{b+14} = tmp{b+8} + tmp{b+13}']
    lines += [f'tl.store(out_ptr{j} + x2, tmp{23+18*j}, None)' for j in range(outputs)]
    return ast.parse('\n'.join(lines)).body


def continued_matches(function, **dimensions):
    return [ast.dump(n) for n in function.body] == [
        ast.dump(n) for n in continued_body(**dimensions)]


def check_source(source, symbol, *, time_start):
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Missing or ambiguous continued recurrence')
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != 'triton' or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton source required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Unique continued recurrence function required')
    fn = functions[0]
    names = [a.arg for a in fn.args.args]
    outputs = sum(n.startswith('out_ptr') for n in names)
    inputs = 4 + 2*outputs
    expected = ([f'in_ptr{i}' for i in range(inputs)]
                + [f'out_ptr{i}' for i in range(outputs)] + ['xnumel', 'XBLOCK'])
    if (outputs < 1 or names != expected or fn.args.defaults or fn.args.posonlyargs
            or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected continued recurrence arguments')
    if not continued_matches(fn, outputs=outputs, time_start=time_start):
        raise ValueError('Continued recurrence arithmetic or writes differ')
    signatures = []
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for kw in decorator.keywords:
            if kw.arg == 'triton_meta' and isinstance(kw.value, ast.Dict):
                for key, value in zip(kw.value.keys, kw.value.values):
                    if isinstance(key, ast.Constant) and key.value == 'signature':
                        signatures.append(ast.literal_eval(value))
    signature = {f'in_ptr{i}': '*fp32' if i < 2 else '*bf16' for i in range(inputs)}
    signature.update({f'out_ptr{i}': '*fp32' for i in range(outputs)})
    signature['xnumel'] = 'i32'
    if signatures != [signature]:
        raise ValueError('Continued recurrence storage types differ')
    return dict(symbol=symbol, channels=1536, state_width=16, outputs=outputs,
                time_start=time_start,
                time_offsets_descending=list(range(time_start, time_start-outputs, -1)),
                output_pointers=[f'out_ptr{i}' for i in range(outputs)],
                segment_input_kind='FP32_PREVIOUS_STATE',
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                runtime_binding_complete=False,
                proof_scope='Source recurrence identity only; not bias or training validity')
