"""Fail-closed full-body check for the observed forward recurrence layout.

This checks source semantics, not runtime launch identity or training bias.
"""
import ast
import hashlib


def expected_body(steps):
    if type(steps) is not int or steps < 2:
        raise ValueError('At least two steps required')
    header = ['xnumel = 24576', 'xoffset = tl.program_id(0) * XBLOCK',
              'xindex = xoffset + tl.arange(0, XBLOCK)[:]',
              'xmask = tl.full([XBLOCK], True, tl.int1)[:]',
              'x2 = xindex', 'x1 = xindex // 16', 'x0 = xindex % 16']
    loads, arithmetic, stores = [], [], []
    counter = 0

    def assign(expr, target=arithmetic):
        nonlocal counter
        name = f'tmp{counter}'
        counter += 1
        target.append(f'{name} = {expr}')
        return name

    def load(pointer, offset):
        return assign(f'tl.load(in_ptr{pointer} + ({offset}), None, '
                      "eviction_policy='evict_last').to(tl.float32)", loads)

    rate = assign('tl.load(in_ptr0 + (x2), None)', loads)
    decay_rate = assign(f'libdevice.exp({rate})')
    negative = assign(f'-{decay_rate}')
    previous = None
    for t in range(steps):
        time = load(1, 'x1' if t == 0 else f'{t*1536} + x1')
        if t == 0:
            bias = load(2, 'x1')
        shifted = assign(f'{time} + {bias}')
        shifted = assign(f'{shifted}.to(tl.float32)')
        if t == 0:
            threshold = assign('tl.full([1], 20.0, tl.float32)')
        large = assign(f'{shifted} > {threshold}')
        exponential = assign(f'libdevice.exp({shifted})')
        softplus = assign(f'libdevice.log1p({exponential})')
        dt = assign(f'tl.where({large}, {shifted}, {softplus})')
        dt = assign(f'{dt}.to(tl.float32)')
        dt = assign(f'{dt}.to(tl.float32)')
        decay = assign(f'{negative} * {dt}')
        decay = assign(f'libdevice.exp({decay})')
        if t == 0:
            previous = assign('tl.full([1], 0.0, tl.float32)')
        carried = assign(f'{decay} * {previous}')
        state = load(3, f'{48+80*t} + x0')
        state = assign(f'{state}.to(tl.float32)')
        injection = assign(f'{dt} * {state}')
        signal = load(4, f'{steps}*x1' if t == 0 else f'{t} + {steps}*x1')
        signal = assign(f'{signal}.to(tl.float32)')
        injection = assign(f'{injection} * {signal}')
        previous = assign(f'{carried} + {injection}')
        if t == steps-1:
            previous = assign(f'{previous}.to(tl.float32)')
        stores.append(f'tl.store(out_ptr{t} + (x2), {previous}, None)')
    return '\n'.join(header + loads + arithmetic + stores)


def check_source(source, symbol):
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Missing or ambiguous source assignment')
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != 'triton' or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton source required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Missing or ambiguous function')
    fn = functions[0]
    names = [a.arg for a in fn.args.args]
    steps = sum(name.startswith('out_ptr') for name in names)
    expected = ([f'in_ptr{i}' for i in range(5)]
                + [f'out_ptr{i}' for i in range(steps)] + ['xnumel', 'XBLOCK'])
    if (names != expected or fn.args.defaults or fn.args.posonlyargs
            or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected pointer signature')
    if ([ast.dump(n) for n in fn.body]
            != [ast.dump(n) for n in ast.parse(expected_body(steps)).body]):
        raise ValueError('Forward recurrence arithmetic or storage offsets differ')
    signatures = []
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for kw in decorator.keywords:
            if kw.arg == 'triton_meta' and isinstance(kw.value, ast.Dict):
                for key, value in zip(kw.value.keys, kw.value.values):
                    if isinstance(key, ast.Constant) and key.value == 'signature':
                        signatures.append(ast.literal_eval(value))
    signature = {f'in_ptr{i}': '*fp32' if i == 0 else '*bf16' for i in range(5)}
    signature.update({f'out_ptr{i}': '*bf16' if i == steps-1 else '*fp32'
                      for i in range(steps)})
    signature['xnumel'] = 'i32'
    if signatures != [signature]:
        raise ValueError('Forward recurrence storage types differ')
    return dict(symbol=symbol, steps=steps, channels=1536, state_width=16,
                packed_width=80, state_offset=48,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                runtime_binding_complete=False, population_bias_proved=False,
                launch_requirement='Exact row coverage; source loads are unmasked')
