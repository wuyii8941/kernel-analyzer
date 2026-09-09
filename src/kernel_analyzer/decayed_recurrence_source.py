"""Exact body template for one observed unrolled recurrence layout.

Body matching alone does not validate pointer dtypes, runtime layouts or training
binding. This helper is not yet registered as a complete source reference.
"""
import ast
import hashlib


def expected_body(channels=1536, width=16, outputs=63):
    if min(channels,width,outputs)<1:
        raise ValueError('Positive dimensions required')
    lines=[f'xnumel = {channels*width}', 'xoffset = tl.program_id(0) * XBLOCK',
           'xindex = xoffset + tl.arange(0, XBLOCK)[:]',
           'xmask = tl.full([XBLOCK], True, tl.int1)[:]',
           f'x1 = xindex // {width}', f'x0 = xindex % {width}', 'x2 = xindex']
    def load(tmp, pointer, offset):
        lines.append(f'tmp{tmp} = tl.load(in_ptr{pointer} + ({offset}), None, eviction_policy="evict_last").to(tl.float32)')
    load(0,0,'x1'); load(2,1,'x0')
    lines.append('tmp5 = tl.load(in_ptr2 + x2, None)')
    for j in range(outputs):
        time=8 if j==0 else 10+18*j
        load(time,3,f'{(outputs-j)*channels} + x1')
        if j==0: load(9,4,'x1')
        load(22+18*j,5+2*j,'x1'); load(24+18*j,6+2*j,'x0')
    lines += ['tmp1 = tmp0.to(tl.float32)', 'tmp3 = tmp2.to(tl.float32)',
              'tmp4 = tmp1 * tmp3', 'tmp6 = libdevice.exp(tmp5)', 'tmp7 = -tmp6']
    for j in range(outputs):
        base=10+18*j
        time=8 if j==0 else base
        # Later blocks omit the first block's shared threshold allocation.
        start=base if j==0 else base+1
        lines += [f'tmp{start} = tmp{time} + tmp9',
                  f'tmp{start+1} = tmp{start}.to(tl.float32)']
        if j==0: lines.append('tmp12 = tl.full([1], 20.0, tl.float32)')
        b=13+18*j
        lines += [f'tmp{b} = tmp{base+1 if j==0 else base+2} > tmp12',
                  f'tmp{b+1} = libdevice.exp(tmp{base+1 if j==0 else base+2})',
                  f'tmp{b+2} = libdevice.log1p(tmp{b+1})',
                  f'tmp{b+3} = tl.where(tmp{b}, tmp{base+1 if j==0 else base+2}, tmp{b+2})',
                  f'tmp{b+4} = tmp{b+3}.to(tl.float32)',
                  f'tmp{b+5} = tmp{b+4}.to(tl.float32)',
                  f'tmp{b+6} = tmp7 * tmp{b+5}',
                  f'tmp{b+7} = libdevice.exp(tmp{b+6})',
                  f'tmp{b+8} = tmp{4 if j==0 else 9+18*j} * tmp{b+7}',
                  f'tmp{b+10} = tmp{b+9}.to(tl.float32)',
                  f'tmp{b+12} = tmp{b+11}.to(tl.float32)',
                  f'tmp{b+13} = tmp{b+10} * tmp{b+12}',
                  f'tmp{b+14} = tmp{b+8} + tmp{b+13}']
    lines += [f'tl.store(out_ptr{j} + x2, tmp{27+18*j}, None)' for j in range(outputs)]
    return '\n'.join(lines)


def body_matches(function, **dimensions):
    expected=ast.parse(expected_body(**dimensions)).body
    return [ast.dump(n) for n in function.body]==[ast.dump(n) for n in expected]


def check_source(source, symbol):
    matches=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(matches)!=1:
        raise ValueError('Missing or ambiguous recurrence source')
    call=matches[0].value
    if (not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute)
            or call.func.attr!='triton' or len(call.args)<2
            or not isinstance(call.args[1],ast.Constant) or not isinstance(call.args[1].value,str)):
        raise ValueError('Literal Triton source required')
    functions=[n for n in ast.parse(call.args[1].value).body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1: raise ValueError('Missing recurrence function')
    fn=functions[0]
    names=[a.arg for a in fn.args.args]
    outputs=sum(n.startswith('out_ptr') for n in names)
    inputs=5+2*outputs
    expected=[f'in_ptr{i}' for i in range(inputs)]+[f'out_ptr{i}' for i in range(outputs)]+['xnumel','XBLOCK']
    if (outputs<1 or names!=expected or fn.args.defaults or fn.args.posonlyargs
            or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected recurrence arguments')
    # First supported layout; other dimensions must be explicitly validated.
    if not body_matches(fn,channels=1536,width=16,outputs=outputs):
        raise ValueError('Recurrence arithmetic, offsets or writes differ')
    signatures=[]
    for decorator in fn.decorator_list:
        if not isinstance(decorator,ast.Call): continue
        for kw in decorator.keywords:
            if kw.arg=='triton_meta' and isinstance(kw.value,ast.Dict):
                for key,value in zip(kw.value.keys,kw.value.values):
                    if isinstance(key,ast.Constant) and key.value=='signature':
                        signatures.append(ast.literal_eval(value))
    signature={f'in_ptr{i}':'*fp32' if i==2 else '*bf16' for i in range(inputs)}
    signature.update({f'out_ptr{i}':'*fp32' for i in range(outputs)})
    signature['xnumel']='i32'
    if signatures!=[signature]: raise ValueError('Recurrence storage types differ')
    return dict(symbol=symbol,channels=1536,state_width=16,outputs=outputs,
        time_offsets_descending=list(range(outputs,0,-1)),
        output_pointers=[f'out_ptr{i}' for i in range(outputs)],
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
        runtime_binding_complete=False,proof_scope='Source recurrence identity only; not bias or training validity')
