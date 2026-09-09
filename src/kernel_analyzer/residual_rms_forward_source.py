"""Complete body template for residual write followed by RMS reload.

Source matching is not runtime binding or proof of numerical bias.
"""
import ast
import math
import hashlib


def expected_body(rows, width, epsilon):
    if (type(rows) is not int or type(width) is not int or min(rows,width)<1
            or type(epsilon) not in (int,float) or not math.isfinite(epsilon) or epsilon<=0):
        raise ValueError('Invalid RMS dimensions or epsilon')
    return ast.parse(f'''
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
_tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = _tmp6 + tmp5
    _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tl.store(in_out_ptr0 + (r0_1 + {width}*x0), tmp2, r0_mask & xmask)
tmp6 = tl.sum(_tmp6, 1)[:, None]
tmp8 = tl.full([1, 1], {float(width)!r}, tl.float32)
tmp9 = tmp6 / tmp8
tmp10 = tl.full([1, 1], {epsilon!r}, tl.float32)
tmp11 = tmp9 + tmp10
tmp12 = libdevice.rsqrt(tmp11)
tl.store(in_out_ptr1 + x0, tmp12, xmask)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp13 = tl.load(in_ptr1 + r0_1, r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_out_ptr0 + (r0_1 + {width}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp12
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp13 * tmp17
    tl.store(out_ptr0 + (r0_1 + {width}*x0), tmp18, r0_mask & xmask)
''').body


def body_matches(function, *, rows, width, epsilon):
    return [ast.dump(n) for n in function.body] == [
        ast.dump(n) for n in expected_body(rows,width,epsilon)]


def check_source(source, symbol):
    assignments=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(assignments)!=1: raise ValueError('Missing or ambiguous RMS source')
    call=assignments[0].value
    if (not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute)
            or call.func.attr!='triton' or len(call.args)<2
            or not isinstance(call.args[1],ast.Constant) or not isinstance(call.args[1].value,str)):
        raise ValueError('Literal Triton source required')
    functions=[n for n in ast.parse(call.args[1].value).body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1: raise ValueError('Unique RMS function required')
    fn=functions[0]
    names=['in_out_ptr0','in_out_ptr1','in_ptr0','in_ptr1','out_ptr0','xnumel','r0_numel','XBLOCK','R0_BLOCK']
    if ([a.arg for a in fn.args.args]!=names or fn.args.defaults or fn.args.posonlyargs
            or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg):
        raise ValueError('RMS argument layout differs')
    def value(name):
        matches=[n.value for n in fn.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
        if len(matches)!=1: raise ValueError('Ambiguous RMS constant: '+name)
        return matches[0]
    rows=ast.literal_eval(value('xnumel'))
    width=ast.literal_eval(value('r0_numel'))
    epsilon_call=value('tmp10')
    if not isinstance(epsilon_call,ast.Call) or len(epsilon_call.args)!=3:
        raise ValueError('Explicit epsilon required')
    epsilon=ast.literal_eval(epsilon_call.args[1])
    if not body_matches(fn,rows=rows,width=width,epsilon=epsilon):
        raise ValueError('RMS arithmetic, write/reload order or offsets differ')
    signatures=[]
    for d in fn.decorator_list:
        if not isinstance(d,ast.Call): continue
        for kw in d.keywords:
            if kw.arg=='triton_meta' and isinstance(kw.value,ast.Dict):
                for k,v in zip(kw.value.keys,kw.value.values):
                    if isinstance(k,ast.Constant) and k.value=='signature': signatures.append(ast.literal_eval(v))
    expected=dict(in_out_ptr0='*bf16',in_out_ptr1='*fp32',in_ptr0='*bf16',
                  in_ptr1='*bf16',out_ptr0='*bf16',xnumel='i32',r0_numel='i32')
    if signatures!=[expected]: raise ValueError('RMS storage types differ')
    return dict(symbol=symbol,rows=rows,width=width,epsilon=epsilon,
        output_pointers=['in_out_ptr0','in_out_ptr1','out_ptr0'],
        required_pre_call_inputs=['in_ptr0','in_out_ptr0','in_ptr1'],
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
        runtime_binding_complete=False,proof_scope='Source ordering only; not bias or training validity')
