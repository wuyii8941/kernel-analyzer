"""Mathematics of gated convolution input/bias gradients, not runtime support.

The saved Mamba computation combines skip, recurrent and gated gradients before
applying SiLU backward. Its bias reduction precedes output storage rounding.
Source validation and exact two-output binding remain separate requirements.
"""

import ast
import hashlib


BODY = '''
xnumel = {channels}
r0_numel = {padded}
rnumel = r0_numel
RBLOCK: tl.constexpr = R0_BLOCK
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
xmask = xindex < xnumel
r0_base = tl.arange(0, R0_BLOCK)[None, :]
rbase = r0_base
x0 = xindex
_tmp38 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
    r0_index = r0_offset + r0_base
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    tmp0 = r0_1
    tmp1 = tl.full([1, 1], {steps}, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (r0_1 + {steps}*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0 + {channels}*r0_1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
    tmp5 = -tmp4
    tmp6 = libdevice.exp(tmp5)
    tmp7 = tl.full([1, 1], 1.0, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = (tmp4 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp3 * tmp10
    tmp12 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.load(in_ptr3 + (r0_1 + {steps}*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tl.load(in_ptr4 + (r0_1 + {steps}*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.load(in_out_ptr0 + (r0_1 + {padded}*x0), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = -tmp21
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tmp23 + tmp7
    tmp25 = (tmp7 / tmp24)
    tmp26 = tmp25 * tmp7
    tmp27 = tmp19 * tmp26
    tmp28 = tmp7 - tmp26
    tmp29 = tmp21 * tmp28
    tmp30 = tmp29 + tmp7
    tmp31 = tmp27 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp2, tmp32, tmp33)
    tmp35 = tl.full([1, 1], 0.0, tl.float32)
    tmp36 = tl.where(tmp2, tmp34, tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = _tmp38 + tmp37
    _tmp38 = tl.where(r0_mask & xmask, tmp39, _tmp38)
    tl.store(in_out_ptr0 + (r0_1 + {padded}*x0), tmp36, r0_mask & xmask)
tmp38 = tl.sum(_tmp38, 1)[:, None]
tl.store(out_ptr0 + (x0), tmp38, xmask)
'''


def check_source(source, symbol, output_pointer='out_ptr0'):
    assignments=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(assignments)!=1:
        raise ValueError('Missing or ambiguous kernel assignment')
    call=assignments[0].value
    if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
            and call.func.attr=='triton' and len(call.args)>=2
            and isinstance(call.args[1],ast.Constant) and isinstance(call.args[1].value,str)):
        raise ValueError('Literal Triton source required')
    functions=[n for n in ast.parse(call.args[1].value).body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1: raise ValueError('Missing function')
    fn=functions[0]
    names=['in_out_ptr0','in_ptr0','in_ptr1','in_ptr2','in_ptr3','in_ptr4','out_ptr0',
           'xnumel','r0_numel','XBLOCK','R0_BLOCK']
    if ([a.arg for a in fn.args.args]!=names or fn.args.defaults or fn.args.kwonlyargs
            or fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected pointer signature')
    try:
        channels,padded=[n.value.value for n in fn.body[:2]]
        steps=next(n.value.args[1].value for n in ast.walk(fn) if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='tmp1' for t in n.targets))
    except (AttributeError,IndexError,StopIteration,TypeError) as e:
        raise ValueError('Literal dimensions required') from e
    if (any(type(x)is not int or x<=0 for x in (channels,padded,steps))
            or padded<steps or channels*padded>=2**31):
        raise ValueError('Invalid dimensions')
    expected=ast.parse(BODY.format(channels=channels,padded=padded,steps=steps)).body
    if [ast.dump(n) for n in fn.body]!=[ast.dump(n) for n in expected]:
        raise ValueError('Gradient composition, storage or reduction differs')
    signatures=[]
    for d in fn.decorator_list:
        if isinstance(d,ast.Call):
            for kw in d.keywords:
                if kw.arg=='triton_meta' and isinstance(kw.value,ast.Dict):
                    for k,v in zip(kw.value.keys,kw.value.values):
                        if isinstance(k,ast.Constant) and k.value=='signature': signatures.append(ast.literal_eval(v))
    signature={name:('*fp32' if name in ('in_ptr1','in_ptr3') else '*bf16') for name in names[:7]}
    signature.update(xnumel='i32',r0_numel='i32')
    if signatures!=[signature] or output_pointer not in ('in_out_ptr0','out_ptr0'):
        raise ValueError('Storage dtype or output differs')
    return dict(family='GATED_CONV_GRADIENT_V1',symbol=symbol,channels=channels,steps=steps,
                padding=padded-steps,output_pointer=output_pointer,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                function_semantic_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=[
                    ast.FunctionDef(name=fn.name,args=fn.args,body=fn.body,decorator_list=[],
                                    returns=fn.returns,type_comment=fn.type_comment)
                ],type_ignores=[])).encode()).hexdigest(),
                scope='One declared output of gated gradient composition; no nonzero-bias claim')


def evaluate(preactivation, gate, upstream, scale, recurrent, skip):
    """Inputs use [channels, steps], except gate [steps, channels], scale [channels].

    preactivation additionally includes trailing convolution padding. Return
    unrounded padded gradient and bias gradient in the common arithmetic dtype.
    """
    import torch
    if upstream.ndim != 2:
        raise ValueError('Expected channel/time gradient')
    channels, steps = upstream.shape
    if (channels <= 0 or steps <= 0 or preactivation.ndim != 2
            or preactivation.shape[0] != channels or preactivation.shape[1] < steps
            or gate.shape != (steps, channels) or scale.shape != (channels,)
            or recurrent.shape != upstream.shape or skip.shape != upstream.shape):
        raise ValueError('Gradient, gate, scale or padding shape differs')
    tensors = (preactivation, gate, upstream, scale, recurrent, skip)
    if any(not x.is_floating_point() or x.dtype != upstream.dtype
           or x.device != upstream.device or not torch.isfinite(x).all() for x in tensors):
        raise ValueError('Finite common arithmetic dtype and device required')
    x = preactivation[:, :steps]
    sigmoid = torch.sigmoid(x)
    combined = upstream * torch.nn.functional.silu(gate.T) * scale[:, None] + recurrent + skip
    gradient = combined * sigmoid * (1 + x * (1 - sigmoid))
    padded = torch.nn.functional.pad(gradient, (0, preactivation.shape[1] - steps))
    return padded, gradient.sum(dim=1)


def reference(metadata, candidate, contract):
    """Select one explicitly declared output using pre-call pointer snapshots.

    This adapter is not registered for capture until source checks are added.
    """
    import torch
    channels, steps, padding = (contract[k] for k in ('channels','steps','padding'))
    output = contract['output_pointer']
    if output not in ('in_out_ptr0','out_ptr0') or metadata.get('formal_pointer') != output:
        raise ValueError('Undeclared output boundary')
    sizes = {'in_out_ptr0': (channels,steps+padding), 'in_ptr0': (channels,steps),
             'in_ptr1': (steps,channels), 'in_ptr2': (channels,),
             'in_ptr3': (channels,steps), 'in_ptr4': (channels,steps)}
    import math
    values = {}
    for name,shape in sizes.items():
        value = metadata.get('runtime_pointers',{}).get(name)
        dtype = torch.float32 if name in ('in_ptr1','in_ptr3') else torch.bfloat16
        if (not isinstance(value,torch.Tensor) or value.dtype != dtype
                or value.device != candidate.device or not value.is_contiguous()
                or value.numel() != math.prod(shape)):
            raise ValueError('Pre-call input contract differs: '+name)
        values[name] = value.float().reshape(shape)
    expected_shape = sizes['in_out_ptr0'] if output == 'in_out_ptr0' else (channels,)
    if (candidate.dtype != torch.bfloat16 or not candidate.is_contiguous()
            or candidate.numel() != math.prod(expected_shape)
            or metadata.get('input_output_storage_aliases') != []):
        raise ValueError('Output storage contract differs')
    gradient,bias = evaluate(values['in_out_ptr0'],values['in_ptr1'],values['in_ptr0'],
                             values['in_ptr2'],values['in_ptr3'],values['in_ptr4'])
    return (gradient if output == 'in_out_ptr0' else bias).to(candidate.dtype).reshape(candidate.shape)
