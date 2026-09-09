"""Independent grouped RoPE VJP mathematics; source/runtime binding pending.

Inputs describe repeated attention heads, with only the leading rotary
coordinates transformed. The other coordinates are a separate output boundary.
No numerical bias or training consequence follows from this identity alone.
"""
import ast
import hashlib


BODY = '''
ynumel = {rows}
xnumel = {steps}
yoffset = tl.program_id(1) * YBLOCK
yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
ymask = yindex < ynumel
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
xmask = xindex < xnumel
y0 = yindex % 96
x2 = xindex
y1 = yindex // 96
y3 = yindex
tmp27 = tl.load(in_ptr0 + (x2 + {steps}*y0 + {group_stride}*y1), xmask & ymask).to(tl.float32)
tmp28 = tl.load(in_ptr0 + ({head_stride} + x2 + {steps}*y0 + {group_stride}*y1), xmask & ymask).to(tl.float32)
tmp30 = tl.load(in_ptr0 + ({two_heads} + x2 + {steps}*y0 + {group_stride}*y1), xmask & ymask).to(tl.float32)
tmp32 = tl.load(in_ptr2 + (y0 + 96*x2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
tmp0 = y0
tmp1 = tl.full([1, 1], 48, tl.int64)
tmp2 = tmp0 >= tmp1
tmp3 = tl.load(in_ptr0 + ((-{half_stride}) + x2 + {steps}*y0 + {group_stride}*y1), tmp2 & xmask & ymask, other=0.0).to(tl.float32)
tmp4 = tl.load(in_ptr0 + ({head_minus_half} + x2 + {steps}*y0 + {group_stride}*y1), tmp2 & xmask & ymask, other=0.0).to(tl.float32)
tmp5 = tmp3 + tmp4
tmp6 = tl.load(in_ptr0 + ({two_heads_minus_half} + x2 + {steps}*y0 + {group_stride}*y1), tmp2 & xmask & ymask, other=0.0).to(tl.float32)
tmp7 = tmp5 + tmp6
tmp8 = tl.load(in_ptr1 + ((-48) + y0 + 96*x2), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
tmp9 = tmp7 * tmp8
tmp10 = -tmp9
tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
tmp12 = tl.where(tmp2, tmp10, tmp11)
tmp13 = tl.full([1, 1], 0.0, tl.float32)
tmp14 = tl.where(tmp2, tmp12, tmp13)
tmp15 = tmp0 < tmp1
tmp16 = tl.load(in_ptr0 + ({half_stride} + x2 + {steps}*y0 + {group_stride}*y1), tmp15 & xmask & ymask, other=0.0).to(tl.float32)
tmp17 = tl.load(in_ptr0 + ({head_plus_half} + x2 + {steps}*y0 + {group_stride}*y1), tmp15 & xmask & ymask, other=0.0).to(tl.float32)
tmp18 = tmp16 + tmp17
tmp19 = tl.load(in_ptr0 + ({two_heads_plus_half} + x2 + {steps}*y0 + {group_stride}*y1), tmp15 & xmask & ymask, other=0.0).to(tl.float32)
tmp20 = tmp18 + tmp19
tmp21 = tl.load(in_ptr1 + (48 + y0 + 96*x2), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
tmp22 = tmp20 * tmp21
tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
tmp24 = tl.where(tmp15, tmp22, tmp23)
tmp25 = tl.where(tmp15, tmp24, tmp13)
tmp26 = tmp14 + tmp25
tmp29 = tmp27 + tmp28
tmp31 = tmp29 + tmp30
tmp33 = tmp31 * tmp32
tmp34 = tmp26 + tmp33
tl.store(out_ptr0 + (x2 + {steps}*y3), tmp34, xmask & ymask)
'''


def expected_body(rows, steps):
    return BODY.format(rows=rows,steps=steps,group_stride=384*steps,
                       head_stride=128*steps,two_heads=256*steps,half_stride=48*steps,
                       head_minus_half=80*steps,two_heads_minus_half=208*steps,
                       head_plus_half=176*steps,two_heads_plus_half=304*steps)


def check_source(source,symbol):
    matches=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(matches)!=1:
        raise ValueError('Missing or ambiguous kernel definition')
    call=matches[0].value
    if not (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
            and call.func.attr=='triton' and len(call.args)>=2
            and isinstance(call.args[1],ast.Constant) and isinstance(call.args[1].value,str)):
        raise ValueError('Literal Triton source required')
    functions=[n for n in ast.parse(call.args[1].value).body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1:
        raise ValueError('Missing or ambiguous function')
    fn=functions[0]
    if ([a.arg for a in fn.args.args]!=['in_ptr0','in_ptr1','in_ptr2','out_ptr0','ynumel','xnumel','YBLOCK','XBLOCK']
            or fn.args.defaults or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected pointer signature')
    try: rows,steps=[n.value.value for n in fn.body[:2]]
    except (AttributeError,TypeError,ValueError) as e: raise ValueError('Literal dimensions required') from e
    if type(rows)is not int or type(steps)is not int or rows<=0 or rows%96 or steps<=0 or rows*4*steps>=2**31:
        raise ValueError('Invalid grouped rotary dimensions')
    if [ast.dump(n) for n in fn.body]!=[ast.dump(n) for n in ast.parse(expected_body(rows,steps)).body]:
        raise ValueError('Rotary arithmetic, grouping or addressing differs')
    signatures=[]
    for d in fn.decorator_list:
        if isinstance(d,ast.Call):
            for kw in d.keywords:
                if kw.arg=='triton_meta' and isinstance(kw.value,ast.Dict):
                    for k,v in zip(kw.value.keys,kw.value.values):
                        if isinstance(k,ast.Constant) and k.value=='signature':signatures.append(ast.literal_eval(v))
    if signatures!=[dict(in_ptr0='*bf16',in_ptr1='*bf16',in_ptr2='*bf16',out_ptr0='*bf16',ynumel='i32',xnumel='i32')]:
        raise ValueError('Unexpected storage types')
    return dict(family='GROUPED_ROTARY_BACKWARD_V1',symbol=symbol,groups=rows//96,steps=steps,
                output_pointer='out_ptr0',source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                function_semantic_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=[
                    ast.FunctionDef(name=fn.name,args=fn.args,body=fn.body,decorator_list=[],
                                    returns=fn.returns,type_comment=fn.type_comment)
                ],type_ignores=[])).encode()).hexdigest(),
                scope='Rotary 96 of 128 coordinates, three repeated heads; remaining coordinates excluded',
                proof_scope='Transpose of declared rotation and repeated-head sum; not a numerical-bias proof')


def evaluate(gradient, sine, cosine):
    """gradient: [groups, repeats, head_width, steps]; sin/cos: [steps, rotary].

    For y = x*cos + rotate_half(x)*sin, sum the repeated-head cotangents
    and apply the transpose rotation. No assumption that sin/cos halves match
    is necessary. This function intentionally preserves input arithmetic dtype.
    """
    import torch
    if gradient.ndim != 4 or sine.ndim != 2 or cosine.shape != sine.shape:
        raise ValueError('Grouped gradient and matching two-dimensional sin/cos required')
    steps,rotary = sine.shape
    if (rotary <= 0 or rotary % 2 or steps != gradient.shape[-1]
            or rotary > gradient.shape[-2] or gradient.shape[1] <= 0):
        raise ValueError('Invalid rotary, repeat or sequence dimensions')
    if any(x.device != gradient.device or x.dtype != gradient.dtype for x in (sine,cosine)):
        raise ValueError('Common arithmetic dtype/device required')
    g = gradient[:,:,:rotary,:].sum(1)
    half = rotary//2
    weighted = g * sine.T
    rotated = torch.cat((weighted[:,half:,:], -weighted[:,:half,:]),dim=1)
    return g * cosine.T + rotated


def reference(metadata,candidate,contract):
    import torch
    groups,steps=contract['groups'],contract['steps']
    transposed = (candidate.shape == (1,groups,steps,96)
                  and candidate.stride() == (groups*96*steps,96*steps,1,steps))
    if (metadata.get('input_output_storage_aliases')!=[] or candidate.dtype!=torch.bfloat16
            or candidate.numel()!=groups*96*steps
            or not (candidate.is_contiguous() or transposed)):
        raise ValueError('Output layout or alias contract differs')
    values=[]
    for name,size in [('in_ptr0',groups*384*steps),('in_ptr1',96*steps),('in_ptr2',96*steps)]:
        x=metadata.get('runtime_pointers',{}).get(name)
        if (not isinstance(x,torch.Tensor) or x.dtype!=torch.bfloat16 or x.device!=candidate.device
                or x.numel()!=size or not x.is_contiguous() or not torch.isfinite(x).all()):
            raise ValueError('Input layout, dtype or finite-value contract differs: '+name)
        values.append(x.float())
    result = evaluate(values[0].reshape(groups,3,128,steps),values[1].reshape(steps,96),
                      values[2].reshape(steps,96)).to(candidate.dtype)
    # Kernel stores [group, rotary, step] physically. The generated output
    # exposes the same storage as [batch=1, group, step, rotary]. A reshape
    # alone would scramble logical coordinates, even with the correct numel.
    if transposed:
        return result.transpose(-1,-2).unsqueeze(0)
    return result.reshape(candidate.shape)
