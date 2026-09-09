"""Same-input reference for scaled scores, repeated mask, and saved softmax stats.

Checks one complete generated formula; does not recompute a different forward
state or infer a nonzero bias from correct symbolic differentiation.
"""
import ast
import hashlib
import math


BODY = '''
xnumel = {rows}
r0_numel = {width}
R0_BLOCK: tl.constexpr = {width}
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
r0_2 = r0_index
x3 = xindex
x0 = xindex % {width}
tmp0 = tl.load(in_ptr0 + (r0_2 + {width} * x3), xmask, other=0.0).to(tl.float32)
tmp2 = tl.load(in_out_ptr0 + (r0_2 + {width} * x3), xmask, other=0.0).to(tl.float32)
tmp5 = tl.load(in_ptr1 + (r0_2 + {width} * x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
tmp8 = tl.load(in_ptr2 + x3, xmask, eviction_policy='evict_last')
tmp11 = tl.load(in_ptr3 + x3, xmask, eviction_policy='evict_last')
tmp1 = tmp0.to(tl.float32)
tmp3 = tl.full([1, 1], {scale}, tl.float32)
tmp4 = tmp2 * tmp3
tmp6 = tmp4 + tmp5
tmp7 = tmp6.to(tl.float32)
tmp9 = tmp7 - tmp8
tmp10 = libdevice.exp(tmp9)
tmp12 = tmp10 / tmp11
tmp13 = tmp1 * tmp12
tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
tmp16 = tl.where(xmask, tmp14, 0)
tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
tmp18 = -tmp12
tmp19 = tl.fma(tmp18, tmp17, tmp13)
tmp20 = tmp19.to(tl.float32)
tmp21 = tmp20 * tmp3
tl.store(in_out_ptr0 + (r0_2 + {width} * x3), tmp21, xmask)
'''


def check_source(source, symbol):
    assignments=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(assignments)!=1: raise ValueError('Kernel definition absent or ambiguous')
    call=assignments[0].value
    if not (isinstance(call,ast.Call) and len(call.args)>=2
            and isinstance(call.args[1],ast.Constant) and isinstance(call.args[1].value,str)):
        raise ValueError('Literal kernel source required')
    functions=[n for n in ast.parse(call.args[1].value).body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1: raise ValueError('Kernel function absent or ambiguous')
    fn=functions[0]
    if ([a.arg for a in fn.args.args]!=['in_out_ptr0','in_ptr0','in_ptr1','in_ptr2','in_ptr3','xnumel','r0_numel','XBLOCK']
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults):
        raise ValueError('Unexpected pointer signature')
    try:
        rows,width=(n.value.value for n in fn.body[:2])
        scale=next(n.value.args[1].value for n in fn.body if isinstance(n,ast.Assign)
                   and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='tmp3')
    except (AttributeError,IndexError,TypeError,StopIteration) as exc:
        raise ValueError('Unknown dimensions or scale') from exc
    if (type(rows) is not int or type(width) is not int or rows<=0 or width<=0
            or width & (width-1) or rows*width>=2**31
            or type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0):
        raise ValueError('Invalid dimensions or scale')
    expected=ast.parse(BODY.format(rows=rows,width=width,scale=repr(scale))).body
    if [ast.dump(n) for n in fn.body]!=[ast.dump(n) for n in expected]:
        raise ValueError('Scaled masked softmax expression or indexing differs')
    return dict(family='SCALED_MASKED_SAVED_SOFTMAX_BACKWARD_V1',symbol=symbol,
        rows=rows,width=width,scale=scale,output_pointer='in_out_ptr0',
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
        function_semantic_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=[
            ast.FunctionDef(name=fn.name,args=fn.args,body=fn.body,decorator_list=[],
                            returns=fn.returns,type_comment=fn.type_comment)
        ],type_ignores=[])).encode()).hexdigest(),
        mathematical_expression='s*(g*p-p*sum(g*p)); p=exp(s*scores+mask[row%width]-saved_max)/saved_denominator',
        proof_scope='Declared real formula; softmax derivative when saved statistics are consistent; not bias proof')


def evaluate(gradient, scores, mask, maximum, denominator, scale):
    probability=((scores*scale+mask)-maximum).exp()/denominator
    product=gradient*probability
    return scale*(product-probability*product.sum(-1,keepdim=True))


def reference(metadata,candidate,contract):
    import torch
    if metadata.get('input_output_storage_aliases')!=[]:
        raise ValueError('Nonaliasing read inputs were not established')
    rows,width=contract['rows'],contract['width']
    values=[]
    for name,size in (('in_ptr0',rows*width),('in_out_ptr0',rows*width),
                      ('in_ptr1',width*width),('in_ptr2',rows),('in_ptr3',rows)):
        value=metadata.get('runtime_pointers',{}).get(name)
        if (not isinstance(value,torch.Tensor) or not value.is_contiguous() or value.numel()!=size
                or value.device!=candidate.device or value.dtype not in (torch.float16,torch.bfloat16,torch.float32)):
            raise ValueError('Input layout or representation differs: '+name)
        if name in ('in_ptr2','in_ptr3') and value.dtype!=torch.float32:
            raise ValueError('Saved normalizers must be FP32')
        values.append(value.float())
    if (not candidate.is_contiguous() or candidate.numel()!=rows*width
            or candidate.dtype not in (torch.float16,torch.bfloat16,torch.float32)):
        raise ValueError('Output layout or representation differs')
    gradient,scores,mask,maximum,denominator=values
    if not torch.isfinite(maximum).all() or not (torch.isfinite(denominator)&(denominator>0)).all():
        raise ValueError('Undefined saved softmax normalizers')
    repeated_mask=mask.reshape(width,width)[torch.arange(rows,device=candidate.device)%width]
    scale=torch.tensor(contract['scale'],dtype=torch.float32,device=candidate.device)
    return evaluate(gradient.reshape(rows,width),scores.reshape(rows,width),repeated_mask,
                    maximum.reshape(rows,1),denominator.reshape(rows,1),scale).to(candidate.dtype).reshape(candidate.shape)
