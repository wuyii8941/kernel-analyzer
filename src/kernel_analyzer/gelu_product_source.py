"""Check the complete single-output tanh-GELU backward product definition."""
import ast
import hashlib


def expected_body(elements, width, stride, offset):
    return ast.parse(f'''
xnumel = {elements}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = tl.full([XBLOCK], True, tl.int1)[:]
x2 = xindex
x0 = (xindex % {width})
x1 = xindex // {width}
tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
tmp1 = tl.load(in_ptr1 + ({offset} + x0 + {stride}*x1), None).to(tl.float32)
tmp4 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
tmp2 = tmp0 * tmp1
tmp3 = tmp2.to(tl.float32)
tmp5 = tmp4.to(tl.float32)
tmp6 = tmp5 * tmp5
tmp7 = tmp6 * tmp5
tmp8 = tl.full([1], 0.044715, tl.float32)
tmp9 = tmp7 * tmp8
tmp10 = tmp5 + tmp9
tmp11 = tl.full([1], 0.7978845608028654, tl.float32)
tmp12 = tmp10 * tmp11
tmp13 = libdevice.tanh(tmp12)
tmp14 = tl.full([1], 1.0, tl.float32)
tmp15 = tmp13 + tmp14
tmp16 = tl.full([1], 0.5, tl.float32)
tmp17 = tmp15 * tmp16
tmp18 = tmp5 * tmp16
tmp19 = tmp13 * tmp13
tmp20 = tmp14 - tmp19
tmp21 = tmp18 * tmp20
tmp22 = tl.full([1], 0.134145, tl.float32)
tmp23 = tmp6 * tmp22
tmp24 = tmp23 + tmp14
tmp25 = tmp24 * tmp11
tmp26 = tmp21 * tmp25
tmp27 = tmp17 + tmp26
tmp28 = tmp3 * tmp27
tmp29 = tmp28.to(tl.float32)
tl.store(out_ptr0 + (x2), tmp29, None)
''').body


def check_source(source, symbol, *, elements, width, stride, offset):
    if (any(type(v) is not int for v in (elements,width,stride,offset))
            or min(elements,width,stride)<=0 or offset<0 or elements%width
            or offset+width>stride):
        raise ValueError('Invalid declared GELU indexing')
    assignments=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(assignments)!=1: raise ValueError('Unique literal source required')
    call=assignments[0].value
    if (not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute)
            or call.func.attr!='triton' or len(call.args)<2
            or not isinstance(call.args[1],ast.Constant) or not isinstance(call.args[1].value,str)):
        raise ValueError('Literal Triton source required')
    functions=[n for n in ast.parse(call.args[1].value).body
               if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1: raise ValueError('Unique function required')
    fn=functions[0]
    if ([a.arg for a in fn.args.args]!=['in_ptr0','in_ptr1','in_ptr2','out_ptr0','xnumel','XBLOCK']
            or fn.args.defaults or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg):
        raise ValueError('GELU signature differs')
    if [ast.dump(n) for n in fn.body]!=[ast.dump(n) for n in expected_body(elements,width,stride,offset)]:
        raise ValueError('GELU arithmetic or addressing differs')
    signatures=[]
    for decorator in fn.decorator_list:
        if not isinstance(decorator,ast.Call): continue
        for kw in decorator.keywords:
            if kw.arg=='triton_meta' and isinstance(kw.value,ast.Dict):
                for key,value in zip(kw.value.keys,kw.value.values):
                    if isinstance(key,ast.Constant) and key.value=='signature':
                        signatures.append(ast.literal_eval(value))
    if signatures!=[dict(in_ptr0='*bf16',in_ptr1='*bf16',in_ptr2='*bf16',out_ptr0='*bf16',xnumel='i32')]:
        raise ValueError('GELU storage types differ')
    return dict(symbol=symbol,elements=elements,width=width,stride=stride,offset=offset,
        output_pointer='out_ptr0',reference_variant='TANH_GELU_PRODUCT_FP32_BF16_WRITE',
        function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),runtime_binding_complete=False,
        launch_requirement='No-mask loads require launch coverage within declared elements')
