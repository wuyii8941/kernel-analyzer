"""Complete generated-body contract; matching is not runtime execution proof."""
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
tmp0 = tl.load(in_out_ptr0 + (r0_2 + {width}*x3), xmask, other=0.0).to(tl.float32)
tmp8 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
tmp9 = tl.load(in_ptr0 + r0_2, None, eviction_policy='evict_last')
tmp1 = tl.full([1, 1], {scale}, tl.float32)
tmp2 = tmp0 * tmp1
tmp3 = r0_2
tmp4 = x0
tmp5 = tmp3 <= tmp4
tmp6 = tl.full([1, 1], True, tl.int1)
tmp7 = tmp6 & tmp5
tmp10 = tmp8 == tmp9
tmp11 = tmp7 & tmp10
tmp12 = tl.full([1, 1], 0.0, tl.float32)
tmp13 = tl.full([1, 1], -3.3895313892515355e+38, tl.float32)
tmp14 = tl.where(tmp11, tmp12, tmp13)
tmp15 = tmp2 + tmp14
tmp16 = tmp15.to(tl.float32)
tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
tmp19 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
tmp21 = tl.where(xmask, tmp19, float("-inf"))
tmp22 = triton_helpers.max2(tmp21, 1)[:, None].to(tl.float32)
tmp23 = tmp17 - tmp22
tmp24 = libdevice.exp(tmp23)
tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
tmp27 = tl.where(xmask, tmp25, 0)
tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
tmp29 = tmp16 - tmp22
tmp30 = libdevice.exp(tmp29)
tmp31 = tmp30 / tmp28
tmp32 = tmp31.to(tl.float32)
tl.store(in_out_ptr0 + (r0_2 + {width}*x3), tmp15, xmask)
tl.store(out_ptr2 + (r0_2 + {width}*x3), tmp32, xmask)
tl.store(out_ptr0 + x3, tmp22, xmask)
tl.store(out_ptr1 + x3, tmp28, xmask)
'''


def expected_body(rows, width, scale, unmasked=False):
    text = BODY.format(rows=rows, width=width, scale=repr(scale))
    if unmasked:
        text = text.split('tmp21 =')[0]
        text = text.replace('xmask = xindex < xnumel',
                            'xmask = tl.full([XBLOCK], True, tl.int1)[:, None]')
        text = text.replace('xmask, other=0.0', 'None')
        text = text.replace('xmask, eviction_policy', 'None, eviction_policy')
        text += f'''
tmp21 = triton_helpers.max2(tmp19, 1)[:, None].to(tl.float32)
tmp22 = tmp17 - tmp21
tmp23 = libdevice.exp(tmp22)
tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
tmp26 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
tmp27 = tmp16 - tmp21
tmp28 = libdevice.exp(tmp27)
tmp29 = tmp28 / tmp26
tmp30 = tmp29.to(tl.float32)
tl.store(in_out_ptr0 + (r0_2 + {width}*x3), tmp15, None)
tl.store(out_ptr2 + (r0_2 + {width}*x3), tmp30, None)
tl.store(out_ptr0 + x3, tmp21, None)
tl.store(out_ptr1 + x3, tmp26, None)
'''
    return ast.parse(text).body


def check_source(source, symbol):
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Missing or ambiguous kernel')
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != 'triton' or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant) or not isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton source required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Missing or ambiguous function')
    fn = functions[0]
    names = ['in_out_ptr0', 'in_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2',
             'xnumel', 'r0_numel', 'XBLOCK']
    if ([a.arg for a in fn.args.args] != names or fn.args.defaults or fn.args.posonlyargs
            or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected arguments')
    def value(name):
        values = [n.value for n in fn.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        if len(values) != 1:
            raise ValueError('Missing or ambiguous constant: ' + name)
        return values[0]
    rows, width = ast.literal_eval(value('xnumel')), ast.literal_eval(value('r0_numel'))
    scale_call = value('tmp1')
    if not isinstance(scale_call, ast.Call) or len(scale_call.args) != 3:
        raise ValueError('Explicit scale required')
    scale = ast.literal_eval(scale_call.args[1])
    if (type(rows) is not int or type(width) is not int or min(rows, width) <= 0
            or rows % width or width & (width-1) or rows*width >= 2**31
            or type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid dimensions or scale')
    actual = [ast.dump(n) for n in fn.body]
    matches = [unmasked for unmasked in (False, True)
               if actual == [ast.dump(n) for n in expected_body(rows, width, scale, unmasked)]]
    if len(matches) != 1:
        raise ValueError('Arithmetic, mask, stores or indexing differs')
    signatures = []
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for kw in decorator.keywords:
            if kw.arg == 'triton_meta' and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant) and k.value == 'signature':
                        signatures.append(ast.literal_eval(v))
    expected_types = dict(in_out_ptr0='*bf16', in_ptr0='*i64', out_ptr0='*fp32',
                          out_ptr1='*fp32', out_ptr2='*bf16', xnumel='i32', r0_numel='i32')
    if signatures != [expected_types]:
        raise ValueError('Storage types differ')
    return dict(symbol=symbol, rows=rows, width=width, scale=scale,
                row_bounds_masked=not matches[0],
                launch_condition='EXACT_ROW_COVERAGE_REQUIRED' if matches[0] else 'ROW_BOUNDS_MASKED',
                output_pointers=['in_out_ptr0', 'out_ptr0', 'out_ptr1', 'out_ptr2'],
                required_pre_call_inputs=['in_out_ptr0', 'in_ptr0'],
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                runtime_binding_complete=False,
                proof_scope='SOURCE_FORMULA_AND_STORAGE_ONLY_NOT_BIAS_PROOF')
