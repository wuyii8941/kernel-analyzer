"""Exact source check for separate BF16 channel bias addition."""
import ast
import hashlib


def check_source(source, symbol, *, channels=1536, length=67):
    if any(type(x) is not int or x <= 0 for x in (channels, length)):
        raise ValueError('Positive integer dimensions required')
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Missing or ambiguous bias source')
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != 'triton' or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton definition required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Missing bias function')
    fn = functions[0]
    if ([x.arg for x in fn.args.args] != ['in_out_ptr0', 'in_ptr0', 'xnumel', 'XBLOCK']
            or fn.args.defaults or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Unexpected bias arguments')
    body = f'''xnumel = {channels*length}
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)[:]
xmask = xindex < xnumel
x2 = xindex
x1 = xindex // {length}
tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
tmp2 = tmp0 + tmp1
tl.store(in_out_ptr0 + (x2), tmp2, xmask)
'''
    if [ast.dump(n) for n in fn.body] != [ast.dump(n) for n in ast.parse(body).body]:
        raise ValueError('Bias arithmetic or indexing differs')
    signatures = []
    for d in fn.decorator_list:
        if not isinstance(d, ast.Call): continue
        for kw in d.keywords:
            if kw.arg == 'triton_meta' and isinstance(kw.value, ast.Dict):
                for k, v in zip(kw.value.keys, kw.value.values):
                    if isinstance(k, ast.Constant) and k.value == 'signature':
                        signatures.append(ast.literal_eval(v))
    # PyTorch/Inductor releases disagree on whether launch-time constexpr
    # parameters are repeated in ``triton_meta['signature']``.  XBLOCK is not
    # a storage operand and both conventions describe the same kernel ABI.
    # Keep this allow-list exact: accepting arbitrary extra entries here could
    # hide a changed tensor operand or a changed storage dtype.
    allowed_signatures = [
        dict(in_out_ptr0='*bf16', in_ptr0='*bf16', xnumel='i32'),
        dict(in_out_ptr0='*bf16', in_ptr0='*bf16', xnumel='i32',
             XBLOCK='constexpr'),
    ]
    if len(signatures) != 1 or signatures[0] not in allowed_signatures:
        raise ValueError('Bias storage precision differs')
    semantic = ast.Module(body=[ast.FunctionDef(
        name=fn.name, args=fn.args, body=fn.body, decorator_list=[],
        returns=fn.returns, type_comment=fn.type_comment,
    )], type_ignores=[])
    return dict(symbol=symbol, channels=channels, length=length,
                function_ast_sha256=hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                # Generated decorator metadata changes between Inductor
                # releases and devices.  The checked function ABI, indexing,
                # storage dtypes and arithmetic above are the semantic
                # identity used to bind a frozen task to the executed source.
                function_semantic_ast_sha256=hashlib.sha256(
                    ast.dump(semantic).encode()).hexdigest(),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                runtime_binding_complete=False)
