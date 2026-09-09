"""Initial exact-body binding for the reviewed in-place Triton NLL backward.

The digest covers the entire AST body, not its name. Other layouts remain
unsupported until reviewed; this alone does not prove runtime execution.
"""
import ast
import hashlib

BODY_SHA256 = 'bdbf5264d42034754b0fb1d31353fd3904f6c8c0770682585a7b99af91e536b4'
SIGNATURE = dict(in_out_ptr0='*bf16', in_ptr0='*i64', in_ptr1='*fp32',
                 in_ptr2='*fp32', in_ptr3='*fp32', in_ptr4='*fp32',
                 xnumel='i32', r0_numel='i32')


def check_source(source, symbol):
    assignments = [n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        raise ValueError('Unique source assignment required')
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
            or call.func.attr != 'triton' or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant) or not isinstance(call.args[1].value, str)):
        raise ValueError('Literal Triton source required')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Unique function required')
    fn = functions[0]
    if ([a.arg for a in fn.args.args] != list(SIGNATURE)+['XBLOCK', 'R0_BLOCK']
            or fn.args.defaults or fn.args.posonlyargs or fn.args.kwonlyargs
            or fn.args.vararg or fn.args.kwarg):
        raise ValueError('Pointer argument contract differs')
    body = ast.Module(body=fn.body, type_ignores=[])
    digest = hashlib.sha256(ast.dump(body).encode()).hexdigest()
    if digest != BODY_SHA256:
        raise ValueError('Unreviewed NLL arithmetic or addressing')
    signatures = []
    for decorator in fn.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for kw in decorator.keywords:
            if kw.arg == 'triton_meta' and isinstance(kw.value, ast.Dict):
                for key, value in zip(kw.value.keys, kw.value.values):
                    if isinstance(key, ast.Constant) and key.value == 'signature':
                        signatures.append(ast.literal_eval(value))
    if signatures != [SIGNATURE]:
        raise ValueError('Storage types differ')
    return dict(body_sha256=digest, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                tokens=128, vocabulary=151936, label_offset=1,
                output_pointer='in_out_ptr0', output_dtype='bfloat16',
                clone_input_before_execution=True,
                reference='selected_nll_backward', runtime_binding_complete=False)
