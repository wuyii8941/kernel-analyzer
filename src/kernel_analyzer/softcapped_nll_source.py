"""Reviewed Gemma softcap/NLL backward source contract, not runtime evidence."""
import ast
import hashlib
from kernel_analyzer.selected_nll_source import SIGNATURE

BODY_SHA256 = '5948e6e853b4d73aadb7e3b3f38f4036761781bd490625e1ffa8a209833d9983'


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
    digest = hashlib.sha256(ast.dump(ast.Module(body=fn.body, type_ignores=[])).encode()).hexdigest()
    if digest != BODY_SHA256:
        raise ValueError('Unreviewed softcap arithmetic or addressing')
    signatures = []
    for decorator in fn.decorator_list:
        if isinstance(decorator, ast.Call):
            for kw in decorator.keywords:
                if kw.arg == 'triton_meta' and isinstance(kw.value, ast.Dict):
                    for key, value in zip(kw.value.keys, kw.value.values):
                        if isinstance(key, ast.Constant) and key.value == 'signature':
                            signatures.append(ast.literal_eval(value))
    if signatures != [SIGNATURE]:
        raise ValueError('Storage types differ')
    return dict(body_sha256=digest, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                tokens=128, vocabulary=262144, label_offset=1, cap=30.,
                output_pointer='in_out_ptr0', output_dtype='bfloat16',
                clone_input_before_execution=True, reference='softcapped_nll_backward',
                scope='FUSED_SOFTCAP_NLL_BACKWARD_NOT_ISOLATED_TANH',
                runtime_binding_complete=False)
