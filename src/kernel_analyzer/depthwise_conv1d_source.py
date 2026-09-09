"""Check declared external convolution calls; runtime tensor checks still needed."""
import ast
import hashlib


def check_call(expression):
    call = ast.parse(expression, mode='eval').body
    if (not isinstance(call, ast.Call) or ast.unparse(call.func) != 'extern_kernels.convolution'
            or len(call.args) != 2 or any(not isinstance(x, ast.Name) for x in call.args)):
        raise ValueError('Expected generated external convolution with two tensor names')
    if len({kw.arg for kw in call.keywords}) != len(call.keywords):
        raise ValueError('Repeated convolution keyword')
    try:
        options = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    except (ValueError, TypeError):
        raise ValueError('Convolution options must be literal') from None
    expected = dict(stride=(1,), padding=(3,), dilation=(1,), transposed=False,
                    output_padding=(0,), groups=1536, bias=None)
    # Include types: bools must not masquerade as integer options.
    if (options != expected or any(type(options[k]) is not type(v) for k, v in expected.items())
            or any(type(x) is not int for k in ('stride','padding','dilation','output_padding')
                   for x in options[k])):
        raise ValueError('Unsupported convolution options')
    return dict(input_name=call.args[0].id, weight_name=call.args[1].id,
                groups=1536, padding=3, stride=1, dilation=1, bias_included=False,
                required_input_shape='[batch,1536,length]', required_weight_shape=[1536,1,4],
                call_ast_sha256=hashlib.sha256(ast.dump(call).encode()).hexdigest(),
                runtime_binding_complete=False)


def scan(source):
    records = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if ast.unparse(node.value.func) != 'extern_kernels.convolution':
            continue
        row = dict(line=node.lineno, expression=ast.unparse(node.value))
        try:
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError('Unsupported output binding')
            row.update(status='SOURCE_CHECKED', output_name=node.targets[0].id,
                       contract=check_call(row['expression']))
        except ValueError as exc:
            row.update(status='UNSUPPORTED', reason=str(exc))
        records.append(row)
    return sorted(records, key=lambda r: r['line'])
