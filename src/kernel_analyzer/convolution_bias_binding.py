"""Strict local generated-code association, not a runtime alias proof."""
import ast
from kernel_analyzer.depthwise_conv1d_source import check_call


def associate(statements, convolution_index, bias_symbol):
    first = statements[convolution_index]
    if (not isinstance(first, ast.Assign) or len(first.targets) != 1
            or not isinstance(first.targets[0], ast.Name)):
        raise ValueError('Expected single convolution output assignment')
    contract = check_call(ast.unparse(first.value))
    aliases = {first.targets[0].id}
    for node in statements[convolution_index+1:]:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.value.id in aliases):
                aliases.add(node.targets[0].id)
                continue
            raise ValueError('Unreviewed assignment between convolution and bias')
        if isinstance(node, ast.Delete):
            if any(not isinstance(t, ast.Name) or t.id not in aliases for t in node.targets):
                raise ValueError('Unreviewed deletion')
            aliases.difference_update(t.id for t in node.targets)
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            name = ast.unparse(call.func)
            if name in ('assert_size_stride', 'assert_alignment'):
                if any(isinstance(n, ast.Call) for arg in call.args for n in ast.walk(arg)):
                    raise ValueError('Nested computation in assertion')
                continue
            if name == bias_symbol+'.run':
                if (len(call.args) != 3 or any(not isinstance(a, ast.Name) for a in call.args[:2])
                        or call.args[0].id not in aliases or call.args[1].id in aliases
                        or not isinstance(call.args[2], ast.Constant)
                        or type(call.args[2].value) is not int
                        or len(call.keywords) != 1 or call.keywords[0].arg != 'stream'
                        or not isinstance(call.keywords[0].value, ast.Name)):
                    raise ValueError('Bias arguments or buffer association differ')
                return dict(convolution=contract, convolution_line=first.lineno,
                            bias_line=node.lineno, bias_symbol=bias_symbol,
                            output_name=call.args[0].id, bias_name=call.args[1].id,
                            numel=call.args[2].value, runtime_binding_complete=False)
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.startswith('raw_stream')
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == 'get_raw_stream'
                and len(node.value.args) == 1 and not node.value.keywords
                and isinstance(node.value.args[0], ast.Constant)
                and type(node.value.args[0].value) is int):
            continue
        raise ValueError('Intervening computation not reviewed')
    raise ValueError('Missing subsequent bias call')
