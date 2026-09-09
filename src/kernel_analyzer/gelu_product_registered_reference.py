"""Registry adapter for the reviewed tanh-GELU backward product family.

The older family scanner inferred layout constants before calling the strict
source checker.  This adapter performs the same inference behind the common
registry interface so every frozen source pool can be checked automatically.
"""

import ast

from kernel_analyzer.gelu_product_observer import validate_pointers
from kernel_analyzer.gelu_product_reference import evaluate
from kernel_analyzer.gelu_product_source import check_source as _check_source


def check_source(source, symbol):
    assignments = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == symbol
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("Missing or ambiguous GELU source")
    call = assignments[0].value
    if (not isinstance(call, ast.Call) or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError("Literal Triton GELU source required")
    functions = [
        node for node in ast.parse(call.args[1].value).body
        if isinstance(node, ast.FunctionDef) and node.name == symbol
    ]
    if len(functions) != 1:
        raise ValueError("Missing or ambiguous GELU function")
    function = functions[0]
    try:
        values = {
            target.id: statement.value
            for statement in function.body if isinstance(statement, ast.Assign)
            for target in statement.targets if isinstance(target, ast.Name)
        }
        elements = ast.literal_eval(values["xnumel"])
        width = ast.literal_eval(values["x0"].right)
        address = values["tmp1"].func.value.args[0].right
        offset = ast.literal_eval(address.left.left)
        stride = ast.literal_eval(address.right.left)
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("GELU layout constants are not the reviewed form") from exc
    return _check_source(source, symbol, elements=elements, width=width,
                         stride=stride, offset=offset)


def reference(metadata, candidate, contract):
    if metadata.get("symbol") != contract.get("symbol"):
        raise ValueError("GELU symbol differs")
    if metadata.get("formal_pointer") != contract.get("output_pointer"):
        raise ValueError("GELU output boundary differs")
    if metadata.get("input_output_storage_aliases"):
        raise ValueError("Unsupported GELU input/output alias")
    decoded = validate_pointers(metadata.get("runtime_pointers") or {}, contract)
    return evaluate(*decoded, output_dtype=candidate.dtype).reshape(candidate.shape)
