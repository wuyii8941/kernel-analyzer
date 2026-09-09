"""Inspect saved Triton pointer types; never infer a numerical family by name."""
import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


def inspect_definition(assignment, symbol):
    if not isinstance(assignment, ast.Assign):
        return {'status': 'UNSUPPORTED_DEFINITION'}
    call = assignment.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'triton' and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant)
            and isinstance(call.args[1].value, str)):
        return {'status': 'UNSUPPORTED_DEFINITION'}
    tree = ast.parse(call.args[1].value)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        return {'status': 'AMBIGUOUS_FUNCTION'}
    signatures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'signature':
                    try:
                        signatures.append(ast.literal_eval(value))
                    except (ValueError, TypeError):
                        return {'status': 'NONLITERAL_SIGNATURE'}
    if len(signatures) != 1 or not isinstance(signatures[0], dict):
        return {'status': 'AMBIGUOUS_SIGNATURE'}
    signature = signatures[0]
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in signature.items()):
        return {'status': 'UNSUPPORTED_SIGNATURE'}
    pointers = {k: v for k, v in signature.items() if v.startswith('*')}
    floating = {'*fp16', '*bf16', '*fp32', '*fp64'}
    integer = {'*i1', '*i8', '*i16', '*i32', '*i64', '*u8', '*u16', '*u32', '*u64'}
    if not pointers or any(v not in floating | integer for v in pointers.values()):
        status = 'UNKNOWN_POINTER_TYPES'
    elif any(v in floating for v in pointers.values()):
        status = 'FLOATING_POINTERS_REQUIRES_SEMANTIC_REVIEW'
    else:
        status = 'INTEGER_BOOLEAN_POINTERS_REQUIRES_CONTROL_REVIEW'
    return dict(status=status, pointer_types=pointers,
                numerical_family_verified=False, runtime_verified=False)


def audit(records):
    definitions, rows = {}, []
    for row in records:
        path = row['source']
        if path not in definitions:
            raw = Path(path).read_bytes()
            by_name = {}
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            by_name.setdefault(target.id, []).append(node)
            definitions[path] = (hashlib.sha256(raw).hexdigest(), by_name)
        digest, by_name = definitions[path]
        if digest != row['source_sha256']:
            raise ValueError('Saved source changed: ' + path)
        found = by_name.get(row['symbol'], [])
        result = inspect_definition(found[0], row['symbol']) if len(found) == 1 else {
            'status': 'AMBIGUOUS_OR_MISSING_DEFINITION'}
        rows.append(dict(source=path, source_sha256=digest, symbol=row['symbol'],
                         body_sha256=row.get('operation_inventory', {}).get('body_sha256'),
                         type_review=result))
    return dict(schema='operator-candidate-type-audit-v1', records=rows,
                counts=dict(Counter(r['type_review']['status'] for r in rows)),
                scope='Saved source metadata only; integer paths may still affect training. '
                      'No family, bias, correctness or runtime coverage is inferred.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    raw = a.inventory.read_bytes()
    inventory = json.loads(raw)
    if inventory.get('schema') != 'kernel-operation-inventory-v1':
        raise ValueError('Unsupported inventory')
    result = audit(inventory['records'])
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(result['counts'])


if __name__ == '__main__':
    main()
