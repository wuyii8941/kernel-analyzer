#!/usr/bin/env python3
"""Describe saved kernel bodies, without inferring semantics from their names."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.scan_registered_reference_pool import checked_sources


def describe(source, symbol):
    assignments = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == symbol for t in n.targets)]
    if len(assignments) != 1:
        return dict(status='AMBIGUOUS_OR_MISSING_DEFINITION')
    call = assignments[0].value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and call.func.attr == 'triton' and len(call.args) >= 2
            and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
        return dict(status='NON_LITERAL_DEFINITION')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        return dict(status='AMBIGUOUS_OR_MISSING_FUNCTION')
    body = ast.Module(body=functions[0].body, type_ignores=[])
    calls = Counter(ast.unparse(n.func) for n in ast.walk(body) if isinstance(n, ast.Call))
    operations = Counter(type(n.op).__name__ for n in ast.walk(body)
                         if isinstance(n, (ast.BinOp, ast.UnaryOp)))
    has_load = bool(calls['tl.load'])
    has_store = bool(calls['tl.store'])
    has_atomic = any(name.startswith('tl.atomic_') for name in calls)
    has_reduction = any(name in ('tl.sum', 'tl.max', 'tl.min') or
                        name.startswith('triton_helpers.max') for name in calls)
    # This is deliberately structural rather than semantic. In particular, a
    # name containing "embedding_backward" does not make a no-load kernel an
    # embedding-gradient calculation.
    meaningful_calls = [name for name in calls if name not in {
        'tl.full', 'tl.program_id', 'tl.store',
    }]
    if (has_store and not has_load and not has_atomic and not has_reduction
            and not operations and not meaningful_calls):
        structural_role = 'OUTPUT_INITIALIZATION_ONLY'
    elif (has_load and has_store and not has_atomic and not has_reduction
          and not operations and not any(name.startswith(('libdevice.', 'tl_math.')) for name in calls)):
        structural_role = 'DATA_MOVEMENT_OR_CAST_ONLY'
    else:
        structural_role = 'NUMERICAL_COMPUTATION_REQUIRES_SEMANTIC_REVIEW'
    return dict(status='BODY_INSPECTED', calls=dict(sorted(calls.items())),
                arithmetic=dict(sorted(operations.items())),
                loop_count=sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(body)),
                has_load=has_load, has_store=has_store, has_atomic=has_atomic,
                has_reduction=has_reduction,
                structural_role=structural_role,
                body_sha256=hashlib.sha256(ast.dump(body).encode()).hexdigest(),
                semantics_verified=False)


def reuse_inventory(records, prior_records):
    """Reuse body analysis only when every frozen source identity is unchanged."""
    prior = {}
    for row in prior_records:
        key = (row['source'], row['symbol'], row['source_sha256'])
        if key in prior:
            raise ValueError('Duplicate prior operation identity')
        operation = row.get('operation_inventory')
        if not isinstance(operation, dict) or operation.get('status') != 'BODY_INSPECTED':
            raise ValueError('Prior operation inventory is incomplete')
        prior[key] = operation
    current = [(row['source'], row['symbol'], row['source_sha256']) for row in records]
    if len(set(current)) != len(current):
        raise ValueError('Duplicate current operation identity')
    if set(current) != set(prior):
        raise ValueError('Prior and current frozen source identities differ')
    return [dict(**row, operation_inventory=prior[key])
            for row, key in zip(records, current)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reuse-inventory', type=Path)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use new output under /data1/tzh')
    raw = a.pool.read_bytes()
    pool = json.loads(raw)
    checked_sources(pool)
    reused_raw = a.reuse_inventory.read_bytes() if a.reuse_inventory else None
    if reused_raw is not None:
        reused = json.loads(reused_raw)
        if reused.get('schema') != 'kernel-operation-inventory-v1':
            raise ValueError('Unsupported prior operation inventory')
        rows = reuse_inventory(pool['records'], reused['records'])
    else:
        sources = {path: Path(path).read_text() for path in checked_sources(pool)}
        # Parse each generated wrapper once, retaining all rebindings so ambiguity
        # cannot disappear when large sources are partitioned for inspection.
        definitions = {}
        for path, source in sources.items():
            by_name = {}
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            by_name.setdefault(target.id, []).append(ast.unparse(node))
            definitions[path] = by_name
        rows = []
        for row in pool['records']:
            local = '\n'.join(definitions[row['source']].get(row['symbol'], []))
            rows.append(dict(**row, operation_inventory=describe(local, row['symbol'])))
    result = dict(schema='kernel-operation-inventory-v1',
                  scope='All saved definition records in input pool; not external-call or runtime coverage',
                  input_sha256=hashlib.sha256(raw).hexdigest(), numerical_results_read=False,
                  definition_records=len(rows), records=rows,
                  counts=dict(Counter(r['operation_inventory']['status'] for r in rows)),
                  reused_inventory_sha256=(hashlib.sha256(reused_raw).hexdigest()
                                           if reused_raw is not None else None),
                  warning='Identical call lists or kernel names do not establish identical mathematical semantics')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
