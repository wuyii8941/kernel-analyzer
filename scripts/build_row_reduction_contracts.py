#!/usr/bin/env python3
"""Discover source-checked row square sums without reading numerical outcomes."""
import argparse
import ast
import hashlib
from pathlib import Path
from kernel_analyzer.source_reference_registry import REFERENCES, get_reference
from scripts.run_numerical_coverage import save, sha, read, ROOT


FAMILIES = {family: (spec.check_source, spec.variants, spec.source_filename)
            for family, spec in REFERENCES.items()}


def unique_contracts(sources):
    """Do not bind one release to another source's same-name kernel."""
    accepted = {}
    for source in sources:
        for row in source['rows']:
            if row['status'] != 'SOURCE_CHECKED':
                continue
            previous = accepted.setdefault(row['symbol'], row['contract'])
            identity_key = ('function_semantic_ast_sha256'
                            if 'function_semantic_ast_sha256' in row['contract']
                            else 'function_ast_sha256')
            if previous[identity_key] != row['contract'][identity_key]:
                raise ValueError('Ambiguous same-name kernel; bind one unambiguous release at a time: ' + row['symbol'])
    return accepted


def prepare_source(path):
    """Parse one generated wrapper once for all registered references."""
    path = Path(path)
    text = path.read_text()
    tree = ast.parse(text)
    symbols = set()
    definitions = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions.setdefault(target.id, []).append(node)
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'triton'):
            symbols.update(t.id for t in node.targets if isinstance(t, ast.Name))
    source_digest = hashlib.sha256(text.encode()).hexdigest()
    local_sources = {
        symbol: '\n'.join(ast.unparse(node) for node in definitions[symbol])
        for symbol in sorted(symbols)
    }
    return {
        'source': str(path.resolve()),
        'source_sha256': source_digest,
        'local_sources': local_sources,
    }


def discover_prepared(prepared, family='ROW_SQUARE_SUM'):
    rows = []
    source_digest = prepared['source_sha256']
    for symbol, local_source in prepared['local_sources'].items():
        try:
            # The checker only examines assignments to this symbol. Preserve
            # ALL such assignments, including duplicates/rebindings, while
            # avoiding reparsing the full generated model for every kernel.
            contract = FAMILIES[family][0](local_source, symbol)
            contract['source_sha256'] = source_digest
            rows.append({'symbol': symbol, 'status': 'SOURCE_CHECKED', 'contract': contract})
        except ValueError as exc:
            rows.append({'symbol': symbol, 'status': 'NOT_THIS_REFERENCE_FAMILY', 'reason': str(exc)})
    return {'source': prepared['source'], 'source_sha256': source_digest, 'rows': rows}


def discover(path, family='ROW_SQUARE_SUM'):
    return discover_prepared(prepare_source(path), family)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=FAMILIES, default='ROW_SQUARE_SUM')
    parser.add_argument('--case-plan', type=Path)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--bound-plan-output', type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Use new output under /data1/tzh')
    sources = [discover(path, args.family) for path in args.source]
    if any((args.case_plan, args.tasks, args.bound_plan_output)):
        if not all((args.case_plan, args.tasks, args.bound_plan_output)):
            parser.error('Binding requires case-plan, tasks and bound-plan-output together')
        if args.bound_plan_output.exists() or not args.bound_plan_output.resolve().is_relative_to(Path('/data1/tzh')):
            parser.error('Use new bound plan under /data1/tzh')
        accepted = unique_contracts(sources)
        tasks = {r['task_id']: r for r in read(args.tasks)['rows']}
        bound, unresolved = [], []
        for case in read(args.case_plan)['cases']:
            task = tasks.get(case['task_id'], {})
            if (task.get('symbol') in accepted and task.get('formal_pointer') == accepted[task['symbol']]['output_pointer']
                    and case.get('carrier')):
                bound.append({**case, 'source_case_id': case['case_id'],
                              'case_id': case['case_id'] + get_reference(args.family).case_suffix,
                              'reference_contract_symbol': task['symbol'],
                              'reference_method': 'PARTIAL_REDUCTION_FROM_BOUND_INPUT'})
            else:
                unresolved.append({'case': case, 'reason': 'NO_CHECKED_ROW_REFERENCE_OR_PARAMETER_BINDING'})
        save(args.bound_plan_output, {'cases': bound, 'unresolved': unresolved,
                                     'source_plan_sha256': sha(args.case_plan), 'tasks_sha256': sha(args.tasks),
                                     'source_selection_uses_numerical_results': False})
    save(args.output, {'schema': 'source-checked-row-reduction-v1', 'sources': sources,
                      'reference_family': args.family,
                      'adapter_sha256': sha(ROOT / 'src/kernel_analyzer' / FAMILIES[args.family][2]),
                      'legal_variants': list(FAMILIES[args.family][1]), 'numerical_results_read': False,
                      'scope': 'Template semantics only; not runtime validation or nonzero-bias proof'})
    print({'kernels_examined': sum(len(s['rows']) for s in sources),
           'source_checked': sum(r['status'] == 'SOURCE_CHECKED' for s in sources for r in s['rows'])})
