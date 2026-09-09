"""Check every definition in the existing source pool against reviewed NLL code."""
import argparse
import ast
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from scripts.scan_registered_reference_pool import checked_sources
from kernel_analyzer.selected_nll_source import check_source


def scan(pool):
    sources = checked_sources(pool)
    definitions = {}
    for path in sources:
        grouped = {}
        for node in ast.parse(Path(path).read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        grouped.setdefault(target.id, []).append(ast.unparse(node))
        definitions[path] = grouped
    records = []
    seen = set()
    for row in pool['records']:
        key = (row['source'], row['symbol'])
        if key in seen:
            raise ValueError('Duplicate source definition in pool')
        seen.add(key)
        record = dict(source=key[0], symbol=key[1], source_sha256=row['source_sha256'],
                      runtime_measurement_complete=False)
        try:
            contract = check_source('\n'.join(definitions[key[0]].get(key[1], [])), key[1])
            record.update(status='SOURCE_CHECKED', contract=contract)
        except ValueError as exc:
            record.update(status='NOT_SUPPORTED_BY_THIS_REFERENCE', reason=str(exc))
        records.append(record)
    return dict(schema='selected-nll-source-pool-v1', records=records,
                selection='ALL_INPUT_DEFINITIONS_WITHOUT_NUMERICAL_OUTCOMES',
                source_matches=sum(r['status']=='SOURCE_CHECKED' for r in records),
                runtime_measurement_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        raise ValueError('New output under /data1/tzh required')
    result = scan(read(args.pool))
    result['input_sha256'] = sha(args.pool)
    result['source_sha256'] = {str(p.resolve()):sha(p) for p in
        (Path(__file__), Path('src/kernel_analyzer/selected_nll_source.py'))}
    save(args.output, result)
    print(dict(definitions=len(result['records']), source_matches=result['source_matches']))


if __name__ == '__main__':
    main()
