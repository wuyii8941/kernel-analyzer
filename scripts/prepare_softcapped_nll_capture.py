"""Bind reviewed softcap outputs under an existing single-parameter protocol.

Source and task identity come from saved execution records. Parameter reach
is deliberately left for the runtime preflight, not inferred from names.
"""
import argparse
import ast
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha, ROOT
from scripts.bind_gelu_product_tasks import bind
from kernel_analyzer.softcapped_nll_source import check_source


def prepare(capture_path, tasks_path, trainability_path):
    capture = read(capture_path)
    sources = []
    paths = []
    for module in capture['modules']:
        path = Path(module['captured_source'])
        if sha(path) != module['sha256']:
            raise ValueError('Captured source changed')
        paths.append(path)
        text = path.read_text()
        symbols = set()
        for node in ast.parse(text).body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'triton'):
                symbols.update(t.id for t in node.targets if isinstance(t, ast.Name))
        rows = []
        for symbol in sorted(symbols):
            try:
                contract = check_source(text, symbol)
                rows.append(dict(symbol=symbol, status='SOURCE_CHECKED', contract=contract))
            except ValueError as exc:
                rows.append(dict(symbol=symbol, status='NOT_THIS_REFERENCE_FAMILY', reason=str(exc)))
        sources.append(dict(source=str(path.resolve()), source_sha256=sha(path), rows=rows))
    # Existing binding validates unique tasks, backward Triton output and the
    # declared trainability scope. No new numerical selection rule is needed.
    plan = bind(dict(sources=sources), read(tasks_path), read(trainability_path))
    for case in plan['cases']:
        case['case_id'] = 'softcap_'+case['task_id'].replace(':', '_')
        case['reference_method'] = 'SOFTCAPPED_NLL_COMMON_INPUT'
    plan['schema'] = 'softcapped-nll-task-plan-v1'
    plan['source_selection_uses_numerical_results'] = False
    plan['sources'] = sources
    dependencies = [capture_path, tasks_path, trainability_path, Path(__file__),
                    ROOT/'scripts/bind_gelu_product_tasks.py',
                    ROOT/'src/kernel_analyzer/softcapped_nll_source.py',
                    ROOT/'src/kernel_analyzer/selected_nll_source.py', *paths]
    plan['source_sha256'] = {str(p.resolve()): sha(p) for p in dependencies}
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'tasks', 'trainability', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('New output under /data1/tzh required')
    result = prepare(a.capture, a.tasks, a.trainability)
    save(a.output, result)
    print(dict(cases=len(result['cases']), unmatched=result['unmatched_checked_symbols']))


if __name__ == '__main__':
    main()
