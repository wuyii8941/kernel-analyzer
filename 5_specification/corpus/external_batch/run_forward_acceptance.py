"""Separate forward-only entry points; never infer an autograd promise from a task name."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import run as batch
from kernel_spec import registry
from kernel_spec.numeric_io import encode_finite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    (args.out / 'runner.py').write_text(Path(__file__).read_text())
    (args.out / 'shared_runner.py').write_text(Path(batch.__file__).read_text())
    source_root = batch.ROOT / selection['source_root']
    metadata_path = source_root / f"{selection['package']}-{selection['version']}.dist-info"
    metadata = (metadata_path / 'METADATA').read_text()
    if f"\nVersion: {selection['version']}\n" not in metadata:
        raise ValueError('installed package differs from selected version')
    if torch.__version__ != '2.10.0+cu128':
        raise ValueError('standard reference version differs from selection')
    report = {'selection': selection, 'torch': torch.__version__, 'package_metadata': metadata,
              'license': (metadata_path / 'LICENSE').read_text(), 'source_modules': {}, 'cases': [],
              'formal_binding': 'not_established', 'bias_verdict': 'not_established',
              'gradient_acceptance': 'not_tested_forward_entry_points_only'}
    for case in selection['cases']:
        result = {**case, 'samples': []}
        report['cases'].append(result)
        print('START', case['id'], flush=True)
        try:
            path = source_root / case['module']
            source = path.read_text()
            report['source_modules'][case['module']] = source
            spec = importlib.util.spec_from_file_location(case['id'], path)
            module = importlib.util.module_from_spec(spec)
            # Original standalone module: no source rewriting or dependency stubs.
            spec.loader.exec_module(module)
            candidate_fn = getattr(module, case['function'])
            standard_fn = getattr(F, case['function'])
            kwargs = batch.loss_options(case)
            for shape in selection['shapes']:
                for seed in selection['seeds']:
                    sample = {'shape': shape, 'seed': seed}
                    result['samples'].append(sample)
                    try:
                        x, target = batch.make_inputs(case['op'], shape, seed)
                        before = (x.clone(), target.clone())
                        sample['inputs'] = {'x': batch.record(x), 'target': batch.record(target)}
                        with torch.no_grad():
                            standard = standard_fn(x, target, **kwargs)
                            candidate = candidate_fn(x, target, **kwargs)
                        torch.cuda.synchronize()
                        sample['standard'] = {'out': batch.record(standard)}
                        sample['candidate'] = {'out': batch.record(candidate)}
                        sample['inputs_unchanged'] = torch.equal(x, before[0]) and torch.equal(target, before[1])
                        if not sample['inputs_unchanged']:
                            raise ValueError('forward call modified a read-only input')
                        sample['standard_comparison'] = batch.standard_comparison({'out': candidate}, {'out': standard})
                        query = {'op': 'aten.'+case['function'], 'args': [batch.record(x), batch.record(target)],
                                 'kwargs': kwargs}
                        r = registry.query(case['contract'], case['profile'], query)
                        reference = {'contract': case['contract'], 'profile': case['profile'],
                                     'query': query, 'outcome': r.outcome, 'note': r.note}
                        sample['mathematical'] = reference
                        if r.outcome == 'value':
                            value = r.fields['out']
                            reference['value'] = batch.map_values(value, encode_finite)
                            reference['candidate_differences'] = batch.reference_comparison(candidate, value)
                            reference['standard_differences'] = batch.reference_comparison(standard, value)
                        sample['status'] = 'compared' if r.outcome == 'value' else 'reference_unavailable'
                    except Exception as exc:
                        sample.update(status='execution_or_reference_error', error=f'{type(exc).__name__}: {exc}')
            if path.read_text() != source:
                raise ValueError('source module changed during execution')
            result['status'] = 'compared' if all(s['status']=='compared' for s in result['samples']) else 'incomplete'
        except Exception as exc:
            result.update(status='import_or_source_error', error=f'{type(exc).__name__}: {exc}')
        (args.out / (case['id']+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print('END', case['id'], result['status'], result.get('error',''), flush=True)


if __name__ == '__main__':
    main()
