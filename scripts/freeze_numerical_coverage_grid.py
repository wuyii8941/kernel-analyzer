#!/usr/bin/env python3
"""Apply the same coverage planner to the existing four-model release grid."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'qwen': ('qwen', '/data1/tzh/models/Qwen/Qwen3-1.7B'),
    'phi4': ('phi', '/data1/tzh/models/microsoft/Phi-4-mini-instruct'),
    'deepseek8b': ('deepseek8', '/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'),
    'mamba': ('mamba', '/data1/tzh/models/state-spaces/mamba-130m-hf'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = a.output.resolve()
    if not output.is_relative_to(Path('/data1/tzh')):
        p.error('Output must stay under /data1/tzh')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for model, (architecture, checkpoint) in MODELS.items():
        for length in (64, 128, 256):
            cell = f'{model}_seq{length}'
            release = ROOT / f'results/coverage/runtime_releases/{cell}_r1'
            bank = ROOT / f'results/coverage/{cell}_input_bank.json'
            required = [release / 'same_dtype_tasks.json.gz', bank, Path(checkpoint)]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                rows.append({'cell': cell, 'status': 'SOURCE_UNAVAILABLE', 'missing': missing})
                continue
            command = [sys.executable, str(ROOT / 'scripts/run_numerical_coverage.py'), 'freeze',
                       '--output', str(output / cell), '--release', str(release),
                       '--architecture', architecture, '--model', checkpoint, '--input-bank', str(bank),
                       '--carrier-registry', str(ROOT / 'results/property/bias_formation/hotspot_search/multishape_backward_carriers.json'),
                       '--registry-model', model, '--sequence-length', str(length)]
            with (output / (cell + '.log')).open('x') as f:
                result = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            row = {'cell': cell, 'returncode': result.returncode, 'status': 'PLANNING_FAILED'}
            if result.returncode == 0:
                coverage = json.loads((output / cell / 'coverage.json').read_text())
                row.update(status='PLANNED_NOT_MEASURED', endpoint_count=coverage['endpoint_count'], counts=coverage['counts'])
            rows.append(row)
    with (output / 'grid.json').open('x') as f:
        json.dump({'schema': 'numerical-coverage-grid-v1', 'rows': rows,
                   'claim_scope': 'EXISTING_RELEASE_ENDPOINTS_NOT_UNIVERSAL_KERNEL_SUPPORT'}, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
