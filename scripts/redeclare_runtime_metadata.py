#!/usr/bin/env python3
"""Build a new preflight declaration; never overwrite or impersonate lost metadata.

The existing runtime census/trace remains unchanged. Full runtime identity must
still pass during capture; agreeing phase order alone does not certify it.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha


def checked_metadata(release, aot_path, bank_path):
    aot, bank = read(aot_path), read(bank_path)
    first = bank.get('states', bank.get('records'))[0]
    tokens = first.get('input_ids', first.get('token_ids'))
    token_digest = hashlib.sha256(json.dumps(tokens, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if token_digest != aot['input']['token_ids_sha256'] or len(tokens) != aot['input']['sequence_length']:
        raise ValueError('Recorded AOT input and frozen bank differ')
    modules = []
    for directory in (release / 'trace').iterdir():
        match = re.fullmatch(r'model__(\d+)_(forward|backward)_segment(\d+)_executed', directory.name)
        if match:
            source = directory / 'output_code.py'
            modules.append({'order': int(match[1]), 'phase': match[2].upper(),
                            'source': str(source.resolve()), 'source_sha256': sha(source)})
    modules.sort(key=lambda row: row['order'])
    expected = [g['phase'].upper() for g in aot['capture']['graphs']]
    if not modules or [row['phase'] for row in modules] != expected:
        raise ValueError('Frozen trace and recorded AOT phase order differ')
    return {'schema': 'redeclaration-from-existing-input-and-trace-v1',
            'input': aot['input'], 'modules': modules,
            'original_metadata_recovered': False, 'runtime_identity_verified': False}


def redeclare(release, aot_path, bank_path, output):
    if output.exists() or not output.resolve().is_relative_to(Path('/data1/tzh')):
        raise ValueError('Use a new declaration directory under /data1/tzh')
    metadata = checked_metadata(release, aot_path, bank_path)
    output.mkdir(parents=True)
    for path in release.iterdir():
        if path.name != 'capture.json':
            (output / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    save(output / 'capture.json', metadata)
    save(output / 'redeclaration_provenance.json', {
        'source_release': str(release.resolve()), 'original_metadata_preserved': True,
        'original_capture_file_sha256': sha(release / 'capture.json'),
        'aot_sha256': sha(aot_path), 'input_bank_sha256': sha(bank_path),
        'reason': 'New explicit preflight metadata; not reconstruction of the original capture bytes',
        'identity_checks_relaxed': False})
    return {'phases': [m['phase'] for m in metadata['modules']], 'output': str(output), 'runtime_identity_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release', 'aot', 'input-bank', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(redeclare(args.release, args.aot, args.input_bank, args.output))
