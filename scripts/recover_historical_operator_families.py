"""Import omitted family evidence without upgrading historical verdicts."""
import argparse
import hashlib
import json
from pathlib import Path


SOURCES = [
    ('CONVOLUTION', 'results/property/tcmp_allop_v1/candidate/gemma3_vision_convolution_population.json',
     'kernel-analyzer-gemma-vision-convolution-population-v1', 'gemma_vision_convolution'),
    ('CONVOLUTION', 'results/property/tcmp_allop_v1/candidate/gemma3_vision_convolution_backward_population.json',
     'kernel-analyzer-gemma-vision-convolution-population-v1', 'gemma_vision_convolution_backward'),
    ('GELU', 'results/property/direct_persistence_v4/heldout/gemma4_random_gelu_backward1860.formation.json',
     'kernel-analyzer-gemma4-v3-formation-v1', None),
]


def recover(root):
    rows = []
    for family, name, schema, case_id in SOURCES:
        path = root/name
        if not path.exists():
            rows.append(dict(family=family, path=name, import_status='MISSING'))
            continue
        raw = path.read_bytes()
        data = json.loads(raw)
        if data.get('schema') != schema or (case_id and data.get('case_id') != case_id):
            raise ValueError('Historical identity differs: '+name)
        records = data.get('states', data.get('records', []))
        if not records or len({r['state_id'] for r in records}) != len(records):
            raise ValueError('Missing or duplicate historical states')
        if family == 'CONVOLUTION':
            gram = data['complete_gram']
            if len(gram) != len(records) or any(len(row) != len(records) for row in gram):
                raise ValueError('Historical Gram dimensions differ')
        rows.append(dict(family=family, path=name, sha256=hashlib.sha256(raw).hexdigest(),
            import_status='HISTORICAL_RECORD_READ_NOT_RERUN', historical_schema=schema,
            historical_status=data.get('status'), state_count=len(records),
            parameter_scope=data.get('declared_parameter_coordinates'),
            evidence_kind='GRADIENT_GRAM' if family == 'CONVOLUTION' else 'LEGACY_FORMATION',
            current_protocol_three_stage_verified=False,
            new_family_discovery=False, bias_recertified=False))
    return dict(schema='historical-operator-family-recovery-v1', records=rows,
        additional_catalogue_families=sorted({r['family'] for r in rows if r['import_status'] != 'MISSING'}),
        exhaustive_repository_audit=False,
        scope='Recover omitted records; no new measurements or upgraded statistical guarantees')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    report = recover(Path(__file__).resolve().parents[1])
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(report['additional_catalogue_families'])


if __name__ == '__main__':
    main()
