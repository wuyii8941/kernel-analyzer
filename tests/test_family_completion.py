import copy
import hashlib
import json

import numpy as np
import pytest

from scripts import finalize_numerical_family as completion
from test_training_numerical_analysis import PROTOCOL, artifact


def example():
    raw = artifact(np.ones((16, 2)) * .001, np.ones((16, 2)))
    raw['carrier'] = 'weight'
    for key in ('state_ids', 'calibration_state_ids', 'confirmation_state_ids'):
        raw[key] = list(map(str, raw[key]))
    raw['runtime_boundary']['task_id'] = 'backward:1:in_out_ptr0'
    raw['determinism'] = {'all_exact': True}
    raw['reference_comparison_scope'] = {
        'comparison': 'COMMON_OPERAND_SOURCE_CHECKED_RMS_BACKWARD',
        'same_local_operands': True, 'includes_possible_upstream_differences': False,
        'reference_variant': 'FP32_NATIVE',
    }
    for stage in ('LOCAL', 'PARAMETER_GRADIENT'):
        raw['original_coordinate_statistics'][stage] = copy.deepcopy(raw['original_coordinate_statistics']['PARAMETER_WRITE'])
    case = {'case_id': 'synthetic', 'carrier': 'weight', 'task_id': 'backward:1:in_out_ptr0'}
    protocol = dict(PROTOCOL, reference_family='RMS_BACKWARD', variant='FP32_NATIVE')
    return raw, case, protocol


def test_reuses_shared_estimator():
    raw, case, protocol = example()
    record = completion.audit_record(raw, case, protocol, 16)
    assert record['status'] == 'VERIFIED'
    assert record['analysis'] == completion.analyze_artifact(raw, protocol)


def test_same_state_count_does_not_prove_frozen_state_identity():
    raw, case, protocol = example()
    record = completion.audit_record(raw, case, protocol, 16, list(reversed(raw['state_ids'])))
    assert record['status'] == 'INVALID_OR_INCOMPLETE'
    assert record['analysis'] is None


def test_protocol_pinned_analyzer_can_run_from_exact_source_snapshot():
    root = completion.ROOT
    paths = [
        root / 'src/kernel_analyzer/analysis_result.py',
        root / 'src/kernel_analyzer/training_equivalence.py',
        root / 'src/kernel_analyzer/training_numerical_analysis.py',
    ]
    snapshot = {str(path.resolve()): path.read_text() for path in paths}
    frozen = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in snapshot.items()}
    frozen_analyze = completion._analysis_from_snapshot(snapshot, frozen)
    raw, _, protocol = example()
    assert frozen_analyze(raw, protocol) == completion.analyze_artifact(raw, protocol)


@pytest.mark.parametrize('error', ['parameter', 'boundary', 'states', 'statistics', 'reference', 'determinism'])
def test_bad_record_cannot_issue_equivalence(error):
    raw, case, protocol = example()
    if error == 'parameter':
        raw['carrier'] = 'another_weight'
    elif error == 'boundary':
        raw['runtime_boundary']['task_id'] = 'backward:2:in_out_ptr0'
    elif error == 'states':
        raw['state_ids'][-1] = raw['state_ids'][0]
    elif error == 'statistics':
        raw['original_coordinate_statistics']['LOCAL'].pop()
    elif error == 'reference':
        raw['reference_comparison_scope']['same_local_operands'] = False
    else:
        raw.pop('determinism')
    record = completion.audit_record(raw, case, protocol, 16)
    assert record['status'] == 'INVALID_OR_INCOMPLETE'
    assert record['analysis'] is None


def test_missing_positions_stay_in_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, 'ROOT', tmp_path)
    raw, case, protocol = example()
    (tmp_path / 'src/kernel_analyzer').mkdir(parents=True)
    sources = {}
    for name in ('training_numerical_analysis.py', 'training_equivalence.py'):
        path = tmp_path / 'src/kernel_analyzer' / name
        path.write_text('# frozen fixture\n')
        sources[str(path)] = path.read_text()
    plan = tmp_path / 'cases.json'
    plan.write_text(json.dumps({'cases': [case, dict(case, case_id='missing', task_id='backward:2:in_out_ptr0')]}))
    hashes = {p: hashlib.sha256(text.encode()).hexdigest() for p, text in sources.items()}
    hashes[str(plan)] = completion.sha(plan)
    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'states': [{'state_id': i} for i in range(16)]}))
    hashes[str(bank)] = completion.sha(bank)
    protocol.update(capture_arguments=['--case-plan', str(plan), '--states', '16', '--input-bank', str(bank)], source_sha256=hashes)
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw/family_execution_protocol.json').write_text(json.dumps(protocol))
    (tmp_path / 'source_snapshot.json').write_text(json.dumps(sources))
    (tmp_path / 'raw/synthetic.json').write_text(json.dumps(raw))
    report = completion.finalize(tmp_path)
    assert report['declared_positions'] == 2
    assert report['counts'] == {'VERIFIED': 1, 'NOT_CAPTURED': 1}
    assert not report['measurement_complete']
    assert not report['verifies_full_research_plan']
    assert report['capture_cost']['status'] == 'NOT_RECORDED'
    plan.write_text('{}')
    with pytest.raises(ValueError, match='Case plan differs'):
        completion.finalize(tmp_path)
