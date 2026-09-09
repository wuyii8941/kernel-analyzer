import json
import pytest
from scripts.finalize_explicit_output_capture import finalize
from scripts.run_numerical_coverage import sha


def test_missing_records_and_changed_bank_never_verify(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    plan=tmp_path/'plan.json';bank=tmp_path/'bank.json'
    plan.write_text(json.dumps({'cases':[dict(case_id='case',task_id='task',carrier='weight')]}))
    bank.write_text(json.dumps({'states':[{'state_id':'0'},{'state_id':'1'}]}))
    protocol=dict(schema='depthwise-convolution-capture-v1',claim_scope='FIXED_SUITE_UPDATE',
        capture_arguments=['--case-plan',str(plan),'--input-bank',str(bank),'--states','2'],
        source_sha256={str(plan):sha(plan),str(bank):sha(bank)})
    (raw/'family_execution_protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'source_snapshot.json').write_text('{}')
    report=finalize(tmp_path)
    assert report['counts']=={'NOT_CAPTURED':1}
    assert not report['recorded_measurement_complete']
    bank.write_text('{}')
    with pytest.raises(ValueError,match='Frozen input differs'):
        finalize(tmp_path)
