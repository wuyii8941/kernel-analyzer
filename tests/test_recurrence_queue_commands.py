import json
from pathlib import Path
import pytest
from scripts.recurrence_queue_commands import build_command,sha


def fixture(tmp_path):
    full=tmp_path/'full.json'; part=tmp_path/'part.json'
    c=dict(case_id='c',expected_symbol='k',reference_output_pointer='out_ptr0',reference_output_index=0)
    full.write_text(json.dumps(dict(contract=dict(symbol='k',output_pointers=['out_ptr0']),cases=[c])))
    part.write_text(json.dumps(dict(source_plan_sha256=sha(full),cases=[c])))
    protocol=dict(schema='decayed-recurrence-capture-v1',source_sha256={str(full):sha(full)},
        capture_arguments=['--case-plan','old','--output-dir','old','--spool-dir','old',
            '--training-bias-profile-v2-output-dir','old','--states','32','--device','cuda:0',
            '--warmup-steps','8'])
    return protocol,full,part


def test_frozen_measurement_arguments_preserved(tmp_path):
    p,f,b=fixture(tmp_path); old=list(p['capture_arguments'])
    cmd=build_command(p,f,b,Path('/data1/tzh/result'),Path('/data1/tzh/spool'))
    for key,value in [('--states','32'),('--device','cuda:0'),('--warmup-steps','8')]:
        assert cmd[cmd.index(key)+1]==value
    assert cmd.count('--case-plan')==1
    assert cmd[cmd.index('--recurrence-plan')+1]==str(f)
    assert p['capture_arguments']==old


def test_unbound_partition_rejected(tmp_path):
    p,f,b=fixture(tmp_path); data=json.loads(b.read_text()); data['source_plan_sha256']='wrong'
    b.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Partition'): build_command(p,f,b,tmp_path/'r',tmp_path/'s')


def test_changed_original_rejected(tmp_path):
    p,f,b=fixture(tmp_path); f.write_text('{}')
    with pytest.raises(ValueError,match='Full recurrence'): build_command(p,f,b,tmp_path/'r',tmp_path/'s')


def test_shared_queue_dispatches_to_same_command(tmp_path):
    from scripts.run_family_plan_queue import build_command as shared, verification_command
    p,f,b=fixture(tmp_path); root=tmp_path/'r'; spool=tmp_path/'s'
    assert shared(p,f,b,root,spool)==build_command(p,f,b,root,spool)
    cmd=verification_command(p,root)
    assert cmd[1].endswith('/finalize_decayed_recurrence.py')
    assert '--baselines-dir' not in cmd


def test_existing_verification_keeps_baselines(tmp_path):
    from scripts.run_family_plan_queue import verification_command
    cmd=verification_command({},tmp_path)
    assert cmd[1].endswith('/finalize_numerical_family.py')
    assert '--baselines-dir' in cmd
