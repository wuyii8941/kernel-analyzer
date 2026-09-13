import hashlib
import json
import pytest
from scripts import verify_same_path_training_attribution as checker


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    root=tmp_path/'new';old=tmp_path/'old'
    monkeypatch.setattr(checker,'HISTORICAL',old)
    source=tmp_path/'source';source.write_text('test fixture, not measured data')
    protocol=dict(schema='same-path-training-attribution-v1',conditions=['OFF','ON'],
                  source_sha256={str(source):checker.sha(source)},stream_count=8,steps=1,
                  evaluation_steps=[0,1],evaluation_states=2,material_margin=.01)
    save(root/'protocol.json',protocol)
    for i in range(8):
        records=[]
        for mode,value in [('OFF',1.05),('ON',1.0)]:
            checkpoint=root/'checkpoints'/f'{i}_{mode}.pt';checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(b'synthetic fixture')
            record=dict(status='COMPLETE',condition=mode,stream_index=i,
                protocol_sha256=checker.sha(root/'protocol.json'),training_steps=1,
                training_loss=[1.0],evaluation_loss_by_step={'0':[2.,2.],'1':[value,value]},
                checkpoint=str(checkpoint),checkpoint_sha256=checker.sha(checkpoint),
                exact_endpoint_reproduction=True,reevaluated_loss=[value,value],
                final_parameter_sha256='fixture',training_steps_per_second_with_evaluation_overhead=1.)
            save(root/'runs'/f'stream_{i:02d}_{mode}.json',record)
            records.append(dict(record,condition='ADAMW8BIT_COMPENSATED_BLOCK256' if mode=='ON' else 'ADAMW8BIT_BLOCK256'))
        save(old/'streams'/f'stream_{i:02d}.json',dict(records=records))
    return root


def test_complete_records_are_recomputed(experiment):
    result=checker.verify(experiment)
    assert result['status']=='VERIFIED'
    assert result['primary']['mean']==pytest.approx(.05)


def test_missing_is_not_completion(experiment):
    (experiment/'runs/stream_00_OFF.json').unlink()
    assert checker.verify(experiment)['status']=='INCOMPLETE'


def test_changed_checkpoint_is_invalid(experiment):
    (experiment/'checkpoints/0_ON.pt').write_bytes(b'changed')
    assert checker.verify(experiment)['status']=='INVALID'
