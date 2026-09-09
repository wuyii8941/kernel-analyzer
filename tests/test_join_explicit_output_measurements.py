from pathlib import Path
import pytest
from scripts import join_explicit_output_measurements as module


def test_grouped_causal_softmax_is_joined_by_shared_family_path():
    assert module.FAMILIES['grouped-causal-softmax-forward-capture-v1'] == \
        'GROUPED_CAUSAL_SOFTMAX'


def test_join_uses_release_identity_and_no_bias_upgrade(monkeypatch,tmp_path):
    release=tmp_path/'release'
    tasks=(release/'same_dtype_tasks.json.gz').resolve()
    protocol={'schema':'depthwise-convolution-capture-v1','source_sha256':{str(tasks):'hash'}}
    monkeypatch.setattr(module,'read',lambda p:protocol)
    monkeypatch.setattr(module,'sha',lambda p:'hash')
    monkeypatch.setattr(module,'validate_retained_measurements',lambda i:0)
    record=dict(task_id='forward:1',status='RECORDED_MEASUREMENT_CHECKED',
                raw_path='/data1/tzh/raw.json',raw_sha256='rawhash',equivalence_decision='NOT_ASSESSED')
    monkeypatch.setattr(module,'finalize',lambda r:dict(records=[record],counts={record['status']:1}))
    inventory={'records':[dict(release=str(release),task_id='forward:1',runtime_measurement_status='NOT_ASSESSED')]}
    result=module.apply(inventory,release,tmp_path/'capture')
    row=result['records'][0]
    assert row['reference_candidates'][0]['family']=='DEPTHWISE_CONV1D'
    assert row['measurement_evidence']['equivalence_decision']=='NOT_ASSESSED'
    assert 'reference_candidates' not in inventory['records'][0]
    with pytest.raises(ValueError,match='release'):
        module.apply(inventory,tmp_path/'other',tmp_path/'capture')


def test_indexed_join_requires_runtime_recheck(monkeypatch,tmp_path):
    from scripts import finalize_indexed_accumulation as indexed
    from scripts import finalize_numerical_family as common
    release=tmp_path/'release'
    tasks=(release/'same_dtype_tasks.json.gz').resolve()
    protocol={'schema':'indexed-accumulation-capture-v1','source_sha256':{str(tasks):'hash'}}
    monkeypatch.setattr(module,'read',lambda p:protocol)
    monkeypatch.setattr(module,'sha',lambda p:'hash')
    monkeypatch.setattr(module,'validate_retained_measurements',lambda i:0)
    calls=[]
    monkeypatch.setattr(indexed,'verify_runtime',lambda p:calls.append('runtime') or {'checked':True})
    monkeypatch.setattr(common,'finalize',lambda p:dict(records=[dict(task_id='task',status='VERIFIED')],counts={'VERIFIED':1}))
    inventory={'records':[dict(release=str(release),task_id='task',runtime_measurement_status='NOT_ASSESSED')]}
    result=module.apply(inventory,release,tmp_path/'capture')
    assert calls==['runtime']
    assert result['records'][0]['reference_candidates'][0]['family']=='INDEXED_ROW_ACCUMULATION'
    def reject(p):raise ValueError('Runtime source changed')
    monkeypatch.setattr(indexed,'verify_runtime',reject)
    with pytest.raises(ValueError,match='Runtime'):
        module.apply(inventory,release,tmp_path/'capture')
