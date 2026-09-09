import copy
import json
import pytest
from scripts.audit_family_plan_execution import (
    audit, check_partition, check_reference_translation, sha,
)


def cases():
    return [dict(case_id=str(i),task_id=f'backward:{i}:output',carrier='weight',reference_method='declared') for i in range(3)]


def test_partition_preserves_full_case_definition():
    rows=cases()
    assert check_partition(rows,rows[1:])==['1','2']


@pytest.mark.parametrize('field',['case_id','task_id','carrier','reference_method'])
def test_changed_case_is_not_a_valid_partition(field):
    rows=cases(); part=copy.deepcopy(rows[:1]); part[0][field]='changed'
    with pytest.raises(ValueError): check_partition(rows,part)


def test_empty_duplicate_and_unknown_rejected():
    rows=cases()
    for part in ([],[rows[0],rows[0]],[dict(case_id='new')]):
        with pytest.raises(ValueError): check_partition(rows,part)


def queue_fixture(tmp_path):
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(cases=cases(),unresolved=[dict(reason='needs reference')])))
    part=tmp_path/'part.json'
    part.write_text(json.dumps(dict(cases=cases()[:2],source_plan_sha256=sha(source))))
    queue=tmp_path/'queue'; queue.mkdir()
    metadata=dict(jobs=[dict(root=str(tmp_path/'not_created'),plan=str(part),plan_sha256=sha(part))])
    (queue/'queue_protocol.json').write_text(json.dumps(metadata))
    return source,part,queue,metadata


def test_full_denominator_not_reduced_to_scheduled_subset(tmp_path):
    source,part,queue,_=queue_fixture(tmp_path)
    result=audit(source,[],[queue])
    assert result['eligible_positions']==3
    assert result['scheduled_positions']==2
    assert result['source_unresolved_positions']==1
    assert result['counts']=={'NO_VERIFIABLE_CAPTURE_YET':2,'NOT_SCHEDULED':1}
    assert not result['eligible_measurement_complete']
    assert not result['all_source_positions_supported']


def test_same_case_in_two_jobs_is_not_extra_coverage(tmp_path):
    source,part,queue,metadata=queue_fixture(tmp_path)
    metadata['jobs']*=2
    (queue/'queue_protocol.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='more than once'): audit(source,[],[queue])


def test_modified_plan_refused_even_without_capture(tmp_path):
    source,part,queue,_=queue_fixture(tmp_path)
    part.write_text(part.read_text()+' ')
    with pytest.raises(ValueError,match='digest changed'): audit(source,[],[queue])


def test_unpartitioned_frozen_full_plan_is_supported(tmp_path):
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(cases=cases(),source_plan_sha256='upstream-input-plan')))
    root=tmp_path/'capture'; (root/'raw').mkdir(parents=True)
    (root/'raw/family_execution_protocol.json').write_text(json.dumps(dict(
        capture_arguments=['--case-plan',str(source)],source_sha256={str(source.resolve()):sha(source)})))
    result=audit(source,[root],[])
    assert result['scheduled_positions']==3
    assert result['counts']=={'NO_VERIFIABLE_CAPTURE_YET':3}
    assert not result['eligible_measurement_complete']


def test_other_plan_still_requires_parent_digest(tmp_path):
    source,part,queue,metadata=queue_fixture(tmp_path)
    part.write_text(json.dumps(dict(cases=cases()[:2],source_plan_sha256='unrelated')))
    metadata['jobs'][0]['plan_sha256']=sha(part)
    (queue/'queue_protocol.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='not bound'): audit(source,[],[queue])


def test_capture_protocol_can_bind_legacy_translated_plan_to_parent(tmp_path):
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(cases=cases())))
    part=tmp_path/'translated.json'
    translated=[dict(row,reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                     declared_reference_method=row['reference_method']) for row in cases()[:2]]
    part.write_text(json.dumps(dict(cases=translated)))
    root=tmp_path/'capture'; (root/'raw').mkdir(parents=True)
    (root/'raw/family_execution_protocol.json').write_text(json.dumps(dict(
        capture_arguments=['--case-plan',str(part)],
        source_sha256={str(source.resolve()):sha(source),str(part.resolve()):sha(part)})))
    result=audit(source,[root],[])
    assert result['scheduled_positions']==2
    assert result['counts']=={'NO_VERIFIABLE_CAPTURE_YET':2,'NOT_SCHEDULED':1}


def test_legacy_translated_plan_without_protocol_parent_is_rejected(tmp_path):
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(cases=cases())))
    part=tmp_path/'translated.json'
    translated=[dict(row,reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                     declared_reference_method=row['reference_method']) for row in cases()[:2]]
    part.write_text(json.dumps(dict(cases=translated)))
    root=tmp_path/'capture'; (root/'raw').mkdir(parents=True)
    (root/'raw/family_execution_protocol.json').write_text(json.dumps(dict(
        capture_arguments=['--case-plan',str(part)],
        source_sha256={str(part.resolve()):sha(part)})))
    with pytest.raises(ValueError,match='not bound'):
        audit(source,[root],[])


def test_reference_translation_rejects_any_other_case_change():
    original=cases()[:1]
    translated=[dict(original[0],reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                     declared_reference_method=original[0]['reference_method'])]
    assert check_reference_translation(original,translated)==original
    translated[0]['carrier']='different'
    with pytest.raises(ValueError,match='more than reference'):
        check_reference_translation(original,translated)


def test_recurrence_translation_keeps_unmeasured_denominator(tmp_path):
    rows=[dict(case_id=str(i),task_id=str(i),carrier='weight',expected_symbol='k',
               reference_method='DECAYED_RECURRENCE_COMMON_INPUT',
               reference_output_pointer=f'out_ptr{i}',reference_output_index=i) for i in range(2)]
    source=tmp_path/'source.json'
    source.write_text(json.dumps(dict(cases=rows,contract=dict(symbol='k',output_pointers=['out_ptr0','out_ptr1']))))
    part=tmp_path/'translated.json'
    part.write_text(json.dumps(dict(cases=[dict(rows[0],reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
        declared_reference_method='DECAYED_RECURRENCE_COMMON_INPUT')])))
    root=tmp_path/'capture'; (root/'raw').mkdir(parents=True)
    proto=dict(schema='decayed-recurrence-capture-v1',capture_arguments=['--case-plan',str(part)],
               source_sha256={str(source):sha(source),str(part):sha(part)})
    path=root/'raw/family_execution_protocol.json'; path.write_text(json.dumps(proto))
    result=audit(source,[root],[])
    assert result['counts']=={'NO_VERIFIABLE_CAPTURE_YET':1,'NOT_SCHEDULED':1}
    proto['source_sha256'][str(source)]='wrong'; path.write_text(json.dumps(proto))
    with pytest.raises(ValueError,match='not bound'): audit(source,[root],[])
