import pytest
from scripts.build_reference_reach_inventory import classify, merge_carriers


def test_new_reference_inventory_rejects_changed_adapter():
    from scripts.build_reference_reach_inventory import validate_additional_manifest
    from scripts.run_declared_reference_family import declaration
    d=declaration('ROW_SUM','row_sum_reference','-row-sum-common-input')
    m=dict(reference_family=d['family'],adapter_sha256=d['adapter_sha256'],
           additional_reference_declaration=d)
    validate_additional_manifest(m)
    with pytest.raises(ValueError):
        validate_additional_manifest(dict(m,adapter_sha256='changed'))
    with pytest.raises(ValueError):
        validate_additional_manifest(dict(m,additional_reference_declaration=dict(d,entrypoint_sha256='changed')))
    with pytest.raises(ValueError):
        validate_additional_manifest({})


def test_pointer_binding_not_symbol_alone():
    t=dict(implementation_kind='TRITON',formal_pointer='out1',status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT')
    r=[dict(output_pointer='out0',function_ast_sha256='a')]
    assert classify(t,r,'weight')[0]=='NO_CHECKED_REFERENCE_FOR_THIS_OUTPUT'
    t['formal_pointer']='out0'
    assert classify(t,r,None)[0]=='REFERENCE_PRESENT_PARAMETER_BINDING_REQUIRED'
    assert classify(t,r,'weight')[0]=='REFERENCE_AND_STATIC_PARAMETER_BINDING_PRESENT'
    t['status']='INTERNAL'
    assert classify(t,r,'weight')[0]=='EXPLICIT_INTERNAL_BOUNDARY_BINDING_REQUIRED'


def test_ambiguous_sources_and_external_not_unbiased():
    t=dict(implementation_kind='TRITON',formal_pointer='out0')
    refs=[dict(output_pointer='out0',function_ast_sha256=h) for h in ('a','b')]
    assert classify(t,refs,'w')[0]=='AMBIGUOUS_SOURCE_DEFINITION'
    t['implementation_kind']='EXTERN'
    assert classify(t,refs,'w')[0]=='OTHER_IMPLEMENTATION_REQUIRES_REFERENCE_AUDIT'


def test_mapping_conflicts_and_wrong_sources_cannot_be_hidden_by_order():
    task=dict(task_id='b:1:o', symbol='kernel',exact_aot_endpoint_id='endpoint',
              status='EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT')
    case=dict(task_id='b:1:o',expected_symbol='kernel',exact_aot_endpoint_id='endpoint',carrier='weight')
    target={}
    merge_carriers(target,[case],[task])
    merge_carriers(target,[case],[task])
    assert target=={'b:1:o':'weight'}
    with pytest.raises(ValueError,match='Conflicting'):
        merge_carriers(target,[dict(case,carrier='other')],[task])
    with pytest.raises(ValueError,match='disagrees'):
        merge_carriers({},[dict(case,expected_symbol='other')],[task])
    with pytest.raises(ValueError,match='disagrees'):
        merge_carriers({},[case],[])
