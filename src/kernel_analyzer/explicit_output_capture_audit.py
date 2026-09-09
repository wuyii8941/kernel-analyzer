"""Shared record checks for explicitly scoped internal/library substitutions."""
from kernel_analyzer.forward_capture_audit import valid_statistics
from kernel_analyzer.training_numerical_analysis import analyze_artifact

CONTRACTS={
    'depthwise-convolution-capture-v1': (
        'EXTERNAL_CONVOLUTION_OUTPUT_BEFORE_SEPARATE_BIAS',
        'INPUT_ORDER_FP32_ACCUMULATION_BF16_WRITE'),
    'gelu-product-capture-v1': (
        'INTERNAL_GELU_OUTPUT_REPLACEMENT','TANH_GELU_PRODUCT_FP32_BF16_WRITE'),
    'embedding-lookup-forward-capture-v1': (
        'PURE_EMBEDDING_LOOKUP_OUTPUT_REPLACEMENT','DECLARED_SAME_DTYPE_INDEX_SELECT'),
    'grouped-causal-softmax-forward-capture-v1': (
        'SINGLE_GROUPED_CAUSAL_SOFTMAX_OUTPUT_REPLACEMENT',
        'DECLARED_FP32_EXPRESSION_WITH_ORIGINAL_WRITES'),
}


def audit_record(raw,case,protocol,expected_ids,analysis_function=analyze_artifact):
    expected=CONTRACTS.get(protocol.get('schema'))
    scope=raw.get('reference_comparison_scope',{})
    checks=dict(protocol=expected is not None and protocol.get('claim_scope')=='FIXED_SUITE_UPDATE',
        case=raw.get('case_id')==case['case_id'],parameter=raw.get('carrier')==case['carrier'],
        boundary=raw.get('runtime_boundary',{}).get('task_id')==case['task_id'],
        complete=raw.get('status')=='COMPLETE',
        states=bool(expected_ids) and len(set(expected_ids))==len(expected_ids) and raw.get('state_ids')==expected_ids,
        statistics=all(valid_statistics(raw.get('original_coordinate_statistics',{}).get(stage),len(expected_ids))
                       for stage in ('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')),
        actual_write=raw.get('parameter_write_protocol',{}).get('version')=='adamw-readback-v2',
        determinism=raw.get('determinism',{}).get('all_exact') is True,
        reference=expected is not None and scope.get('comparison')==expected[0]
            and scope.get('reference_variant')==expected[1]
            and scope.get('same_local_operands') is True
            and scope.get('includes_possible_upstream_differences') is False
            and scope.get('reference_is_absolute_truth') is False)
    if not all(checks.values()):return dict(status='INVALID_OR_INCOMPLETE',checks=checks,analysis=None)
    declared='full_update_rms' in protocol.get('fixed_suite_margins',{})
    analysis=analysis_function(raw,protocol) if declared else None
    return dict(status='RECORDED_MEASUREMENT_CHECKED',checks=checks,analysis=analysis,
        equivalence_policy_declared=declared,
        equivalence_decision=analysis['equivalence_decision'] if analysis else 'NOT_ASSESSED',
        not_assessed_reason=None if declared else 'EQUIVALENCE_MARGIN_NOT_DECLARED',
        scope='SAVED_RECORD_CHECK; source snapshot and execution identity require separate verification')
