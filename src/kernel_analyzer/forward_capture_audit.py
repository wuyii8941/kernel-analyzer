"""Audit saved single-output forward measurements without inventing margins."""
from kernel_analyzer.training_numerical_analysis import analyze_artifact
import math


def valid_statistics(rows, count):
    if not isinstance(rows, list) or len(rows) != count:
        return False
    for row in rows:
        try:
            x, b, a = [float(row[key]) for key in
                       ('effect_energy', 'repair_energy', 'effect_repair_inner_product')]
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not all(math.isfinite(v) for v in (x, b, a)) or x < 0 or b < 0:
            return False
        bound = math.sqrt(x) * math.sqrt(b)
        if abs(a) > bound + 1e-10 * max(bound, 1e-300):
            return False
    return True


def audit_record(raw, case, protocol, expected_ids):
    scope = raw.get('reference_comparison_scope', {})
    checks = dict(
        protocol=protocol.get('schema') == 'residual-rms-forward-capture-v1'
            and protocol.get('contrast_id') == 'SINGLE_FORWARD_OUTPUT_REPLACEMENT',
        case=raw.get('case_id') == case['case_id'],
        parameter=raw.get('carrier') == case['carrier'],
        boundary=raw.get('runtime_boundary', {}).get('task_id') == case['task_id'],
        complete=raw.get('status') == 'COMPLETE',
        states=bool(expected_ids) and len(set(expected_ids)) == len(expected_ids)
            and raw.get('state_ids') == expected_ids,
        statistics=all(valid_statistics(raw.get('original_coordinate_statistics', {}).get(stage),
                       len(expected_ids)) for stage in ('LOCAL', 'PARAMETER_GRADIENT', 'PARAMETER_WRITE')),
        actual_write=raw.get('parameter_write_protocol', {}).get('version') == 'adamw-readback-v2',
        determinism=raw.get('determinism', {}).get('all_exact') is True,
        reference=scope.get('comparison') == 'SINGLE_FORWARD_OUTPUT_REPLACEMENT'
            and scope.get('same_local_operands') is True
            and scope.get('includes_possible_upstream_differences') is False
            and scope.get('complete_multi_output_implementation_replacement') is False)
    if not all(checks.values()):
        return dict(status='INVALID_OR_INCOMPLETE', checks=checks, analysis=None)
    declared = 'full_update_rms' in protocol.get('fixed_suite_margins', {})
    # The common equivalence analyzer requires a margin. Missing policy is not
    # evidence that valid captured tensors are malformed, nor permission to
    # supply a new threshold after observing them.
    analysis = analyze_artifact(raw, protocol) if declared else None
    return dict(status='RECORDED_MEASUREMENT_CHECKED', checks=checks, analysis=analysis,
        equivalence_policy_declared=declared,
        equivalence_decision=analysis.get('equivalence_decision', 'NOT_ASSESSED') if analysis else 'NOT_ASSESSED',
        not_assessed_reason=None if declared else 'EQUIVALENCE_MARGIN_NOT_DECLARED',
        scope='SAVED_RECORD_CHECK_NOT_INDEPENDENT_EXECUTION_VERIFICATION')
