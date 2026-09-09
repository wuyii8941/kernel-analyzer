"""Resolve recorded comparison scope without inferring semantics from case names."""
import hashlib
import json
from pathlib import Path


def recorded_reference_scope(payload, artifact_path, repository):
    artifact_path, repository = Path(artifact_path).resolve(), Path(repository).resolve()
    if not artifact_path.is_relative_to(repository):
        raise ValueError('Artifact is outside the repository')
    declared = payload.get('reference_comparison_scope')
    result = {'status': 'UNKNOWN_REFERENCE_SCOPE', 'scope': None, 'source': None,
              'single_kernel_bias_proved': False,
              'audit_level': 'RECORDED_PROVENANCE_NOT_INDEPENDENT_RUNTIME_VERIFICATION'}
    if isinstance(declared, dict) and declared.get('comparison'):
        result.update(status='EXPLICIT_ARTIFACT_SCOPE', scope=declared,
                      source={'path': str(artifact_path.relative_to(repository)),
                              'sha256': hashlib.sha256(artifact_path.read_bytes()).hexdigest()})
        return result
    for parent in artifact_path.parents:
        if not parent.is_relative_to(repository):
            break
        plan = parent / 'case_plan.json'
        command_evidence = None
        case_id = payload.get('case_id', '')
        execution = parent / 'execution' / (case_id + '.json') if case_id and Path(case_id).name == case_id else None
        if not plan.is_file() and execution is not None and execution.is_file():
            record = json.loads(execution.read_text())
            command = record.get('command', [])
            flags = ('--case-plan', '--training-bias-profile-v2-output-dir')
            if record.get('returncode') == 0 and all(command.count(flag) == 1 and command.index(flag) + 1 < len(command) for flag in flags):
                linked_plan, output = [(repository / command[command.index(flag) + 1]).resolve() for flag in flags]
                if output == artifact_path.parent and linked_plan.is_relative_to(repository):
                    plan = linked_plan
                    command_evidence = {'path': str(execution.relative_to(repository)),
                                        'sha256': hashlib.sha256(execution.read_bytes()).hexdigest(),
                                        'plan_digest_frozen_by_execution_record': False}
        if plan.is_file():
            data = plan.read_bytes()
            matches = [row for row in json.loads(data).get('cases', [])
                       if row.get('case_id') == payload.get('case_id')]
            if matches:
                source = {'path': str(plan.relative_to(repository)),
                          'sha256': hashlib.sha256(data).hexdigest()}
                if len(matches) != 1:
                    result.update(status='AMBIGUOUS_PLAN_BINDING', source=source)
                    return result
                case = matches[0]
                task_id = payload.get('runtime_boundary', {}).get('task_id')
                if not task_id or task_id != case.get('task_id') or payload.get('carrier') != case.get('carrier'):
                    result.update(status='PLAN_ARTIFACT_SCOPE_MISMATCH', source=source)
                    return result
                method = case.get('reference_method')
                if method == 'AOT_REPLAY':
                    scope = {'comparison': 'REFERENCE_GRAPH_ENDPOINT_SUBSTITUTION',
                             'same_local_operands': False,
                             'includes_possible_upstream_differences': True}
                elif method == 'EXTERNAL_FP32_RECOMPUTE':
                    scope = {'comparison': 'DECLARED_COMMON_OPERAND_EXTERNAL_RECOMPUTE',
                             'pre_invocation_input_capture_verified_by_this_audit': False}
                else:
                    scope = {'comparison': 'SPECIAL_REFERENCE_REQUIRES_ADAPTER_REVIEW',
                             'declared_reference_method': method}
                result.update(status='MATCHING_PLAN_DECLARATION', scope=scope, source=source)
                if command_evidence:
                    result.update(status='COMMAND_LINKED_CURRENT_PLAN_DECLARATION',
                                  execution_record=command_evidence)
                return result
        if parent == repository:
            break
    return result
