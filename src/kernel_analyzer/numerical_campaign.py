"""Metadata-only coverage planning; no numerical outcome affects eligibility."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reference_scope(method):
    """Describe what a reference comparison identifies, not its verdict."""
    if method == 'AOT_REPLAY':
        return {'comparison': 'REFERENCE_GRAPH_ENDPOINT_SUBSTITUTION',
                'same_local_operands': False,
                'single_kernel_source_attribution': 'NOT_ESTABLISHED',
                'includes_possible_upstream_differences': True}
    if method == 'EXTERNAL_FP32_RECOMPUTE':
        return {'comparison': 'COMMON_OPERAND_EXTERNAL_RECOMPUTE',
                'same_local_operands': True,
                'single_kernel_source_attribution': 'REQUIRES_RUNTIME_BINDING',
                'includes_possible_upstream_differences': False}
    if method == 'REGISTERED_SAME_INPUT_REFERENCE':
        return {'comparison': 'COMMON_INPUT_REGISTERED_REFERENCE',
                'same_local_operands': True,
                'single_kernel_source_attribution': 'SOURCE_AND_RUNTIME_BINDING_REQUIRED',
                'includes_possible_upstream_differences': False}
    return {'comparison': 'UNVERIFIED_REFERENCE', 'same_local_operands': None,
            'single_kernel_source_attribution': 'NOT_ESTABLISHED'}


def plan_release(tasks, declared_cases):
    """Retain every endpoint, including unsupported and ambiguous bindings.

    A declared parameter mapping is required. This deliberately does not infer
    reference semantics or parameter reach from a kernel name.
    """
    cuts = {str(r['task_id']).removeprefix('same-dtype:')
            for r in tasks.get('reference_cut_tasks', [])}
    bindings = {}
    for case in declared_cases:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', case.get('case_id', '')) or case['case_id'] in {'.', '..'}:
            raise ValueError('Unsafe or absent case identity')
        bindings.setdefault(case['task_id'], []).append(case)
    rows = []
    seen = set()
    for task in tasks['rows']:
        key = task['task_id']
        if key in seen:
            raise ValueError('Duplicate endpoint identity: ' + key)
        seen.add(key)
        candidates = bindings.get(key, [])
        status = 'READY_FOR_CAPTURE'
        if task.get('status') != 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT':
            status = 'UNRESOLVED_EXECUTION_BOUNDARY'
        elif not candidates:
            status = 'PARAMETER_MAPPING_UNAVAILABLE'
        elif len(candidates) != 1:
            status = 'AMBIGUOUS_DECLARED_COMPARISON'
        elif not candidates[0].get('carrier'):
            status = 'PARAMETER_MAPPING_UNAVAILABLE'
        else:
            method = candidates[0].get('reference_method')
            if method == 'AOT_REPLAY':
                if task.get('exact_aot_endpoint_id') not in cuts:
                    status = 'REFERENCE_UNAVAILABLE'
            elif method == 'EXTERNAL_FP32_RECOMPUTE':
                if (task.get('implementation_kind') != 'EXTERN'
                        or str(task.get('symbol')).removeprefix('extern_kernels.') not in {'mm', 'bmm', 'addmm'}):
                    status = 'REFERENCE_ADAPTER_REQUIRED'
            else:
                status = 'REFERENCE_ADAPTER_REQUIRED'
        rows.append({'task_id': key, 'implementation_kind': task.get('implementation_kind', 'UNKNOWN'),
                     'symbol': task.get('symbol'), 'status': status,
                     'reference_scope': reference_scope(candidates[0].get('reference_method')) if len(candidates) == 1 else None,
                     'case': candidates[0] if len(candidates) == 1 else None})
    return {'schema': 'training-numerical-coverage-v1', 'denominator_kind': 'RELEASE_ENDPOINTS_NOT_KERNEL_FAMILIES',
            'endpoint_count': len(rows), 'counts': dict(Counter(r['status'] for r in rows)),
            'declared_tasks_absent_from_release': sorted(set(bindings) - seen),
            'selection_uses_numerical_results': False, 'rows': rows}
