"""Merge audited reference labels for reporting, never infer a bias mechanism.

The catalogue is an organizational convention, not exhaustive kernel support.
Unknown labels and multiply classified positions remain explicit.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

FAMILIES = {
    'LINEAR': '矩阵乘法与线性层', 'NORMALIZATION': 'Normalization',
    'SOFTMAX': 'Softmax', 'CROSS_ENTROPY': 'Cross-entropy / fused loss',
    'SILU_GATING': 'SiLU 与门控乘法', 'SOFTPLUS': 'Softplus',
    'RECURRENCE': '状态递推 / scan', 'ROTARY': 'Rotary / RoPE',
    'REDUCTION': '独立求和与平方和归约', 'INDEXED_ACCUMULATION': '索引梯度累加',
    'GELU': 'GELU', 'CONVOLUTION': '卷积（含逐通道与视觉卷积）',
    'EMBEDDING': 'Embedding lookup', 'SELECTION': 'Top-k / sort selection',
    'OPTIMIZER_UPDATE': 'Optimizer parameter and moment update',
    'FUSED_ATTENTION': 'Fused causal attention',
    'ELEMENTWISE_BIAS': '按通道偏置加法',
}
ALIASES = {
    'INDEXED_ROW_ACCUMULATION': 'INDEXED_ACCUMULATION',
    'DEPTHWISE_CONV1D': 'CONVOLUTION', 'GELU_PRODUCT': 'GELU',
    'RMS_BACKWARD': 'NORMALIZATION', 'RMS_SIMPLE_BACKWARD': 'NORMALIZATION',
    'RESIDUAL_RMS_FORWARD': 'NORMALIZATION',
    'RESIDUAL_RMS_FORWARD_NORMALIZED': 'NORMALIZATION',
    'RMS_FORWARD_NORMALIZED': 'NORMALIZATION',
    'SOFTMAX_BACKWARD': 'SOFTMAX', 'GROUPED_CAUSAL_SOFTMAX': 'SOFTMAX',
    'GROUPED_CAUSAL_SOFTMAX_PROBABILITY': 'SOFTMAX',
    'SCALED_MASKED_SOFTMAX': 'SOFTMAX',
    'SILU_BACKWARD': 'SILU_GATING', 'SELECTED_SILU_PRODUCT': 'SILU_GATING',
    'SOFTPLUS_BIAS_BACKWARD': 'SOFTPLUS', 'DECAYED_RECURRENCE': 'RECURRENCE',
    'CONTINUED_RECURRENCE': 'RECURRENCE', 'SEGMENTED_RECURRENCE_FIRST': 'RECURRENCE',
    'FORWARD_STATE_RECURRENCE': 'RECURRENCE',
    'FORWARD_STATE_RECURRENCE_FINAL': 'RECURRENCE',
    'GROUPED_ROTARY_BACKWARD': 'ROTARY',
    'ATTENTION_POSITION_SCALING': 'ROTARY',
    'EMBEDDING_LOOKUP': 'EMBEDDING',
    'EXPONENTIAL_WEIGHTED_REDUCTION': 'REDUCTION',
    'GATED_CONV_GRADIENT': 'CONVOLUTION',
    'GELU_PRODUCT_BACKWARD': 'GELU', 'GELU_FORWARD_PRODUCT': 'GELU',
    'CHANNEL_BIAS_ADD': 'ELEMENTWISE_BIAS',
    'SELECTION_TOPK_SORT': 'SELECTION',
    'SELECTED_NLL': 'CROSS_ENTROPY', 'SELECTED_NLL_BACKWARD': 'CROSS_ENTROPY',
    'SOFTCAPPED_NLL_BACKWARD': 'CROSS_ENTROPY',
    'ROW_SUM': 'REDUCTION', 'ROW_SQUARE_SUM': 'REDUCTION',
    'normalization/reduction': 'NORMALIZATION', 'normalization backward': 'NORMALIZATION',
    'softmax backward': 'SOFTMAX', 'lm_head backward matrix multiplication': 'LINEAR',
    'attention projection backward': 'LINEAR',
    'fused cross entropy and weight-gradient accumulation': 'CROSS_ENTROPY',
}

VALID_MEASUREMENT_STATUSES = {'VERIFIED', 'RECORDED_MEASUREMENT_CHECKED'}


def support_stage(row):
    """Return the strongest evidenced support stage for one saved position."""
    runtime = row.get('runtime_measurement_status', 'NOT_ASSESSED_BY_THIS_INVENTORY')
    if runtime in VALID_MEASUREMENT_STATUSES:
        return 'VALID_MEASUREMENT_COMPLETED'
    references = row.get('reference_candidates', [])
    if references and row.get('carrier'):
        return 'REFERENCE_AND_TRAINING_BINDING_READY_NOT_VALIDLY_MEASURED'
    if references:
        return 'REFERENCE_AVAILABLE_TRAINING_BINDING_REQUIRED'
    eligibility = row.get('eligibility')
    if eligibility == 'OTHER_IMPLEMENTATION_REQUIRES_REFERENCE_AUDIT':
        return 'ORDINARY_IMPLEMENTATION_REFERENCE_AUDIT_REQUIRED'
    if eligibility == 'AMBIGUOUS_SOURCE_DEFINITION':
        return 'AMBIGUOUS_SOURCE_DEFINITION'
    return 'IDENTIFIED_REFERENCE_ADAPTER_REQUIRED'


def summarize(records, roles, historical_records=(), additional_evidence=()):
    groups = defaultdict(list)
    seen, unknown, ambiguous = set(), Counter(), []
    for row in records:
        identity = (row['release'], row['task_id'])
        if identity in seen:
            raise ValueError('Duplicate release-qualified position')
        seen.add(identity)
        labels = {r['family'] for r in row.get('reference_candidates', [])}
        mapped = {ALIASES[label] for label in labels if label in ALIASES}
        for label in labels-ALIASES.keys(): unknown[label] += 1
        if len(mapped) == 1 and labels <= ALIASES.keys():
            groups[next(iter(mapped))].append(row)
        elif len(mapped) > 1:
            ambiguous.append(dict(release=identity[0], task_id=identity[1], families=sorted(mapped)))
    role_counts = Counter()
    for row in roles:
        label = row.get('semantic_family', 'UNKNOWN')
        if label in ALIASES: role_counts[ALIASES[label]] += 1
        else: unknown['ROLE::'+label] += 1
    historical=defaultdict(list)
    historical_seen=set()
    for row in historical_records:
        if row.get('import_status')!='HISTORICAL_RECORD_READ_NOT_RERUN': continue
        family=row['family']
        if family not in FAMILIES: raise ValueError('Unknown historical operator family')
        identity=row['sha256']
        if identity in historical_seen: raise ValueError('Repeated historical artifact')
        historical_seen.add(identity)
        historical[family].append(row)
    evidence=defaultdict(list)
    evidence_seen=set()
    for row in additional_evidence:
        family=row['family']
        if family not in FAMILIES: raise ValueError('Unknown additional-evidence operator family')
        identity=(row['artifact_sha256'],row['evidence_kind'])
        if identity in evidence_seen: raise ValueError('Repeated additional evidence')
        evidence_seen.add(identity)
        evidence[family].append(row)
    return dict(catalogue_family_count=len(FAMILIES), input_positions=len(records),
        families=[dict(family_id=k, label=label, classified_positions=len(groups[k]),
                       support_stage_counts=dict(Counter(support_stage(r) for r in groups[k])),
                       recorded_runtime_status_counts=dict(Counter(
                           r['runtime_measurement_status'] for r in groups[k])),
                       valid_measurement_implementation_kind_counts=dict(Counter(
                           r.get('implementation_kind', 'UNDECLARED') for r in groups[k]
                           if support_stage(r) == 'VALID_MEASUREMENT_COMPLETED')),
                       historical_role_records=role_counts[k],
                       additional_historical_artifacts=historical[k],
                       historical_artifacts_are_current_protocol_verification=False,
                       additional_measurement_evidence=evidence[k]) for k, label in FAMILIES.items()],
        families_in_supplied_reference_inventory=sum(bool(groups[k]) for k in FAMILIES),
        families_in_supplied_roles=sum(bool(role_counts[k]) for k in FAMILIES),
        support_stage_counts=dict(Counter(support_stage(r) for r in records)),
        support_stage_order=[
            'IDENTIFIED_REFERENCE_ADAPTER_REQUIRED',
            'REFERENCE_AVAILABLE_TRAINING_BINDING_REQUIRED',
            'REFERENCE_AND_TRAINING_BINDING_READY_NOT_VALIDLY_MEASURED',
            'VALID_MEASUREMENT_COMPLETED',
        ],
        support_counts_are_positions_not_distinct_operator_families=True,
        unclassified_positions=len(records)-sum(len(v) for v in groups.values()),
        unknown_labels=dict(unknown), ambiguous_positions=ambiguous,
        confirmed_problem_count=None, confirmed_root_cause_count=None,
        scope='REPORTING_GROUPS_ONLY; not new runtime verification or proof of bias',
        rule='Model, shape, stage and recurrence segments do not create additional operator families.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--roles', type=Path, required=True)
    p.add_argument('--historical-recovery', type=Path)
    p.add_argument('--additional-evidence', type=Path, action='append', default=[])
    p.add_argument('--measurement-audit', type=Path)
    p.add_argument('--measurement-merge-manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    inventory, roles = a.inventory.read_bytes(), a.roles.read_bytes()
    inventory_payload = json.loads(inventory)
    if bool(a.measurement_audit) != bool(a.measurement_merge_manifest):
        p.error('--measurement-audit and --measurement-merge-manifest are required together')
    if a.measurement_audit:
        try:
            from scripts.merge_family_execution_measurements import merge
        except ModuleNotFoundError:
            # Direct ``python scripts/...`` execution places scripts/, rather
            # than the repository root, on sys.path.
            from merge_family_execution_measurements import merge
        audit_bytes = a.measurement_audit.read_bytes()
        merge_manifest_bytes = a.measurement_merge_manifest.read_bytes()
        audit = json.loads(audit_bytes)
        merge_manifest = json.loads(merge_manifest_bytes)
        if merge_manifest.get('schema') != 'family-measurement-inventory-merge-v1':
            raise ValueError('Unexpected merge-manifest schema')
        inventory_payload = merge(
            inventory_payload, audit, merge_manifest['mappings'], verify_artifacts=True)
    historical=[]
    if a.historical_recovery:
        recovered=json.loads(a.historical_recovery.read_bytes())
        if recovered.get('schema')!='historical-operator-family-recovery-v1':
            raise ValueError('Unexpected historical evidence format')
        historical=recovered['records']
        root=Path(__file__).resolve().parents[1]
        for row in historical:
            if row.get('import_status')=='HISTORICAL_RECORD_READ_NOT_RERUN':
                if hashlib.sha256((root/row['path']).read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('Historical artifact changed')
    extra=[]
    for path in a.additional_evidence:
        payload=json.loads(path.read_bytes())
        if payload.get('schema')!='operator-family-additional-evidence-v1':
            raise ValueError('Unexpected additional evidence format')
        for row in payload['records']:
            artifact=Path(row['artifact_path'])
            if not artifact.is_absolute(): artifact=Path(__file__).resolve().parents[1]/artifact
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=row['artifact_sha256']:
                raise ValueError('Additional evidence artifact changed')
            extra.append(row)
    result = summarize(inventory_payload['records'], json.loads(roles)['cases'],historical,extra)
    result.update(schema='operator-family-report-v2', input_sha256={
        str(a.inventory): hashlib.sha256(inventory).hexdigest(),
        str(a.roles): hashlib.sha256(roles).hexdigest(),
        str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if a.historical_recovery:
        result['input_sha256'][str(a.historical_recovery)]=hashlib.sha256(a.historical_recovery.read_bytes()).hexdigest()
    for path in a.additional_evidence:
        result['input_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    if a.measurement_audit:
        result['input_sha256'][str(a.measurement_audit)]=hashlib.sha256(audit_bytes).hexdigest()
        result['input_sha256'][str(a.measurement_merge_manifest)]=hashlib.sha256(
            merge_manifest_bytes).hexdigest()
        result['selected_plan_measurement_merge'] = inventory_payload[
            'selected_plan_measurement_merge']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f: json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256', 'ambiguous_positions')}, ensure_ascii=False))


if __name__ == '__main__': main()
