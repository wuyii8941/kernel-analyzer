from kernel_analyzer import forward_capture_audit as audit


def test_incomplete_record_never_analyzed(monkeypatch):
    monkeypatch.setattr(audit, 'analyze_artifact', lambda *a: (_ for _ in ()).throw(AssertionError()))
    result = audit.audit_record({}, dict(case_id='x', carrier='w', task_id='t'), {}, ['0'])
    assert result['status'] == 'INVALID_OR_INCOMPLETE'
    assert result['analysis'] is None


def test_missing_margin_not_backfilled(monkeypatch):
    protocol = dict(schema='residual-rms-forward-capture-v1', contrast_id='SINGLE_FORWARD_OUTPUT_REPLACEMENT')
    raw = dict(case_id='x', carrier='w', runtime_boundary=dict(task_id='t'), status='COMPLETE',
        state_ids=['0'], determinism=dict(all_exact=True), original_coordinate_statistics={
        s:[dict(effect_energy=0., repair_energy=1., effect_repair_inner_product=0.)]
        for s in ('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')},
        parameter_write_protocol=dict(version='adamw-readback-v2'),
        reference_comparison_scope=dict(comparison='SINGLE_FORWARD_OUTPUT_REPLACEMENT',
            same_local_operands=True, includes_possible_upstream_differences=False,
            complete_multi_output_implementation_replacement=False))
    def analyze(payload, actual_protocol):
        raise AssertionError('Do not invoke margin-required analyzer with an undeclared policy')
    monkeypatch.setattr(audit, 'analyze_artifact', analyze)
    result = audit.audit_record(raw, dict(case_id='x', carrier='w', task_id='t'), protocol, ['0'])
    assert not result['equivalence_policy_declared']
    assert result['analysis'] is None
    assert result['equivalence_decision'] == 'NOT_ASSESSED'
    assert result['not_assessed_reason'] == 'EQUIVALENCE_MARGIN_NOT_DECLARED'


def test_corrupt_statistics_rejected_without_equivalence_policy():
    for row in ({}, dict(effect_energy=-1, repair_energy=1, effect_repair_inner_product=0),
                dict(effect_energy=1, repair_energy=1, effect_repair_inner_product=2),
                dict(effect_energy=float('nan'), repair_energy=1, effect_repair_inner_product=0)):
        assert not audit.valid_statistics([row], 1)
    assert not audit.valid_statistics([], 1)
