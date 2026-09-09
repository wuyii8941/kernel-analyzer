from kernel_analyzer.explicit_output_capture_audit import audit_record,CONTRACTS


def test_grouped_causal_softmax_uses_shared_saved_record_audit():
    assert CONTRACTS['grouped-causal-softmax-forward-capture-v1'] == (
        'SINGLE_GROUPED_CAUSAL_SOFTMAX_OUTPUT_REPLACEMENT',
        'DECLARED_FP32_EXPRESSION_WITH_ORIGINAL_WRITES',
    )


def test_missing_margin_never_silently_becomes_equivalence():
    for schema,(comparison,variant) in CONTRACTS.items():
        case=dict(case_id='case',carrier='weight',task_id='task')
        protocol=dict(schema=schema,claim_scope='FIXED_SUITE_UPDATE')
        raw=dict(case_id='case',carrier='weight',runtime_boundary={'task_id':'task'},status='COMPLETE',
            state_ids=['0','1'],parameter_write_protocol={'version':'adamw-readback-v2'},
            determinism={'all_exact':True},reference_comparison_scope=dict(comparison=comparison,
                reference_variant=variant,same_local_operands=True,includes_possible_upstream_differences=False,
                reference_is_absolute_truth=False),original_coordinate_statistics={s:[dict(effect_energy=0.,
                repair_energy=1.,effect_repair_inner_product=0.) for _ in range(2)]
                for s in ('LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE')})
        result=audit_record(raw,case,protocol,['0','1'])
        assert result['status']=='RECORDED_MEASUREMENT_CHECKED'
        assert result['equivalence_decision']=='NOT_ASSESSED'
        raw['carrier']='other'
        assert audit_record(raw,case,protocol,['0','1'])['status']=='INVALID_OR_INCOMPLETE'
