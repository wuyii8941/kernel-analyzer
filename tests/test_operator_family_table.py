from scripts.render_operator_family_table import rows_from_report


def test_table_keeps_position_and_extra_evidence_separate():
    family=dict(family_id='SELECTION',label='Top-k',classified_positions=0,
        support_stage_counts={},additional_measurement_evidence=[{'x':1}],
        valid_measurement_implementation_kind_counts={},
        historical_role_records=0,additional_historical_artifacts=[])
    report=dict(schema='operator-family-report-v2',families=[family])
    row=rows_from_report(report)[0]
    assert row['classified_positions']==0
    assert row['valid_measurement_positions']==0
    assert row['additional_fixed_suite_evidence']==1
