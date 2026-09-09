from scripts.build_optimizer_family_evidence import build


def test_checked_optimizer_family_evidence():
    root='results/property/numerical_coverage_v1/torchao_adamw8bit_mamba_v2'
    evidence=build(__import__('pathlib').Path(root))['records'][0]
    assert evidence['family']=='OPTIMIZER_UPDATE'
    assert evidence['equivalence_decision']=='NON_EQUIVALENT'
    assert evidence['stage_effects']['GRADIENT_INPUT']['confirmation_total_rms']==0.0
    assert evidence['stage_effects']['PARAMETER_WRITE']['confirmation_total_rms']>0.05
    assert evidence['generated_files_containing_triton_jit']>=1
