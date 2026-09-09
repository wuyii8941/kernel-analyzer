from pathlib import Path

from scripts.build_attention_family_evidence import build


def test_checked_attention_family_evidence():
    root=Path('results/property/numerical_coverage_v1/qwen_flash_sdpa_attention_v1')
    evidence=build(root)['records'][0]
    assert evidence['family']=='FUSED_ATTENTION'
    assert evidence['equivalence_decision']=='NON_EQUIVALENT'
    assert evidence['stage_effects']['LOCAL']['confirmation_total_rms']<.002
    assert evidence['stage_effects']['PARAMETER_GRADIENT']['confirmation_total_rms']>.1
    assert evidence['stage_effects']['PARAMETER_WRITE']['confirmation_total_rms']>.3
    assert evidence['training_quality_claim'] is False
