import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/case_causal_audit_v1/natural_problem_group_discovery_current.json"


def test_natural_problem_group_discovery_is_conservative_and_ranked():
    subprocess.run(
        [sys.executable, "scripts/build_natural_problem_group_discovery.py"],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(OUT.read_text())
    assert report["status"] == "TRIAGE_ONLY_NO_PROMOTION"
    assert report["candidate_frontier"]["candidate_family_count"] == 8
    assert report["semantic_gap_audit"]["semantic_gap_count"] == 0
    assert report["ledger_snapshot"]["root_cause_closed_final_count"] == 52
    assert report["ledger_snapshot"]["root_cause_unresolved_count"] == 0
    inventory = report["saved_probe_inventory"]
    assert inventory["probe_count_with_write_effect"] == 128
    assert inventory["high_effect_probe_count"] == 94
    # Every saved high-effect probe which is not directly cited by an active
    # or open ledger row is an alternate run of an already audited group.  A
    # genuinely unowned high-effect probe must fail this test and be reviewed
    # before the search is declared exhausted.
    assert {
        row["file"] for row in inventory["unreferenced_high_effect_probes"]
    } == {
        "results/property/new_problem_group_search_v1/bert_linear_fused_addmm_cuda32_20260919.json",
        "results/property/new_problem_group_search_v1/bert_softmax_materialization_cuda_32_20260919.json",
        "results/property/new_problem_group_search_v1/bert_tiny_layernorm_natural_20260918.json",
        "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_16_20260919.json",
        "results/property/new_problem_group_search_v1/deberta_disentangled_attention_materialization_natural_32_20260919.json",
        "results/property/new_problem_group_search_v1/gemma4_audio_language_output_projection_natural_8_20260919.json",
        "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer0_natural_8_20260919.json",
        "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_16_20260920.json",
        "results/property/new_problem_group_search_v1/gemma4_audio_subsample_conv_layer1_natural_8_20260919.json",
        "results/property/new_problem_group_search_v1/mamba_official_local_scan_layer3_8_20260920.json",
        "results/property/new_problem_group_search_v1/gemma4_audio_attention_softmax_layer0_16_repeat_20260920.json",
    }
    assert inventory["unreferenced_high_effects_all_classified"] is True
    active_or_open = set(report["ledger_snapshot"]["active_problem_groups"])
    active_or_open.update(report["ledger_snapshot"]["open_problem_groups"])
    assert all(
        row["known_duplicate_group"] in active_or_open
        for row in inventory["unreferenced_high_effect_probes"]
    )
    assert report["next_source_isolation_order"]
    remaining = {row["family_id"]: row for row in report["remaining_operator_family_frontier"]}
    assert {"RECURRENCE", "DATA_MOVEMENT_LAYOUT", "ELEMENTWISE", "FUSED_MIXED"}.issubset(remaining)
    assert remaining["RECURRENCE"]["priority_rank"] == 3
    assert all(not row["eligible_for_final_group_now"] for row in remaining.values())
    assert all(not row["natural_problem_group_eligible_now"] for row in report["candidate_frontier"]["families"])
    assert report["non_promotions"][-1]["items"] == ["indexed_accumulation_reduction_order"]
    assert {row["operator"] for row in report["recent_natural_screens"]} == {
        "BERT MLP GELU evaluation/materialization",
        "Llama SwiGLU gate-times-up product materialization",
        "OLMoE router score materialization",
            "Qwen3 attention output layout materialization (k_proj)",
            "Mamba fused recurrence x_proj projection",
            "Gemma4 audio attention softmax materialization",
        "DeepSeek generated SiLU backward source intervention",
        "RWKV channel-mix square/ReLU materialization",
        }
    statuses = {row["operator"]: row["status"] for row in report["recent_natural_screens"]}
    assert {
        row["file"]
        for row in report["recent_natural_screens"]
        if row["operator"] == "OLMoE router score materialization"
    } == {
        "results/property/new_problem_group_search_v1/olmoe_router_score_natural_16_20260920.json",
        "results/property/new_problem_group_search_v1/olmoe_router_score_natural_32_20260920.json",
        "results/property/new_problem_group_search_v1/olmoe_router_score_natural_64_20260920.json",
        "results/property/new_problem_group_search_v1/olmoe_router_score_natural_128_20260920.json",
    }
    assert statuses["BERT MLP GELU evaluation/materialization"] == "SCREENED_EXACT_IDENTITY"
    assert statuses["Llama SwiGLU gate-times-up product materialization"] == "SCREENED_EXACT_IDENTITY"
    assert statuses["OLMoE router score materialization"] == "MERGED_WITH_EXISTING_ROUTER_SCORE_GROUP"
    assert statuses["Qwen3 attention output layout materialization (k_proj)"] == "SCREENED_EXACT_IDENTITY"
    assert statuses["Mamba fused recurrence x_proj projection"] == "SCREENED_FUSED_RECURRENCE_NO_CONFIRMED_ALIGNED_BIAS"
    assert statuses["Gemma4 audio attention softmax materialization"] == "SCREENED_HIGH_VARIANCE_NO_DIRECTIONAL_BIAS"
    assert statuses["RWKV channel-mix square/ReLU materialization"] == "SCREENED_SPARSE_EFFECT_NO_DIRECTIONAL_BIAS"
    assert {
        row["file"]
        for row in report["recent_natural_screens"]
        if row["operator"] == "RWKV channel-mix square/ReLU materialization"
    } == {
        "results/property/new_problem_group_search_v1/rwkv_channel_mix_readme_128_20260920.json",
        "results/property/new_problem_group_search_v1/rwkv_channel_mix_block1_readme_64_20260920.json",
    }
    assert {
        row["file"]
        for row in report["recent_natural_screens"]
        if row["operator"] == "Mamba fused recurrence x_proj projection"
    } == {
        "results/property/new_problem_group_search_v1/mamba_fused_x_proj_layer0_natural_24_20260920.json",
    }
    olmoe128 = json.loads(
        (
            ROOT
            / "results/property/new_problem_group_search_v1/olmoe_router_score_natural_128_20260920.json"
        ).read_text()
    )
    assert olmoe128["summary"]["confirmation_aligned_write_interval_normal_95"][1] < 0.0
