import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load():
    with gzip.open(
        ROOT / "results/coverage/endpoint_case_rescreen.json.gz", "rt"
    ) as handle:
        return json.load(handle)


def test_rescreen_reconciles_every_frozen_endpoint():
    result = load()
    counts = result["denominator"]["dispositions"]
    assert result["schema"] == "kernel-analyzer-endpoint-case-rescreen-v1"
    assert result["denominator"]["directional_endpoints"] == 1562
    assert result["denominator"]["complete_concrete_fb_proofs"] == 1562
    assert sum(counts.values()) == 1562
    assert len(result["rows"]) == 1562
    assert all("negative" not in key.lower() for key in counts)


def test_rescreen_finds_41_new_strict_unique_endpoint_cases():
    result = load()
    cases = result["strict_endpoint_cases"]
    summary = result["new_strict_endpoint_cases"]
    assert len(cases) == summary["count"] == 41
    assert summary["unique_candidate_ids"] == 41
    assert summary["unique_aot_endpoints_within_cell"] == 41
    assert summary["unique_generated_regions_within_cell"] == 41
    assert summary["by_model"] == {"deepseek8b": 32, "phi4": 1, "qwen": 8}
    assert summary["by_operation"] == {
        "add": 25, "bmm": 7, "mm": 3, "rsqrt": 5, "sum": 1,
    }
    for row in cases:
        assert row["disposition"] == "PASS_STRICT_ENDPOINT_FLASH_STYLE_CASE"
        assert row["complete_concrete_fb_proof"] is True
        assert all(row["binding_gates"].values())
        assert row["single_kernel_root_attribution"] is False


def test_recurrence_patterns_are_not_mislabeled_as_mechanisms():
    result = load()
    patterns = result["provisional_recurrence_patterns"]
    assert patterns["count"] == 9
    assert sum(row["endpoint_count"] for row in patterns["rows"]) == 41
    assert "not yet a deduplicated causal mechanism" in patterns["claim_boundary"]
    reconciliation = result["case_count_reconciliation"]
    assert reconciliation["previously_retained_strict_cases"] == 7
    assert reconciliation["new_same_dtype_endpoint_cases"] == 41
    assert reconciliation["combined_strict_cases_before_mechanism_deduplication"] == 48
