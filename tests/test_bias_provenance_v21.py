import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v21_roster_has_static_provenance_and_no_runner_commit_claim():
    report = json.loads((ROOT / "results/property/bias_formation_v2_1/feasibility_report.json").read_text())
    provenance = report["provenance"]
    assert provenance["binding_generator_commit"] == "2642ca2bdb007cbf47edee1a6e9b53021906549a"
    assert provenance["protocol_freeze_commit"] == "02d9743a1c91b260e15fb0133576ab28f55eba21"
    assert provenance["runtime_commit"] is None
    assert provenance["binding_script_sha256"]


def test_not_available_is_never_a_bound_repair_or_sham():
    roster = json.loads((ROOT / "results/property/bias_formation_v2_1/roster_bound.json").read_text())
    for case in roster["cases"]:
        if case["sham_id"] == "NOT_AVAILABLE":
            assert case["sham_binding"]["bound"] is False
            assert case["repair_and_sham_bound"] is False
