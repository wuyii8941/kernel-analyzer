from forkcert.localization import StageObservation, ddmin_regions, localization_certificate, screen_stages
from forkcert.localization_runtime import run_localization


def test_stage_screen_does_not_infer_a_unique_first_bad_stage():
    screen = screen_stages([
        StageObservation("eager", True),
        StageObservation("dynamo_eager", True),
        StageObservation("aot_eager", False),
        StageObservation("inductor", False),
    ])
    assert screen["allowed_stage_claim"] == "SUPPORTED_STAGE_CANDIDATE"
    assert screen["failing_stages"] == ["aot_eager", "inductor"]
    assert "unique first-bad" in screen["not_claimed"]


def test_stage_screen_records_unknown_gap_fail_closed():
    screen = screen_stages([StageObservation("eager", True), StageObservation("aot", None), StageObservation("inductor", False)])
    assert screen["allowed_stage_claim"] == "SUPPORTED_STAGE_CANDIDATE_WITH_UNKNOWN_GAPS"


def test_ddmin_is_name_agnostic_and_returns_one_minimal_set():
    calls = []
    def predicate(regions):
        calls.append(regions)
        return "r3" in regions and "r4" in regions
    result = ddmin_regions(["r1", "r2", "r3", "r4"], predicate)
    assert result["status"] == "ONE_MINIMAL_CANDIDATE_SET"
    assert result["candidate_regions"] == ["r3", "r4"]
    assert result["reduction_ratio"] == 0.5
    assert calls


def test_singleton_inventory_is_not_mislabeled_as_a_reduction():
    result = ddmin_regions(["only_region"], lambda regions: regions == ("only_region",))
    assert result["status"] == "UNREDUCIBLE_SINGLETON_INVENTORY"
    assert result["reduction_ratio"] == 0.0
    assert "symptom-preserving reduction" in result["not_claimed"]


def test_certificate_requires_explicit_evidence_inputs():
    certificate = localization_certificate(
        case_identity={"case_id": "x"}, semantic_contract={"endpoint": "e"},
        stage_screen={}, reduction={}, provenance={}, evidence={}, manual_decisions=[],
    )
    assert certificate["case_identity"]["case_id"] == "x"
    assert "stops" in certificate["stopping_reason"]


def test_runtime_core_does_not_need_case_or_operator_names():
    class Adapter:
        def case_identity(self): return {"case_id": "opaque"}
        def semantic_contract(self): return {"endpoint": "opaque predicate"}
        def stage_ids(self): return ("s0", "s1")
        def run_stage(self, stage): return {"contract_holds": stage == "s0", "artifact_ids": [stage]}
        def region_ids(self): return ("r0", "r1", "r2")
        def preserves_symptom(self, enabled): return "r1" in enabled
        def provenance(self, regions): return {"regions": list(regions)}
        def evidence(self, regions): return {"regions": list(regions)}
    result = run_localization(Adapter())
    assert result.stage_screen["failing_stages"] == ["s1"]
    assert result.reduction["candidate_regions"] == ["r1"]
    assert result.certificate["manual_decisions"] == []
