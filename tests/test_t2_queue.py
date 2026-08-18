from scripts.build_t2_queue import build_queue


def test_t2_queue_deduplicates_regions_and_retains_many_to_one_units():
    t1 = {"rows": {
        "u1": {"status": "PASS", "evidence": {"candidate_region_ids": ["r1"], "max_abs": 2.0}},
        "u2": {"status": "PASS", "evidence": {"candidate_region_ids": ["r1", "r2"], "max_abs": 1.0}},
        "u3": {"status": "FAIL", "evidence": {"candidate_region_ids": ["r3"], "max_abs": 9.0}},
    }}
    campaign = {"rows": [
        {"region_id": "r1", "reference_entrypoint": "pkg:fn", "symbol": "s1", "phase": "FORWARD", "output_names": ["out"]},
        {"region_id": "r2", "reference_entrypoint": None, "symbol": "s2", "phase": "BACKWARD", "output_names": ["out"]},
    ]}
    observations = {"rows": [{"region_id": "r1", "max_abs_max": 0.25}]}
    result = build_queue(t1, campaign, None, observations)
    assert result["t1_passing_proof_units"] == 2
    assert result["unique_t1_regions"] == 2
    assert result["executable_regions"] == 1
    assert result["selected_arms"] == [{"region_id": "r1", "endpoints": ["out"]}]
    r1 = next(row for row in result["rows"] if row["region_id"] == "r1")
    assert r1["proof_unit_ids"] == ["u1", "u2"]
    assert r1["t1_max_abs"] == 0.25


def test_t2_queue_selects_all_regions_for_a_closed_proof_unit():
    t1 = {"rows": {
        "u1": {"status": "PASS", "evidence": {"candidate_region_ids": ["r1", "r2"], "max_abs": 2.0}},
        "u2": {"status": "PASS", "evidence": {"candidate_region_ids": ["r3"], "max_abs": 1.0}},
    }}
    campaign = {"rows": [
        {"region_id": r, "reference_entrypoint": "pkg:fn", "symbol": r, "phase": "FORWARD", "output_names": ["out"]}
        for r in ("r1", "r2", "r3")
    ]}
    result = build_queue(t1, campaign, None, unit_limit=1)
    assert result["selected_proof_unit_ids"] == ["u2"]
    assert result["selected_arms"] == [{"region_id": "r3", "endpoints": ["out"]}]


def test_t2_queue_excludes_empty_or_elided_units_from_fb_selection():
    t1 = {"rows": {
        "empty": {"status": "PASS", "evidence": {"candidate_region_ids": ["r1"], "max_abs": 4.0}},
        "actual": {"status": "PASS", "evidence": {"candidate_region_ids": ["r2"], "max_abs": 1.0}},
    }}
    campaign = {"rows": [
        {"region_id": r, "reference_entrypoint": "pkg:fn", "symbol": r, "phase": "FORWARD", "output_names": ["out"]}
        for r in ("r1", "r2")
    ]}
    proof = {"proof_units": [
        {"unit_id": "empty", "unit_kind": "EMPTY_OR_ELIDED_FB_UNIT"},
        {"unit_id": "actual", "unit_kind": "FORWARD_ACTUAL_BACKWARD_UNIT"},
    ]}
    result = build_queue(t1, campaign, None, unit_limit=1, proof_units=proof)
    assert result["selected_proof_unit_ids"] == ["actual"]
    assert result["selected_arms"][0]["region_id"] == "r2"
