from forkcert.historical_evaluation import (
    seal_pre_reveal_certificate,
    score_post_reveal,
    validate_pre_reveal_certificate,
)


def certificate():
    return {
        "case_identity": {"case_id": "opaque-case"},
        "stage_screen": {"failing_stages": ["aot_eager", "inductor"]},
        "region_reduction": {"input_region_count": 8, "candidate_regions": ["n3", "n4"]},
        "provenance": {"stage_tags": ["aot_eager"], "candidate_regions": [{"id": "n3"}],
                       "stopping_level": "aot"},
    }


def truth(sealed):
    return {
        "schema_version": "external-truth.v0.1",
        "case_id": "opaque-case",
        "certificate_sha256": sealed["pre_reveal"]["certificate_sha256"],
        "revealed_by_independent_evaluator": True,
        "patch_stage_tags": ["aot_eager"],
        "patch_candidate_ids": ["n3"],
        "correct_stopping_level": "aot",
    }


def test_seal_and_post_reveal_score_are_bound_to_the_same_certificate():
    sealed = seal_pre_reveal_certificate(certificate())
    assert validate_pre_reveal_certificate(sealed) == []
    score = score_post_reveal(sealed, truth(sealed))
    assert score["valid"]
    assert score["stage_coverage"]
    assert score["candidate_mechanism_coverage"]
    assert score["stopping_decision_correct"]
    assert not score["erroneous_kernel_descent"]


def test_post_reveal_scoring_fails_closed_after_certificate_mutation():
    sealed = seal_pre_reveal_certificate(certificate())
    sealed["region_reduction"]["candidate_regions"] = ["n3"]
    score = score_post_reveal(sealed, truth(seal_pre_reveal_certificate(certificate())))
    assert not score["valid"]
    assert "hash mismatch" in " ".join(score["errors"])


def test_wrong_kernel_descent_is_scored_even_if_stage_matches():
    raw = certificate()
    raw["provenance"]["stopping_level"] = "kernel"
    sealed = seal_pre_reveal_certificate(raw)
    score = score_post_reveal(sealed, truth(sealed))
    assert score["stage_coverage"]
    assert score["erroneous_kernel_descent"]
    assert not score["stopping_decision_correct"]


def test_seal_refuses_certificate_that_already_contains_reveal_material():
    raw = certificate()
    raw["post_reveal"] = {"patch": "forbidden"}
    try:
        seal_pre_reveal_certificate(raw)
    except ValueError as error:
        assert "reveal" in str(error)
    else:
        raise AssertionError("seal must fail closed")
