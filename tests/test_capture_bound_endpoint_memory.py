import torch

from scripts.capture_bound_endpoint_bias_formation_v21 import (
    copy_reference_and_count_changes,
)


def test_reference_copy_change_audit_is_chunked_and_exact():
    candidate = torch.zeros(33)
    reference = torch.arange(33, dtype=torch.float32)
    changed = copy_reference_and_count_changes(
        candidate, reference, chunk_elements=5
    )
    assert changed == 32
    assert torch.equal(candidate, reference)


def test_large_vector_profile_records_exact_stats_without_sketch(monkeypatch):
    import scripts.run_training_bias_profile_v2_empirical as empirical

    monkeypatch.setattr(empirical, "LARGE_VECTOR_SKETCH_LIMIT", 4)
    store = empirical._new_stage_store(include_parameter_write=True)
    original = empirical._new_original_statistics()
    effect = torch.tensor([1., 0., 0., 0., 0.])
    repair = torch.ones_like(effect)
    empirical._append_contrast(
        store, "ADAMW_UPDATE", effect, repair, original_statistics=original
    )
    assert len(original["ADAMW_UPDATE"]) == 1
    assert store["ADAMW_UPDATE"] == {}
    assert store["__metadata__"]["ADAMW_UPDATE"]["direction_views"] == "NOT_RETAINED"
