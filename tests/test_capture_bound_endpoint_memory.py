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

