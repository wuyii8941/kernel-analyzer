"""Large tensor records are NumPy arrays (path A2 deviation D-A2-6: nested lists of 10^8 elements exhausted memory).
With the threshold lowered, the array record validates, compares and evaluates exactly like the list record."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "5_specification"), str(ROOT / "5_specification" / "corpus")]
import call_binding as B  # noqa: E402
from kernel_analyzer import contract_binding  # noqa: E402
from kernel_spec import bulk  # noqa: E402


def test_array_records_match_list_records(monkeypatch):
    x = torch.tensor([[1.5, -2.0, 3.25], [0.0, 7.0, -0.5]], dtype=torch.bfloat16)
    y = torch.tensor([[2.0, 1.0, -1.0], [4.0, 0.5, 0.25]], dtype=torch.bfloat16)
    as_list = contract_binding._tensor_record(x), contract_binding._tensor_record(y)
    monkeypatch.setattr(contract_binding, "LARGE_RECORD", 4)
    as_array = contract_binding._tensor_record(x), contract_binding._tensor_record(y)
    assert isinstance(as_array[0]["v"], np.ndarray) and isinstance(as_list[0]["v"], list)
    assert B._tensor_record(as_array[0]) is as_array[0]
    assert B._same_record_values(as_array[0], as_list[0])
    ref_list = bulk.evaluate("ARITH", "real", {"op": "aten.add", "args": list(as_list)}, force=True)
    ref_array = bulk.evaluate("ARITH", "real", {"op": "aten.add", "args": list(as_array)}, force=True)
    assert np.array_equal(ref_list.fields["out"].lo, ref_array.fields["out"].lo)
    assert np.array_equal(ref_list.fields["out"].hi, ref_array.fields["out"].hi)
