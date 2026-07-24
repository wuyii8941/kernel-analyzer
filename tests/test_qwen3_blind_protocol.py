from pathlib import Path

from theory_oracle.verify_qwen3_blind_protocol_v0_1 import audit


def test_case001_is_patch_free_opaque() -> None:
    root = Path(__file__).resolve().parents[1]
    report = audit(root / "data/qwen_bug_sources/qwen3_attention_varlen_layout_case_001")
    assert report["status"] == "VALID_PATCH_FREE_OPAQUE_CASE"
    assert report["manifest_key_leaks"] == []
    assert report["path_leaks"] == []


def test_case004_is_patch_free_opaque() -> None:
    root = Path(__file__).resolve().parents[1]
    report = audit(root / "data/qwen_bug_sources/qwen3_1p7b_attention_varlen_layout_case_004")
    assert report["status"] == "VALID_PATCH_FREE_OPAQUE_CASE"
