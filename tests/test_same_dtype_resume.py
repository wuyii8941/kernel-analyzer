import gzip
import json
from pathlib import Path

from scripts.run_same_dtype_semantic_oracle import load


def test_load_accepts_gzip_checkpoint_with_partial_suffix(tmp_path: Path) -> None:
    checkpoint = tmp_path / ".same_dtype_oracle.json.gz.partial"
    expected = {"status": "PARTIAL_FAIL_CLOSED", "states": {"0": {}}}
    with gzip.open(checkpoint, "wt", encoding="utf-8") as handle:
        json.dump(expected, handle)
    assert load(checkpoint) == expected


def test_load_accepts_plain_json_input(tmp_path: Path) -> None:
    path = tmp_path / "bank.json"
    expected = {"states": []}
    path.write_text(json.dumps(expected), encoding="utf-8")
    assert load(path) == expected
