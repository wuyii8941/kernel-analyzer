import gzip
import json
from pathlib import Path
import sys

import pytest

from scripts.build_same_dtype_case_ledger import main


def write(path: Path, value: dict) -> None:
    with gzip.open(path, "wt") as handle:
        json.dump(value, handle)


def test_ledger_fails_closed_before_all_twelve_cells(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "ledger", "--runtime-root", str(runtime),
        "--output", str(tmp_path / "ledger.json.gz"),
    ])
    with pytest.raises(RuntimeError, match="12-cell ledger is incomplete"):
        main()
