from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "r1_merge_paths.py"
    spec = importlib.util.spec_from_file_location("r1_merge_paths_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_metadata_reads_json(tmp_path):
    module = load_module()
    path = tmp_path / "meta.json"
    path.write_text('{"pid": 7}\n', encoding="utf-8")
    assert module.load_metadata(str(path)) == {"pid": 7}
