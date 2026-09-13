import gzip
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_refresh_can_target_one_exact_cut_when_stored_graph_index_is_stale(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    old_hash = "a" * 64
    new_hash = "b" * 64
    plan = {
        "reference_cut_tasks": [
            {
                "task_id": "same-dtype:backward:graph0:x",
                "phase": "BACKWARD",
                "graph_index": 1,
                "expected_graph_code_sha256": old_hash,
            },
            {
                "task_id": "same-dtype:backward:graph1:y",
                "phase": "BACKWARD",
                "graph_index": 1,
                "expected_graph_code_sha256": old_hash,
            },
        ]
    }
    with gzip.open(release / "same_dtype_tasks.json.gz", "wt") as stream:
        json.dump(plan, stream)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/refresh_reference_graph_hashes.py"),
            "--release",
            str(release),
            "--phase",
            "BACKWARD",
            "--cut-id",
            "same-dtype:backward:graph0:x",
            "--old-hash",
            old_hash,
            "--new-hash",
            new_hash,
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    with gzip.open(release / "same_dtype_tasks.json.gz", "rt") as stream:
        refreshed = json.load(stream)
    assert refreshed["reference_cut_tasks"][0]["expected_graph_code_sha256"] == new_hash
    assert refreshed["reference_cut_tasks"][1]["expected_graph_code_sha256"] == old_hash
    provenance = json.loads((release / "reference_graph_hash_rebind.json").read_text())
    assert provenance["changes"][0]["cut_id"] == "same-dtype:backward:graph0:x"
    assert provenance["changes"][0]["changed_reference_cut_count"] == 1
