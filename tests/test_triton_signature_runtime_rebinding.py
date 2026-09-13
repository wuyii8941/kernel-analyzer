from pathlib import Path

from scripts.rebind_triton_signature_runtimes import plan


def test_rebind_plan_deduplicates_release_and_preserves_runtime(tmp_path):
    release = tmp_path / "old"
    manifest = {"groups": [{
        "release": str(release),
        "runtime": {"architecture": "qwen", "model": "/data1/tzh/model",
                    "input_bank": "/data1/tzh/bank", "allow_graph_breaks": False},
    }, {
        "release": str(release),
        "runtime": {"architecture": "qwen", "model": "/data1/tzh/model",
                    "input_bank": "/data1/tzh/bank", "allow_graph_breaks": False},
    }]}
    rows = plan(manifest, release_root=tmp_path / "new")
    assert len(rows) == 1
    assert rows[0]["old_release"] == str(release.resolve())
    assert rows[0]["new_release"].endswith("old_signature_r2")
    assert rows[0]["architecture"] == "qwen"


def test_rebind_plan_uses_declared_existing_release(tmp_path):
    old, new = tmp_path / "old", tmp_path / "known-good"
    manifest = {"groups": [{
        "release": str(old),
        "runtime": {"architecture": "phi", "model": "/data1/tzh/model",
                    "input_bank": "/data1/tzh/bank", "allow_graph_breaks": True},
    }]}
    rows = plan(manifest, release_root=tmp_path / "new",
                existing_overrides={str(old.resolve()): str(new.resolve())})
    assert rows[0]["new_release"] == str(new.resolve())
    assert rows[0]["declared_existing_override"] is True
    assert rows[0]["allow_graph_breaks"] is True
