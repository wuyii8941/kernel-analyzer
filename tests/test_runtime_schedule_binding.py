import json
from types import SimpleNamespace

import torch

from scripts.runtime_schedule_binding import bind_runtime_schedule


def test_runtime_binding_copies_the_exact_warmed_fb_wrappers(tmp_path, monkeypatch) -> None:
    modules = []
    for kind in ("forward", "backward"):
        source = tmp_path / f"{kind}.py"
        source.write_text(f"# AOT ID: ['0_{kind}']\nclass Runner: pass\n")
        modules.append(SimpleNamespace(__file__=str(source)))
    calls = []
    monkeypatch.setattr(
        "scripts.runtime_schedule_binding.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    trace = tmp_path / "trace"
    manifest = tmp_path / "manifest.json"
    bind_runtime_schedule(
        modules=modules, work_dir=trace, manifest=manifest,
        inventory=tmp_path / "inventory.json.gz",
        campaign=tmp_path / "campaign.json.gz", architecture="gemma3",
        state={"state_id": "warm-2"},
        input_digests={"token_ids_sha256": "a" * 64, "image_sha256": "b" * 64},
        values=(torch.zeros((1, 3), dtype=torch.long),), modality="IMAGE_TEXT",
        gradient_checkpointing=True, allow_graph_breaks=True,
    )
    payload = json.loads(manifest.read_text())
    assert payload["phase_module_counts"] == {"BACKWARD": 1, "FORWARD": 1}
    assert payload["input"]["state_id"] == "warm-2"
    assert len(list(trace.rglob("output_code.py"))) == 2
    assert len(calls) == 2
