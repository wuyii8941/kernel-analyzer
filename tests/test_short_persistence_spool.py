from __future__ import annotations

import torch

from scripts.run_short_persistence_from_spool import screen_spools


def _write_spool(path, *, field="effective_update"):
    rows = []
    # A persistent 8-step vector path should be a risk candidate.  The values
    # are deliberately tiny but use the same declared tensor ABI as a runner.
    for step in range(1, 9):
        rows.append({
            "phase": "evaluation",
            "step": step + 16,
            "state_id": f"s{step}",
            field: {
                "p": torch.tensor([1.0 + 0.01 * step, 0.5], dtype=torch.float32),
                "q": torch.tensor([0.25, 0.125], dtype=torch.float32),
            },
        })
    torch.save({
        "schema": "kernel-analyzer-seup-geometry-spool-v1",
        "case_id": "toy",
        "carrier_parameters": ["p", "q"],
        "rows": rows,
    }, path)


def test_geometry_spool_screen_keeps_only_projection(tmp_path):
    spool = tmp_path / "toy.pt"
    _write_spool(spool)
    result = screen_spools(
        [spool], field="effective_update", phase="evaluation", steps=8,
        projection_dim=16, projection_seed=7, null_draws=100,
    )
    assert result["status"] == "COMPLETE"
    assert result["input"]["raw_vectors_retained"] is False
    assert len(result["cases"]) == 1
    assert result["cases"][0]["coordinate_count"] == 4


def test_geometry_spool_rejects_missing_parameter(tmp_path):
    spool = tmp_path / "bad.pt"
    _write_spool(spool)
    payload = torch.load(spool, weights_only=False)
    payload["rows"][0]["effective_update"].pop("q")
    torch.save(payload, spool)
    try:
        screen_spools(
            [spool], field="effective_update", phase="evaluation", steps=8,
            projection_dim=16, projection_seed=7, null_draws=10,
        )
    except ValueError as exc:
        assert "missing declared parameter q" in str(exc)
    else:
        raise AssertionError("missing parameter must fail closed")
