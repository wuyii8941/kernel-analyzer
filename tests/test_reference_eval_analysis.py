"""End-to-end test of the unified entry: capture packages + declaration -> report."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gmpy2")
pytest.importorskip("scipy")
pytest.importorskip("triton")

from kernel_analyzer.reference_eval.analysis import reference_stage, statistics_stage  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_unified_entry_runs_reference_and_statistics(tmp_path):
    from scripts import reference_eval_kernels as k

    n, block, chunks = 1 << 14, 1024, 6
    blocks = [1, 5, 9]
    window = np.concatenate([np.arange(b * block, (b + 1) * block) for b in blocks])
    root = tmp_path / "capture"
    root.mkdir()
    (root / "design.json").write_text(json.dumps({"rows": blocks}))
    g = torch.Generator(device="cuda").manual_seed(3)
    for unit in range(6):
        contributions = [torch.randn(n, device="cuda", generator=g) * 10.0 ** (c % 3) for c in range(chunks)]
        weight = torch.randn(n, device="cuda", generator=g)
        for variant, order in (("forward", range(chunks)), ("backward", reversed(range(chunks)))):
            acc = torch.zeros(n, device="cuda")
            recorder = TritonLaunchRecorder(window=lambda kernel, name, t: window)
            with recorder:
                for c in order:
                    k.accumulate[(n // block,)](acc, contributions[c], n, BLOCK=block)
                torch.cuda.synchronize()
            vdir = root / f"unit{unit:03d}" / variant
            for j, launch in enumerate(recorder.launches):
                save_launch(launch, vdir / f"launch{j:03d}")
            np.savez(vdir / "arrays.npz", weight=weight.cpu().double().numpy()[window])
    decl = {
        "location": "synthetic accumulation", "capture_root": str(root), "variants": ["forward", "backward"],
        "target_buffer": "ACC", "programs": {"rows_from": "design.json"}, "mode": "numerical_difference",
        "measurement": {"type": "adamw_write", "lr": 1e-3, "betas": [0.9, 0.95], "eps": 1e-8,
                        "parameter_values": "weight"},
        "comparisons": [{"candidate": "forward", "reference": "K_R"}, {"candidate": "backward", "reference": "K_R"},
                        {"candidate": "forward", "reference": "backward"}],
        "direction_rules": ["fixed_direction", "aligned_reference_update"],
        "population": {"description": "synthetic", "calibration": 2, "confirmation": 4}, "alpha": 0.05,
    }
    summaries = reference_stage(decl, tmp_path / "ref")
    assert len(summaries) == 6
    for s in summaries:
        for v in ("forward", "backward"):
            assert s[v]["classes"]["complete_composed"] == window.size
            assert not s[v]["target_rewritten_outside"]
    report = statistics_stage(decl, tmp_path / "ref", device="cuda")
    assert len(report["results"]) == 6
    assert report["reference"]["forward"]["classes_total"]["complete_composed"] == 6 * window.size
    assert all("final_verdict" in r for r in report["results"])
    assert report["ambiguous_rounding_elements"] == 0
    # The candidate-minus-reference residual is computed from the stored arrays, not re-derived.
    arrays = np.load(tmp_path / "ref" / "unit000.npz")
    assert np.array_equal(arrays["forward__index"], np.sort(arrays["forward__index"]))
    assert report["coordinates_used"] == report["coordinates_total"] == window.size
    # A coordinate whose reference is not established in one unit leaves the statistics everywhere.
    data = dict(np.load(tmp_path / "ref" / "unit003.npz"))
    data["backward__st"] = data["backward__st"].copy()
    data["backward__st"][7] = 5
    np.savez(tmp_path / "ref" / "unit003.npz", **data)
    gated = statistics_stage(decl, tmp_path / "ref", device="cuda")
    assert gated["coordinates_used"] == window.size - 1
    assert gated["excluded_reference_elements"]["not_established"] == 1
