#!/usr/bin/env python3
"""Item B (docs/protocol_essential_bugs_phase2_20261007.md): W5 test-criteria review for scatter_reduce, index_reduce,
max_pool{1,2,3}d and cross_entropy at the locked v2.10.0 sources (.cache/essential/w5/v2.10.0/, fetched from GitHub),
and gradcheck on the 50 B020 conditions of phase 1.

    python scripts/essential/w5_review.py
"""
import gzip
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import run_phase1 as R  # noqa: E402

ROOT = common.ROOT
OUT = ROOT / "results/essential/phase2a"
SRC = "v2.10.0: torch/testing/_internal/common_methods_invocations.py; test/inductor/test_torchinductor_opinfo.py; " \
      "test/test_scatter_gather_ops.py; test/nn/test_pooling.py; test/test_nn.py"

REVIEW = {
    "scatter_reduce": {
        "opinfo": "variants sum / prod / mean / amin / amax; amin, amax: supports_forward_ad=True, supports_fwgrad_bwgrad=True "
                  "(gradcheck and forward-mode AD against finite differences run on the samples)",
        "samples": "sample_inputs_scatter_reduce: make_tensor (continuous random values), include_self in (False, True, False); "
                   "hand-written zero cases for prod only",
        "samples_reach": {"include_self=False with the excluded value equal to the result": False,
                          "ties among participants": False, "padding-only window": "not applicable"},
        "eager_as_reference": "Inductor OpInfo test compares compiled against eager (test_comprehensive)",
        "skipped_or_disabled": [{"where": "inductor_one_sample['cuda']", "what": "scatter_reduce.amax / .amin / .mean run only "
                                 "the first sample (f16, f32, f64)", "reason": "not stated (speed list)"}],
        "dedicated_tests": "test_scatter_reduce_amax / _amin (test_scatter_gather_ops.py 312-345): forward values and NaN / inf "
                           "propagation for include_self True and False; no gradient test",
        "tolerances": "OpInfo defaults",
    },
    "index_reduce": {
        "opinfo": "variants mean / prod / amin / amax; gradcheck on the samples (no forward-mode AD)",
        "samples": "sample_inputs_index_reduce: make_tensor (continuous random values), include_self in (True, False); "
                   "hand-written zero cases for prod only",
        "samples_reach": {"include_self=False with the excluded value equal to the result": False,
                          "ties among participants": False, "padding-only window": "not applicable"},
        "eager_as_reference": "Inductor OpInfo test compares compiled against eager",
        "skipped_or_disabled": [
            {"where": "inductor_override_kwargs['cuda']", "what": "index_reduce.mean check_gradient False (f16, f32, f64)",
             "reason": "'Unreasonably high atol requirement'"},
            {"where": "inductor_override_kwargs['cuda']", "what": "index_reduce.amin / .amax check_gradient False (f16, f32, f64)",
             "reason": "'Gradient contains non-finite entries' (the compiled backward on single-element samples: B016)"},
            {"where": "inductor_skips['cpu']", "what": "index_reduce prod / mean f16", "reason": "'flaky'"}],
        "dedicated_tests": "none for amax / amin gradients",
        "tolerances": "OpInfo defaults; f16 atol 2e-3 rtol 3e-3 in TestInductorOpInfo",
    },
    "max_pool": {
        "opinfo": "nn.functional.max_pool1d / 2d / 3d and max_pool2d_with_indices_backward; supports_forward_ad=True",
        "samples": "kernel 3 / (3,2) / (3,2,3), stride 2 / None / (2,1) / (2,1,2), padding 0 / 1 / (1,1), dilation 1 / (1,2) / "
                   "(1,2,1), signal sizes 3 and 6 (2d adds 6, 3d adds 6 and 5); make_tensor values",
        "samples_reach": {"padding-only window": False, "ties": "not deliberately",
                          "why": "with kernel >= 2, padding <= 1 and dilation <= 2 on axes of size >= 3, every window samples at "
                                 "least one input position; a padding-only window needs a very short axis"},
        "eager_as_reference": "Inductor OpInfo test compares compiled against eager",
        "skipped_or_disabled": [
            {"where": "inductor_one_sample['cuda']", "what": "max_pool1d (f16, f32, f64), max_pool3d (f16), "
             "max_pool2d_with_indices_backward (f16, f32, f64) run only the first sample", "reason": "not stated"},
            {"where": "inductor_one_sample['cpu']", "what": "max_pool2d_with_indices_backward (f16, f32, f64)", "reason": "not stated"}],
        "dedicated_tests": "test_max_pool1d_corner_cases (test/nn/test_pooling.py ~1110, onlyCPU): checks exactly the padding-only "
                           "geometry (input [[1]], kernel 2, padding 1, dilation 2) but only the forward output (-inf) with "
                           "return_indices=False - neither the returned index nor the backward",
        "tolerances": "OpInfo defaults; max_pool2d f16 atol 1e-4 rtol 2e-3 (xpu)",
    },
    "cross_entropy": {
        "opinfo": "nn.functional.cross_entropy; supports_forward_ad=True",
        "samples": "shape (2, 3) and spatial variants; reductions; class weight = make_tensor((3,)) (random, can be negative); "
                   "ignore_index=1; probability targets; NO label_smoothing sample",
        "samples_reach": {"label_smoothing": False, "all targets ignored ('mean' denominator zero)": False,
                          "zero class weight on a present target": False},
        "eager_as_reference": "Inductor OpInfo test compares compiled against eager",
        "skipped_or_disabled": [
            {"where": "inductor_one_sample['cuda'] and ['cpu']", "what": "cross_entropy runs only the first sample (f16, f32, f64)",
             "reason": "not stated"},
            {"where": "OpInfo skips", "what": "TestJit test_variant_consistency_jit expectedFailure on CUDA", "reason": "CUDA memory leak"}],
        "dedicated_tests": "test_nn: test_cross_entropy_label_smoothing_* (consistency between index and probability targets, "
                           "errors); these compare two PyTorch code paths with each other",
        "tolerances": "OpInfo defaults",
    },
}


def gradcheck_b020():
    """torch.autograd.gradcheck (float64, CPU, default tolerances) on every seed of the 50 B020 conditions."""
    d = json.load(gzip.open(ROOT / "results/essential/phase1/classification_index.json.gz", "rt"))
    conds = [r["condition"] for r in d["records"]
             if r["candidates"]["eager_cpu64"]["class"]["bwd"]["class"].startswith("deviates")]
    rows = []
    for cond in conds:
        for seed in R.C.SEEDS:
            inp = R.FAMILIES["index"][1](cond, seed)
            s = torch.tensor(inp["self"], dtype=torch.float64, requires_grad=True)
            src = torch.tensor(inp["source"], dtype=torch.float64, requires_grad=True)
            idx = torch.tensor(inp["index"], dtype=torch.long)
            dim, red = inp["dim"], cond["reduce"]
            if cond["op"] == "index_reduce":
                fn = lambda a, b: torch.index_reduce(a, dim, idx, b, red, include_self=False)  # noqa: E731
            else:
                fn = lambda a, b: torch.scatter_reduce(a, dim, idx, b, red, include_self=False)  # noqa: E731
            ok = torch.autograd.gradcheck(fn, (s, src), raise_exception=False)
            spec = R.load_spec("index", cond["id"], seed, "base", "float64")
            ties = bool(spec and spec.get("ties"))
            rows.append({"condition": cond["id"], "seed": seed, "op": cond["op"], "reduce": red,
                         "gradcheck_passes": bool(ok), "ties_among_participants": ties})
    by = {}
    for r in rows:
        k = "ties" if r["ties_among_participants"] else "no_ties"
        b = by.setdefault(k, {"seeds": 0, "gradcheck_fails": 0})
        b["seeds"] += 1
        b["gradcheck_fails"] += int(not r["gradcheck_passes"])
    cond_fail = {}
    for r in rows:
        cond_fail.setdefault(r["condition"], False)
        cond_fail[r["condition"]] |= not r["gradcheck_passes"]
    return {"conditions": len(conds), "conditions_with_a_gradcheck_failure": sum(cond_fail.values()),
            "by_seed": by, "rows": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gc = gradcheck_b020()
    res = {"sources": SRC, "review": REVIEW, "gradcheck_on_B020": gc}
    (OUT / "w5_review.json").write_text(json.dumps(res, indent=1) + "\n")
    print("B020 conditions:", gc["conditions"], "with a gradcheck failure:", gc["conditions_with_a_gradcheck_failure"], gc["by_seed"])


if __name__ == "__main__":
    main()
