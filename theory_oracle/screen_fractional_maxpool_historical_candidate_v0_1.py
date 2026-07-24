#!/usr/bin/env python
"""Qualification screen for the historical #141538 Inductor-lowering witness.

This script binds the public reproducer to one pre-fix release.  It is only a
screen: a subsequent blind package must hide patch and issue material from the
locator before any Phase-3 score is allowed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint(tensor):
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    import torch
    import torch.nn as nn
    from torch._inductor import config

    if not torch.cuda.is_available():
        raise RuntimeError("this historical witness requires CUDA")
    config.fallback_random = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(141538)
    x = torch.randn(1, 1, 10, 10, device="cuda", dtype=torch.float32)

    def fresh_model():
        return nn.FractionalMaxPool2d(kernel_size=(1, 1), output_ratio=(0.5, 0.5)).eval().cuda()

    def invoke(stage: str):
        torch._dynamo.reset()
        model = fresh_model()
        fn = model if stage == "eager" else torch.compile(model, backend=stage, fullgraph=True)
        torch.manual_seed(0)
        out = fn(x.clone())
        torch.cuda.synchronize()
        return out

    # The eager reference is evaluated under the same reset RNG protocol as
    # every candidate arm.  It is a declared comparison baseline for this
    # historical contract, not a universal mathematical truth claim.
    reference = invoke("eager")
    stages = {}
    for stage_id, backend in (("eager", "eager"), ("dynamo_eager", "eager"),
                              ("aot_eager", "aot_eager"), ("inductor", "inductor")):
        values = [invoke(backend), invoke(backend)]
        rows = []
        for value in values:
            diff = (value.float() - reference.float()).abs()
            rows.append({
                "contract_holds": bool(torch.equal(value, reference)),
                "max_abs": float(diff.max().item()),
                "output_sha256": fingerprint(value),
            })
        stages[stage_id] = rows
    summary = {
        stage: {
            "repeatable": rows[0] == rows[1],
            "contract_holds": rows[0]["contract_holds"],
            "max_abs": rows[0]["max_abs"],
        }
        for stage, rows in stages.items()
    }
    repeatable = all(row["repeatable"] for row in summary.values())
    lower = bool(
        summary["eager"]["contract_holds"]
        and summary["dynamo_eager"]["contract_holds"]
        and summary["aot_eager"]["contract_holds"]
        and not summary["inductor"]["contract_holds"]
        and repeatable
    )
    output = Path("results/historical_candidate_screen/fractional_maxpool_141538_v0_1")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "forkcert.historical-candidate-screen.v0.1",
        "case": "pytorch_issue_141538_fractional_maxpool_lowering",
        "role": "candidate_qualification_only",
        "environment": {"torch": torch.__version__, "triton": __import__("triton").__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "semantic_contract": "under identical reset RNG, compiled FractionalMaxPool2d output must equal declared eager baseline",
        "input_sha256": fingerprint(x), "reference_sha256": fingerprint(reference),
        "stages": stages, "stage_summary": summary,
        "qualification": "QUALIFIED_LOWER_GPU_CANDIDATE" if lower else "REJECTED_ON_BOUND_RUNTIME",
        "not_a_claim": ["blind localization", "external patch validation", "root cause", "first bad pass"],
    }
    (output / "screen.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualification": record["qualification"], "stage_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
