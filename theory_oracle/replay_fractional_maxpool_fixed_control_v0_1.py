#!/usr/bin/env python
"""Post-reveal fixed-runtime control for the #141538 functional witness."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(141538)
    x = torch.randn(1, 1, 10, 10, device="cuda", dtype=torch.float32)
    samples = torch.rand(1, 1, 2, device="cuda", dtype=torch.float32)

    def program(value, random_samples):
        return F.fractional_max_pool2d(
            value, kernel_size=(1, 1), output_ratio=(0.5, 0.5),
            _random_samples=random_samples,
        )

    eager = program(x.clone(), samples.clone())
    torch._dynamo.reset()
    compiled = torch.compile(program, backend="inductor", fullgraph=True)
    outputs = [compiled(x.clone(), samples.clone()), compiled(x.clone(), samples.clone())]
    torch.cuda.synchronize()
    record = {
        "schema_version": "forkcert.historical-fixed-runtime-control.v0.1",
        "case_id": "pytorch_fractional_maxpool_lowering",
        "environment": {"torch": torch.__version__, "triton": __import__("triton").__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "input_sha256": fingerprint(x), "random_samples_sha256": fingerprint(samples),
        "eager_sha256": fingerprint(eager),
        "compiled": [{"sha256": fingerprint(value), "equal_eager": bool(torch.equal(value, eager)),
                      "max_abs": float((value.float() - eager.float()).abs().max().item())}
                     for value in outputs],
        "repeatable": bool(torch.equal(outputs[0], outputs[1])),
    }
    record["contract_clears"] = bool(
        record["repeatable"] and record["compiled"][0]["equal_eager"]
    )
    out = Path("results/historical_post_reveal/fractional_maxpool_141538_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixed_runtime_control.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
