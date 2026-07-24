#!/usr/bin/env python
"""Check whether explicit random-sample plumbing preserves #141538's witness.

This is a guard for a prospective reference-substitution reducer: it may only
operate on a simplified functional graph if that graph preserves the original
semantic-contract violation under the same runtime/version.
"""
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
    # FractionalMaxPool consumes one horizontal and one vertical sample per
    # batch/channel, making the random dependency an explicit boundary input.
    samples = torch.rand(1, 1, 2, device="cuda", dtype=torch.float32)

    def program(inp, random_samples):
        return F.fractional_max_pool2d(
            inp, kernel_size=(1, 1), output_ratio=(0.5, 0.5),
            _random_samples=random_samples,
        )

    def invoke(backend):
        torch._dynamo.reset()
        fn = program if backend is None else torch.compile(program, backend=backend, fullgraph=True)
        result = fn(x.clone(), samples.clone())
        torch.cuda.synchronize()
        return result

    eager = invoke(None)
    compiled = [invoke("inductor"), invoke("inductor")]
    record = {
        "schema_version": "forkcert.explicit-random-boundary-screen.v0.1",
        "environment": {"torch": torch.__version__, "triton": __import__("triton").__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "input_sha256": fingerprint(x), "random_samples_sha256": fingerprint(samples),
        "eager_sha256": fingerprint(eager),
        "compiled": [{"sha256": fingerprint(value), "equal_eager": bool(torch.equal(value, eager)),
                      "max_abs": float((value.float() - eager.float()).abs().max().item())}
                     for value in compiled],
        "repeatable": bool(torch.equal(compiled[0], compiled[1])),
    }
    record["preserves_original_witness"] = bool(
        record["repeatable"] and not record["compiled"][0]["equal_eager"]
    )
    out = Path("results/historical_candidate_screen/fractional_maxpool_explicit_samples_141538_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "screen.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
