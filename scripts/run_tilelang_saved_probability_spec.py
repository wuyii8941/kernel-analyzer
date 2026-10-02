"""Check a probability-mass specification on actual TileLang reconstruction.

Both conditions run the same kernel. Only the source of normalization
statistics changes: original FP32 scores versus the stored BF16 scores.
This is a controlled mechanism reproduction, not an external library bug.
"""

import argparse
import json
from pathlib import Path

import torch
import tilelang
import tilelang.language as T

from kernel_analyzer import check_tilelang_softmax_saved_state


@tilelang.jit
def reconstruct(scores, maximum, denominator):
    N = T.const("N")
    scores: T.Tensor((N,), T.bfloat16)
    maximum: T.Tensor((1,), T.float32)
    denominator: T.Tensor((1,), T.float32)
    out = T.empty((N,), T.float32)
    with T.Kernel(1, threads=128):
        for i in T.Parallel(N):
            out[i] = T.exp(T.cast(scores[i], "float32") - maximum[0]) / denominator[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    protocol_path = output.with_suffix(".protocol.json")
    if not output.is_relative_to(root) or output.exists() or protocol_path.exists() or args.samples < 2:
        parser.error("use new output inside repository and samples >= 2")
    protocol = {
        "schema": "tilelang-probability-mass-spec-v1",
        "data_use": "CONTROLLED_DEVELOPMENT_NOT_NATURAL_TRAINING",
        "specification": "sum of stored reconstructed probabilities equals one up to declared tolerance",
        "tolerance": 1e-5,
        "distribution": "128 independent N(0,4^2) FP32 scores, then stored in BF16",
        "seed_start": 92200,
        "samples": args.samples,
        "conditions": ["statistics_from_original_scores", "statistics_from_stored_scores"],
        "intervention": "same BF16 scores and same kernel, recompute max and denominator from stored scores",
        "property_is_necessary_not_sufficient": True,
        "inference": "property diagnostics and exploratory conditional intervals; no full mean-bias or training verdict",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    def inputs(index, consistent):
        generator = torch.Generator(device="cuda").manual_seed(protocol["seed_start"] + index)
        scores = torch.randn(128, device="cuda", generator=generator) * 4.
        stored = scores.bfloat16()
        source = stored.float() if consistent else scores
        maximum = source.max().reshape(1)
        denominator = (source - maximum).exp().sum().reshape(1)
        return stored, maximum, denominator

    reports = {}
    for consistent, name in enumerate(protocol["conditions"]):
        report = check_tilelang_softmax_saved_state(
            reconstruct, lambda i: inputs(i, bool(consistent)),
            samples=args.samples, tolerance=protocol["tolerance"],
        )
        reports[name] = report
        mass = report["input_consistency"]
        print(json.dumps({"condition": name, "measurement": report["measurement_status"],
                          "failed_count": mass.get("failed_count"),
                          "mean_mass_defect": mass.get("property_residual_mean"),
                          "bias_decision": report["bias_decision"]}), flush=True)
    output.write_text(json.dumps({"protocol": protocol, "results": reports,
                                 "environment": {"tilelang": tilelang.__version__,
                                                 "device": torch.cuda.get_device_name()}},
                                indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
