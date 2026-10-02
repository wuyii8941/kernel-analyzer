"""Validate a built-in sum specification against a simpler symmetry property.

These are controlled inputs to actual TileLang FP32 reduction kernels. They
are not natural training cases or newly discovered library bugs. Enumerating
the complete finite support identifies its signed mean without a sampling
approximation. Independent sampling separately exercises the shared checker.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import time

import torch
import tilelang

from kernel_analyzer import SumSpec, check_tilelang_bias, check_tilelang_odd_symmetry
from scripts.tilelang_reduction_family_smoke import reduce_sum_forward, reduce_sum_interleaved


def make_bank(symmetric):
    bank = []
    for index in range(16):
        values = torch.empty(128, dtype=torch.float16, device="cuda")
        values[:32] = 65504.
        values[32:64] = -65504.
        values[64:96] = 0.1
        values[96:] = 0.1 + index * 0.01
        bank.append(values)
    return bank + [-values for values in bank] if symmetric else bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        parser.error("output must be a new path inside the repository")
    protocol_path = output.with_suffix(".protocol.json")
    if protocol_path.exists() or args.samples < 16:
        parser.error("protocol must be new and samples must be at least 16")
    spec = SumSpec(keepdim=True)
    draws = {}
    for symmetric in (False, True):
        key = "symmetric" if symmetric else "asymmetric"
        rng = random.Random(92100 + int(symmetric))
        draws[key] = [rng.randrange(32 if symmetric else 16) for _ in range(args.samples)]
    protocol = {
        "schema": "tilelang-sum-spec-experiment-v2",
        "purpose": "CONTROLLED_SPECIFICATION_VALIDATION_NOT_NATURAL_TRAINING",
        "data_use": "DEVELOPMENT_REUSES_KNOWN_REDUCTION_INPUT_PATTERN",
        "specification": spec.contract(),
        "candidates": ["reduce_sum_forward", "reduce_sum_interleaved"],
        "input_bank": {
            "definition": "FP16 [32*65504,32*(-65504),32*0.1,32*(0.1+0.01*i)], i=0..15",
            "distributions": "uniform base bank and uniform base-plus-negation bank",
            "arithmetic": "FP16 stored inputs, FP32 accumulator and output",
        },
        "sampling": "independent uniform with replacement; calibration is first quarter",
        "draws": draws,
        "samples": args.samples,
        "family_alpha": 0.05,
        "case_alpha": 0.05 / 4,
        "directional_margin": 0.0,
        "aligned_projection_margin": 0.0,
        "hypotheses": [
            "odd symmetry can hold while mean error on an asymmetric input distribution is nonzero",
            "complete sign symmetrization cancels signed mean but need not remove error energy or aligned effects",
        ],
        "notes": "Shared Student intervals remain conditional approximations; census is exact for this finite support only.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    start = time.perf_counter()
    results = {}
    for symmetric in (False, True):
        key = "symmetric" if symmetric else "asymmetric"
        bank = make_bank(symmetric)
        for name, candidate in (("forward", reduce_sum_forward), ("interleaved", reduce_sum_interleaved)):
            census = []
            fractions = []
            for index, values in enumerate(bank):
                observed = float(candidate(values).item())
                expected = float(spec.reference(values).item())
                error = Fraction.from_float(observed) - Fraction.from_float(expected)
                fractions.append(error)
                census.append({"index": index, "candidate": observed, "exact_sum": expected,
                               "signed_error": float(error)})
            exact_mean = sum(fractions, Fraction(0)) / len(fractions)
            exact_energy = sum((e * e for e in fractions), Fraction(0)) / len(fractions)
            report = check_tilelang_bias(
                candidate, make_inputs=lambda i: (bank[draws[key][i]],), specification=spec,
                samples=args.samples, calibration_samples=args.samples // 4,
                alpha=protocol["case_alpha"], directional_margin=protocol["directional_margin"],
                aligned_projection_margin=protocol["aligned_projection_margin"],
            )
            symmetry = check_tilelang_odd_symmetry(candidate, lambda i: (bank[i],),
                                                   samples=len(bank), tolerance=0.)
            results[f"{name}_{key}"] = {
                "census": census,
                "finite_support_signed_mean": float(exact_mean),
                "finite_support_signed_mean_rational": str(exact_mean),
                "finite_support_mean_error_nonzero": exact_mean != 0,
                "finite_support_error_energy": float(exact_energy),
                "symmetry": symmetry,
                "sampled_bias_report": report,
            }
            print(json.dumps({"case": f"{name}_{key}", "exact_bank_mean": float(exact_mean),
                              "energy": float(exact_energy), "symmetry": symmetry["property_decision"],
                              "checker": report["status"],
                              "mean_bias": report.get("output", {}).get("mean_bias_decision"),
                              "aligned_effect": report.get("output", {}).get("aligned_effect_decision"),
                              "reasons": report.get("output", {}).get("systematic_bias_reasons")}), flush=True)
    payload = {
        "protocol": protocol,
        "environment": {"torch": torch.__version__, "tilelang": tilelang.__version__,
                        "device": torch.cuda.get_device_name()},
        "elapsed_seconds": time.perf_counter() - start,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
