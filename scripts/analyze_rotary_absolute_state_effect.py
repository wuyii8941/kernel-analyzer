"""Recompute absolute state effects from recorded historical captures."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze(summary_path):
    summary = json.loads(summary_path.read_text())
    conditions = {}
    folders = (
        "ministral_fused_rotary_highpos_cold_matched_profile_v1",
        "ministral_fused_rotary_highpos_warm8_profile_v1",
        "ministral_fused_rotary_highpos_warm8_reset_profile_v1",
    )
    for folder in folders:
        path = ROOT / "results/property/numerical_coverage_v1" / folder / "internal_forward_115_out_ptr0-attention-position-scaling-common-input.json"
        raw = path.read_bytes()
        document = json.loads(raw)
        rows = document["original_coordinate_statistics"]["PARAMETER_WRITE"][16:32]
        if len(rows) != 16:
            raise ValueError("expected 16 confirmation states")
        x = math.fsum(row["effect_energy"] for row in rows) / len(rows)
        b = math.fsum(row["repair_energy"] for row in rows) / len(rows)
        a = math.fsum(row["effect_repair_inner_product"] for row in rows) / len(rows)
        if not all(math.isfinite(v) for v in (x, b, a)) or x < 0 or b <= 0:
            raise ValueError("invalid original-coordinate statistics")
        key = "reset" if "reset" in path.parent.name else "warm" if "warm8" in path.parent.name else "cold"
        if key in conditions:
            raise ValueError("duplicate condition")
        conditions[key] = dict(effect_norm_rms=math.sqrt(x), reference_norm_rms=math.sqrt(b),
                               relative_rms=math.sqrt(x / b), inner_mean=a,
                               source=str(path.relative_to(ROOT)))
    warm, reset = conditions["warm"], conditions["reset"]
    return dict(schema="rotary-absolute-state-effect-v1", data_use="HISTORICAL_REANALYSIS",
                scope="FIXED_CONFIRMATION_SUITE_ONE_PARAMETER", conditions=conditions,
                reset_over_warm={key: reset[key] / warm[key] for key in
                                 ("effect_norm_rms", "reference_norm_rms", "relative_rms")},
                interpretation=["Absolute effect changes; denominator alone does not explain the ratio change.",
                                "Reset also sets prior_step to zero; moments alone are not isolated.",
                                "No population or loss consequence follows from this comparison."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside the repository")
    result = analyze(ROOT / "results/property/numerical_coverage_v1/ministral_fused_rotary_optimizer_condition_summary_v1.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["reset_over_warm"], indent=2))
