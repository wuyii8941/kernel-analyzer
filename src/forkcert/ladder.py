from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class LadderLevel:
    level: str
    variable: str
    mechanism: str
    enabled_ref: str
    enabled_alt: str
    notes: str = ""


@dataclass
class AttributionRow:
    level: str
    variable: str
    mechanism: str
    first_observed_diff_l2: float
    max_activation_diff_l2: float
    propagation_gain_first_to_last: float | None
    final_logprob_delta: float
    relative_to_composite_percent: float | None
    additive_attribution_valid: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_LEVELS = [
    LadderLevel("L1", "attention backend", "algorithm_structure", "SDPA math", "SDPA flash/efficient"),
    LadderLevel("L2", "RMSNorm fused/unfused", "materialization_points", "model default", "pure PyTorch reference"),
    LadderLevel("L3", "intermediate materialization", "materialization_points", "native", "bf16 roundtrip hooks"),
    LadderLevel("L4", "log_softmax precision", "rounding_precision", "fp32 upcast", "no fp32 upcast"),
    LadderLevel("L5", "bf16 matmul reduction precision", "reduction_precision", "allow reduced precision off", "on"),
    LadderLevel("L6", "torch.compile", "mixed", "eager", "compile"),
]


def attribution_from_measurements(rows: list[dict[str, Any]]) -> list[AttributionRow]:
    """Build attribution rows from measured ladder JSON.

    Expected input fields per row include paired activation-difference summaries and final_logprob_delta.
    Levels are independent sensitivity experiments, so their ratios are not additive attribution percentages.
    """
    composite = next(
        (abs(float(row.get("final_logprob_delta", 0.0))) for row in rows if str(row.get("level")) == "L6"),
        0.0,
    )
    output: list[AttributionRow] = []
    for row in rows:
        delta = abs(float(row.get("final_logprob_delta", 0.0)))
        output.append(
            AttributionRow(
                level=str(row["level"]),
                variable=str(row["variable"]),
                mechanism=str(row.get("mechanism", "unknown")),
                first_observed_diff_l2=float(row.get("first_observed_diff_l2", 0.0)),
                max_activation_diff_l2=float(row.get("max_activation_diff_l2", 0.0)),
                propagation_gain_first_to_last=(
                    float(row["propagation_gain_first_to_last"])
                    if row.get("propagation_gain_first_to_last") is not None
                    else None
                ),
                final_logprob_delta=float(row.get("final_logprob_delta", 0.0)),
                relative_to_composite_percent=100.0 * delta / composite if composite > 0 else None,
            )
        )
    return output
