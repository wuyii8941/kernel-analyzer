from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TwinStep:
    optimizer_step: int
    case_id: str
    fork_count: int
    max_logprob_delta: float
    weight_divergence: float | None
    relative_weight_divergence: float | None
    interval_had_fork: bool | None

    def to_json_dict(self) -> dict:
        return asdict(self)


def trajectory_summary(rows: list[dict]) -> dict:
    measured = [row for row in rows if row.get("weight_divergence") is not None]
    fork_events = sum(int(row.get("fork_count", 0)) for row in rows)
    fork_steps = sum(1 for row in rows if int(row.get("fork_count", 0)) > 0)
    fork_jumps = []
    no_fork_jumps = []
    for previous, current in zip(measured, measured[1:]):
        jump = float(current["weight_divergence"]) - float(previous["weight_divergence"])
        if current.get("interval_had_fork"):
            fork_jumps.append(jump)
        else:
            no_fork_jumps.append(jump)
    return {
        "optimizer_steps": max((int(row.get("optimizer_step", 0)) for row in rows), default=0),
        "trajectory_rows": len(rows),
        "weight_measurements": len(measured),
        "total_fork_events": fork_events,
        "fork_steps": fork_steps,
        "final_weight_divergence": float(measured[-1]["weight_divergence"]) if measured else None,
        "final_relative_weight_divergence": float(measured[-1]["relative_weight_divergence"]) if measured else None,
        "mean_divergence_jump_fork_intervals": sum(fork_jumps) / len(fork_jumps) if fork_jumps else None,
        "mean_divergence_jump_no_fork_intervals": sum(no_fork_jumps) / len(no_fork_jumps) if no_fork_jumps else None,
        "fork_intervals": len(fork_jumps),
        "no_fork_intervals": len(no_fork_jumps),
    }
