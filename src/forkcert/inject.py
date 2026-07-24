from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BugInjection:
    name: str
    description: str
    logprob_shift: float
    token_selector: str = "all"

    def to_json_dict(self) -> dict[str, Any]:
        return {**asdict(self), "injection_kind": "posthoc_logprob_shift", "claim_ready": False}


DEFAULT_BUGS = [
    BugInjection(
        name="reduction_boundary_off_by_one",
        description="Simulates dropping one term from a reduction/logsumexp denominator.",
        logprob_shift=5e-2,
    ),
    BugInjection(
        name="attention_mask_missing_column",
        description="Simulates a silent attention mask error that shifts selected token logprobs.",
        logprob_shift=-2e-2,
    ),
    BugInjection(
        name="intermediate_precision_wrong",
        description="Simulates an unintended low-precision intermediate in path_alt.",
        logprob_shift=1e-2,
    ),
]


def should_inject(token_index: int, selector: str) -> bool:
    if selector == "all":
        return True
    if selector == "even":
        return token_index % 2 == 0
    if selector == "odd":
        return token_index % 2 == 1
    raise ValueError(f"unsupported token selector: {selector}")


def inject_logprob_bug(rows: list[dict[str, Any]], injection: BugInjection) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        token_index = int(item.get("token_index", 0))
        if should_inject(token_index, injection.token_selector):
            item["logp_alt"] = float(item["logp_alt"]) + injection.logprob_shift
            item["logprob_delta"] = abs(float(item["logp_alt"]) - float(item["logp_ref"]))
            metadata = dict(item.get("metadata", {}))
            metadata["bug_injection"] = injection.to_json_dict()
            item["metadata"] = metadata
        output.append(item)
    return output
