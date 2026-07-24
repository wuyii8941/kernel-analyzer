from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "forkcert.v2.0"


@dataclass
class ForkCertificate:
    case_id: str
    token_index: int
    token_id: int | None
    token_text: str | None
    path_ref: str
    path_alt: str
    logp_ref: float
    logp_alt: float
    old_logp: float
    advantage_sign: int
    eps: float
    logprob_delta: float
    delta_self_ref: float | None
    delta_self_alt: float | None
    clip_boundary: float
    clip_margin: float
    clip_ref: bool
    clip_alt: bool
    delta_bound_legal: float | None
    region: str
    fork_possible: bool
    actual_fork: bool
    grad_contribution_ref: float | None = None
    grad_contribution_alt: float | None = None
    grad_contribution_diff: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def metadata_base(**kwargs: Any) -> dict[str, Any]:
    data = {"schema_version": SCHEMA_VERSION}
    data.update(kwargs)
    return data

