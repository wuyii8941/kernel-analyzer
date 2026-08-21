"""Value-blind identities for de-duplicating implementation measurements."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _canonical_contracts(
    contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    fields = ("shape", "stride", "dtype", "device_type", "layout", "storage_offset")
    return {
        str(name): {field: value.get(field) for field in fields}
        for name, value in sorted(contracts.items())
    }


def _rank_layout_pattern(contract: Mapping[str, Any]) -> dict[str, Any]:
    shape = list(contract.get("shape", []))
    stride = list(contract.get("stride", []))
    contiguous = bool(shape) and stride == _contiguous_stride(shape)
    return {
        "rank": len(shape),
        "dtype": contract.get("dtype"),
        "layout": contract.get("layout"),
        "contiguous": contiguous,
        "has_zero_extent": any(int(value) == 0 for value in shape),
    }


def _contiguous_stride(shape: Sequence[int]) -> list[int]:
    result: list[int] = []
    running = 1
    for extent in reversed(shape):
        result.append(running)
        running *= max(int(extent), 1)
    return list(reversed(result))


def _symbol_pattern(symbol: str) -> str:
    return re.sub(r"(?:_\d+)+$", "", symbol)


def build_implementation_identity(
    *,
    backend: str,
    implementation_kind: str,
    phase: str,
    operation: str,
    operand_contracts: Mapping[str, Mapping[str, Any]],
    program_digest: str | None = None,
    semantic_operations: Sequence[str] = (),
    fusion_boundary: Sequence[str] = (),
    launch_contract: Mapping[str, Any] | None = None,
    structural_program_digest: str | None = None,
) -> dict[str, Any]:
    """Return exact, pattern and semantic identities without tensor values."""

    exact_payload = {
        "backend": backend,
        "implementation_kind": implementation_kind,
        "phase": phase,
        "operation": operation,
        "program_digest": program_digest,
        "operand_contracts": _canonical_contracts(operand_contracts),
        "launch_contract": dict(sorted((launch_contract or {}).items())),
        "fusion_boundary": sorted(map(str, fusion_boundary)),
    }
    pattern_payload = {
        "backend": backend,
        "implementation_kind": implementation_kind,
        "phase": phase,
        "operation_pattern": _symbol_pattern(operation),
        "structural_program_digest": structural_program_digest,
        "semantic_operations": sorted(map(str, semantic_operations)),
        "fusion_boundary": sorted(map(str, fusion_boundary)),
        "operand_patterns": {
            name: _rank_layout_pattern(value)
            for name, value in _canonical_contracts(operand_contracts).items()
        },
    }
    semantic_payload = {
        "phase": phase,
        "semantic_operations": sorted(map(str, semantic_operations))
        or [_symbol_pattern(operation)],
    }
    return {
        "schema": "kernel-analyzer-implementation-identity-v1",
        "exact_implementation_id": _digest(exact_payload),
        "implementation_pattern_id": _digest(pattern_payload),
        "semantic_family_id": _digest(semantic_payload),
        "exact_payload": exact_payload,
        "pattern_payload": pattern_payload,
        "semantic_payload": semantic_payload,
        "candidate_tensor_values_used": False,
    }


def novelty_label(
    identity: Mapping[str, Any],
    development_identities: Sequence[Mapping[str, Any]],
) -> str:
    exact = {row["exact_implementation_id"] for row in development_identities}
    patterns = {row["implementation_pattern_id"] for row in development_identities}
    families = {row["semantic_family_id"] for row in development_identities}
    if identity["exact_implementation_id"] in exact:
        return "SEEN_EXACT_IMPL_NEW_OPERANDS"
    if identity["implementation_pattern_id"] in patterns:
        return "NEW_EXACT_IMPL_SEEN_PATTERN"
    if identity["semantic_family_id"] in families:
        return "NEW_IMPL_PATTERN"
    return "NEW_SEMANTIC_FAMILY"
