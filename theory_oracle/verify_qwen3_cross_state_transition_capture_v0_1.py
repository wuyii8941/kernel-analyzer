#!/usr/bin/env python
"""Jointly verify a replayed transition snapshot and its fresh compiled anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from theory_oracle.verify_qwen3_grpo_transition_snapshot_v0_1 import audit


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def step29(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("optimizer_step", -1)) == 29]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--grad-states", required=True)
    parser.add_argument("--historical-grad-states")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    snapshot_audit = audit(snapshot_dir)
    fresh_rows = step29(read_jsonl(Path(args.grad_states).resolve()))
    fresh = fresh_rows[0] if len(fresh_rows) == 1 else {}
    compile_audit = fresh.get("compile_audit", {})
    anchor_gates = {
        "exactly_one_step29_anchor": len(fresh_rows) == 1,
        "policy_iteration_two": int(fresh.get("policy_iteration", -1)) == 2,
        "candidate_identity_valid": fresh.get("candidate_identity_valid") is True,
        "all_outputs_require_grad": fresh.get("all_outputs_require_grad") is True,
        "gradients_preserved": fresh.get("gradients_preserved") is True,
        "tensor_versions_preserved": fresh.get("tensor_versions_preserved") is True,
        "trainer_steps_preserved": fresh.get("trainer_steps_preserved") is True,
        "rng_restored_exactly": fresh.get("rng_restored_exactly") is True,
        "eager_repeat_exact": float(fresh.get("ref_self_max_abs", float("inf"))) == 0.0,
        "compiled_repeat_exact": float(fresh.get("alt_self_max_abs", float("inf"))) == 0.0,
        "compiled_measured_calls_present": int(
            compile_audit.get("first_runtime_invocations", 0)
        )
        > 0
        and int(compile_audit.get("second_runtime_invocations", 0)) > 0,
    }

    historical_diagnostic: dict[str, Any] | None = None
    if args.historical_grad_states:
        historical_rows = step29(read_jsonl(Path(args.historical_grad_states).resolve()))
        if len(historical_rows) == 1 and fresh:
            historical = historical_rows[0]
            fields = (
                "ref_first_sha256",
                "ref_second_sha256",
                "alt_first_sha256",
                "alt_second_sha256",
            )
            historical_diagnostic = {
                "exactly_one_historical_step29_anchor": True,
                "scorer_hash_fields_exact": {
                    field: fresh.get(field) == historical.get(field) for field in fields
                },
                "batch_size_exact": fresh.get("batch_size") == historical.get("batch_size"),
                "completion_length_exact": fresh.get("completion_length")
                == historical.get("completion_length"),
                "not_a_validity_gate": True,
            }
        else:
            historical_diagnostic = {
                "exactly_one_historical_step29_anchor": False,
                "not_a_validity_gate": True,
            }

    gates = {
        "snapshot_valid": snapshot_audit.get("valid") is True,
        "fresh_anchor_valid": all(anchor_gates.values()),
    }
    payload = {
        "schema_version": "forkcert.qwen3-cross-state-transition-capture-verification.v0.1",
        "status": "VALID_CROSS_STATE_TRANSITION_CAPTURE"
        if all(gates.values())
        else "INVALID_CAPTURE",
        "snapshot_dir": str(snapshot_dir),
        "grad_states": str(Path(args.grad_states).resolve()),
        "gates": gates,
        "anchor_gates": anchor_gates,
        "fresh_step29_anchor": fresh,
        "historical_reproducibility_diagnostic": historical_diagnostic,
        "snapshot_audit": snapshot_audit,
        "claim_limits": [
            "snapshot and candidate-anchor validity is not population representativeness",
            "historical trajectory equality is diagnostic rather than required",
            "capture validity is not operator-effect transport",
            "eager is a baseline rather than correctness authority",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    if payload["status"] != "VALID_CROSS_STATE_TRANSITION_CAPTURE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
