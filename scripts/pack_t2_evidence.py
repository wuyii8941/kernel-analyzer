#!/usr/bin/env python3
"""Pack paired region replacement/sham runs into proof-unit keyed T2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _repeat_exact(repeats: list[dict[str, Any]], key: str) -> bool:
    return len(repeats) == 2 and repeats[0].get(key) == repeats[1].get(key)


def pack(queue: dict[str, Any], replacement: dict[str, Any], sham: dict[str, Any]) -> dict[str, Any]:
    if replacement.get("intervention_candidate_scale") != 0.0:
        raise ValueError("replacement artifact is not candidate_scale=0")
    if sham.get("intervention_candidate_scale") != 1.0:
        raise ValueError("sham artifact is not candidate_scale=1")
    invariant = ("checkpoint_step", "seq_len", "dtype", "tf32", "arms_file")
    if any(replacement.get(k) != sham.get(k) for k in invariant):
        raise ValueError("replacement and sham protocols differ")
    qrows = {row["region_id"]: row for row in queue["rows"]}
    repl = {row["region_id"]: row for row in replacement["arms"]}
    shams = {row["region_id"]: row for row in sham["arms"]}
    if set(repl) != set(shams):
        raise ValueError("replacement and sham arm sets differ")

    region_rows: list[dict[str, Any]] = []
    unit_rows: dict[str, list[dict[str, Any]]] = {}
    selected_units = set(queue.get("selected_proof_unit_ids", ()))
    for region_id in sorted(repl):
        qrow = qrows[region_id]
        rr, sr = repl[region_id]["repeats"], shams[region_id]["repeats"]
        if len(rr) != 2 or len(sr) != 2:
            raise ValueError("T2 requires two replacement and two sham repeats")
        parameter_names = list(replacement["carrier_parameter_names"])
        replacement_delta = max(
            float(rep["carrier"][name]["l2"])
            for rep in rr for name in parameter_names
        )
        parameter_reached = replacement_delta > 0.0
        sham_gradient_exact = all(
            float(rep["carrier"][name]["l2"]) == 0.0
            and int(rep["carrier"][name]["nonzero"]) == 0
            for rep in sr for name in parameter_names
        )
        sham_loss_exact = all(
            float(rep["loss"]) == float(sham["baseline_candidate_loss"])
            for rep in sr
        )
        declared_endpoints = set(qrow.get("endpoints", ()))
        observed_endpoints = set(rr[0]["record"].get("intervened_endpoints", ()))
        full_region_replacement = bool(declared_endpoints) and observed_endpoints == declared_endpoints
        repeat_exact = rr[0]["carrier"] == rr[1]["carrier"] and sr[0]["carrier"] == sr[1]["carrier"]
        row = {
            "region_id": region_id,
            "proof_unit_ids": list(qrow["proof_unit_ids"]),
            "replacement_exact": bool(
                full_region_replacement
                and all(rep["gates"]["region_observed"] for rep in rr)
            ),
            "sham_exact": sham_gradient_exact and sham_loss_exact,
            "parameter_reached": parameter_reached,
            # Every declared output was replaced, so there are no unaccounted
            # same-region output endpoints. Other regions run unmodified.
            "non_target_endpoints_exact": full_region_replacement,
            "delta_norm": replacement_delta,
            "repeat_exact": repeat_exact,
            "replacement_loss": float(rr[0]["loss"]),
            "candidate_loss": float(replacement["baseline_candidate_loss"]),
            "intervened_endpoints": sorted(observed_endpoints),
            "natural": True,
        }
        row["status"] = "PASS" if all((
            row["replacement_exact"], row["sham_exact"], row["parameter_reached"],
            row["non_target_endpoints_exact"], row["repeat_exact"], row["delta_norm"] > 0,
        )) else "FAIL"
        region_rows.append(row)
        for unit_id in row["proof_unit_ids"]:
            if selected_units and unit_id not in selected_units:
                continue
            unit_rows.setdefault(unit_id, []).append(row)

    packed_units = {}
    for unit_id, rows in unit_rows.items():
        # A proof unit touching multiple candidate regions passes only after
        # every region in its mapping has been intervened successfully.
        expected = set()
        for qrow in queue["rows"]:
            if unit_id in qrow["proof_unit_ids"]:
                expected.add(qrow["region_id"])
        observed = {row["region_id"] for row in rows}
        complete = observed == expected
        passed = complete and all(row["status"] == "PASS" for row in rows)
        packed_units[unit_id] = {
            "status": "PASS" if passed else "UNRESOLVED" if not complete else "FAIL",
            "replacement_exact": passed,
            "sham_exact": passed,
            "parameter_reached": passed,
            "non_target_endpoints_exact": passed,
            "delta_norm": max(row["delta_norm"] for row in rows),
            "candidate_region_ids": sorted(observed),
            "expected_candidate_region_ids": sorted(expected),
            "mapping_complete": complete,
            "natural": True,
        }

    payload: dict[str, Any] = {
        "schema": "kernel-analyzer-t2-causal-evidence-v1",
        "region_rows": region_rows,
        "unit_rows": packed_units,
        "gates": {
            "paired_replacement_and_restoration_sham": True,
            "two_exact_repeats_each": all(row["repeat_exact"] for row in region_rows),
            "tensor_values_retained": False,
        },
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--sham", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = pack(*[json.loads(path.read_text()) for path in (
        args.queue, args.replacement, args.sham
    )])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "regions": len(payload["region_rows"]),
        "region_pass": sum(row["status"] == "PASS" for row in payload["region_rows"]),
        "unit_pass": sum(row["status"] == "PASS" for row in payload["unit_rows"].values()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
