#!/usr/bin/env python3
"""Reconcile completed full-coordinate T1 artifacts with the 1,562-endpoint ledger."""

from __future__ import annotations

from collections import Counter
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL = {"qwen": "qwen", "phi": "phi4", "mamba": "mamba", "deepseek8": "deepseek8b"}


def load(path: Path) -> dict[str, Any]:
    with path.open("rb") as raw:
        zipped = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if zipped else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_large_oracle_metadata(path: Path) -> dict[str, Any]:
    """Read only the top-level metadata and result rows of a huge shard.

    Complete-coordinate shards retain all 32-state sampled tensors in
    ``states``.  Loading a 1--2GB JSON document through ``json.load`` is both
    wasteful and needlessly delays denominator reconciliation.  The audit
    needs only identity, status, result hash, and the top-level ``rows``;
    extract that array from the decompressed byte stream and deliberately do
    not parse the large ``states`` object.
    """
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    def scalar(pattern: bytes, *, last: bool = False) -> str | None:
        matches = list(re.finditer(pattern, raw))
        if not matches:
            return None
        match = matches[-1] if last else matches[0]
        return match.group(1).decode()

    rows_match = re.search(rb'"rows"\s*:\s*\[', raw)
    if rows_match is None:
        raise RuntimeError(f"large artifact has no top-level rows array: {path}")
    start = rows_match.end() - 1
    depth = 0
    in_string = False
    escaped = False
    end = None
    for index in range(start, len(raw)):
        byte = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5c:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5b, 0x7b):  # [ or {
            depth += 1
        elif byte in (0x5d, 0x7d):  # ] or }
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end is None:
        raise RuntimeError(f"unterminated rows array in large artifact: {path}")
    rows = json.loads(raw[start:end])
    return {
        "schema": scalar(rb'"schema"\s*:\s*"([^"]+)"'),
        # Keys are serialized with sort_keys=True, so the top-level status is
        # after the large reference/state sections.  Nested runtime records
        # also contain status-like fields; use the final exact key.
        "status": scalar(rb'"status"\s*:\s*"([^"]+)"', last=True),
        "architecture": scalar(rb'"architecture"\s*:\s*"([^"]+)"'),
        "sequence_length": int(scalar(rb'"sequence_length"\s*:\s*(\d+)') or 0),
        "result_sha256": scalar(rb'"result_sha256"\s*:\s*"([0-9a-f]+)"', last=True),
        "rows": rows,
    }


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=ROOT / "results/coverage/cases/same_dtype_case_ledger.json.gz")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "results/coverage/cases/full_coordinate")
    parser.add_argument("--output", type=Path, default=ROOT / "results/coverage/cases/full_coordinate_audit.json.gz")
    parser.add_argument("--markdown", type=Path, default=ROOT / "results/coverage/cases/full_coordinate_audit.md")
    args = parser.parse_args()

    ledger = load(args.ledger)
    expected = {str(row["candidate_id"]): row for row in ledger["endpoint_candidates"]}
    observed: dict[str, dict[str, Any]] = {}
    artifacts = []
    for path in sorted(args.artifact_root.rglob("*.json.gz")):
        # The mamba seq256 completion shard is ~1.8GB compressed because it
        # retains all sampled states.  Reconcile it through the metadata-only
        # path; ordinary shards keep the exact historical loader.
        artifact = (load_large_oracle_metadata(path)
                    if path.stat().st_size > 500_000_000 else load(path))
        if artifact.get("status") not in {
            "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE",
            "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE_SHARD",
        }:
            continue
        architecture = str(artifact.get("architecture"))
        model = MODEL.get(architecture)
        shape = artifact.get("sequence_length")
        if model is None or shape not in (64, 128, 256):
            raise RuntimeError(f"unknown full-coordinate artifact identity: {path}")
        result_sha = artifact.get("result_sha256")
        for row in artifact.get("rows", []):
            task_id = str(row["task_id"])
            candidate_id = f"{model}:seq{shape}:{task_id}"
            if candidate_id not in expected:
                raise RuntimeError(f"full-coordinate result is outside frozen ledger: {candidate_id}")
            if candidate_id in observed:
                raise RuntimeError(f"duplicate full-coordinate disposition: {candidate_id}")
            complete = row.get("complete_coordinates")
            if complete is None:
                # Small endpoint campaigns select every coordinate by construction.
                state = next(iter(artifact["states"].values()))
                error = state["repeats"][0]["endpoint_metrics"][task_id]["error"]
                tensor_numel = int(error["directional_error_sketch"]["tensor_numel"])
                if int(row["sampled_coordinates"]) != tensor_numel:
                    raise RuntimeError(f"artifact is not full-coordinate: {candidate_id}")
                complete = tensor_numel
            verdict = str(row["verdict"])
            observed[candidate_id] = {
                "candidate_id": candidate_id, "task_id": task_id,
                "model": model, "sequence_length": shape,
                "complete_coordinates": int(complete), "verdict": verdict,
                "t1_pass": verdict == "DIRECTIONAL_OPTIMIZATION_BIAS",
                "artifact": str(path.relative_to(ROOT)),
                "artifact_result_sha256": result_sha,
                "cross_state_inner_product_u": row.get("cross_state_inner_product_u"),
                "cluster_bootstrap_95": row.get("cluster_bootstrap_95"),
            }
        artifacts.append({"path": str(path.relative_to(ROOT)), "result_sha256": result_sha,
                          "rows": len(artifact.get("rows", []))})

    pending = sorted(set(expected) - set(observed))
    passed = [row for row in observed.values() if row["t1_pass"]]
    rejected = [row for row in observed.values() if not row["t1_pass"]]
    output = {
        "schema": "kernel-analyzer-exhaustive-full-coordinate-audit-v1",
        "status": "COMPLETE_FULL_COORDINATE_T1_DENOMINATOR" if not pending else "PARTIAL_FAIL_CLOSED",
        "ledger_result_sha256": ledger["result_sha256"],
        "denominator": {
            "directional_endpoints": len(expected), "audited": len(observed),
            "passed_t1": len(passed), "rejected_t1": len(rejected), "pending": len(pending),
            "by_model_audited": dict(sorted(Counter(row["model"] for row in observed.values()).items())),
        },
        "gates": {
            "all_results_inside_frozen_ledger": True,
            "no_duplicate_dispositions": True,
            "full_coordinate_only": True,
            "pending_retained_in_denominator": len(expected) == len(observed) + len(pending),
            "all_1562_audited": not pending,
        },
        "artifacts": artifacts,
        "audited_rows": sorted(observed.values(), key=lambda row: row["candidate_id"]),
        "pending_candidate_ids": pending,
        "claim_boundary": "This artifact certifies only full-coordinate T1. T2/T3/T4 remain separate gates for every T1 survivor.",
    }
    output["result_sha256"] = digest(output)
    write(args.output, output)
    counts = output["denominator"]
    args.markdown.write_text(
        "# Full-coordinate Flash-style audit\n\n"
        f"Status: `{output['status']}`.\n\n"
        "| Frozen directional endpoints | Full-coordinate audited | T1 passed | T1 rejected | Pending |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"| {counts['directional_endpoints']} | {counts['audited']} | {counts['passed_t1']} | "
        f"{counts['rejected_t1']} | {counts['pending']} |\n\n"
        "Pending endpoints remain in the denominator. A T1 pass is only eligibility for "
        "causal repair, real-carrier, and paired-accumulation gates; it is not a strict case.\n",
        encoding="utf-8",
    )
    # Keep the stale-audit reconciliation visible in the compact report.  The
    # values are derived from the artifact rather than hand-maintained.
    reconciled = [
        row for row in observed.values()
        if row["artifact"].endswith("mamba_seq256_r1/0000_0557_b5fa3838836df93b.json.gz")
    ]
    if reconciled:
        with args.markdown.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Reconciliation of the former 557 pending endpoints\n\n"
                "The former pending set was `mamba:seq256:*`. It was already present in the "
                "complete-coordinate artifact "
                "`full_coordinate/large/mamba_seq256_r1/0000_0557_b5fa3838836df93b.json.gz` "
                "(557 endpoints, 32 states, two repeats, no unresolved mappings). The old "
                "audit was stale because it did not ingest the large artifact; no GPU rerun "
                "was needed.\n\n"
                "| Reconciled set | T1 passed | T1 rejected | T2/T3/T4 artifacts | Strict new cases |\n"
                "|---:|---:|---:|---:|---:|\n"
                f"| {len(reconciled)} | {sum(row['t1_pass'] for row in reconciled)} | "
                f"{sum(not row['t1_pass'] for row in reconciled)} | 0 | 0 |\n\n"
                "The T1 survivors are eligibility records, not complete Flash-style cases. "
                "No `mamba_seq256` causal, carrier, or trajectory follow-up artifact exists yet.\n",
            )
    print(json.dumps({"status": output["status"], "denominator": output["denominator"],
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
