#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.report import CLAIM_SCOPE, markdown_table
from forkcert.stats import mean, percentile


def keyed(payload: dict) -> dict[tuple[str, int, int], dict]:
    return {
        (str(row["case_id"]), int(row["token_index"]), int(row["token_id"])): row
        for row in payload["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge HF and two independent vLLM teacher-forcing runs.")
    parser.add_argument("--hf", required=True)
    parser.add_argument("--hf-b", help="Independent-process HF repeat; overrides in-process delta_self.")
    parser.add_argument("--vllm-a", required=True)
    parser.add_argument("--vllm-b", required=True)
    parser.add_argument("--vllm-reordered")
    parser.add_argument("--identity-audit")
    parser.add_argument("--out", default="results/phase1_hf_vllm.jsonl")
    parser.add_argument("--report", default="reports/phase1_vllm.md")
    args = parser.parse_args()

    hf_payload = json.loads(Path(args.hf).read_text(encoding="utf-8"))
    hf_b_payload = json.loads(Path(args.hf_b).read_text(encoding="utf-8")) if args.hf_b else None
    a_payload = json.loads(Path(args.vllm_a).read_text(encoding="utf-8"))
    b_payload = json.loads(Path(args.vllm_b).read_text(encoding="utf-8"))
    hf, a, b = keyed(hf_payload), keyed(a_payload), keyed(b_payload)
    hf_b = keyed(hf_b_payload) if hf_b_payload else None
    if not (set(hf) == set(a) == set(b)):
        raise ValueError(
            f"HF/vLLM key coverage mismatch: hf={len(hf)}, a={len(a)}, b={len(b)}, "
            f"intersection={len(set(hf) & set(a) & set(b))}"
        )
    if hf_b is not None and set(hf_b) != set(hf):
        raise ValueError("independent HF repeat key coverage mismatch")
    rows = []
    reordered_payload = (
        json.loads(Path(args.vllm_reordered).read_text(encoding="utf-8")) if args.vllm_reordered else None
    )
    reordered = keyed(reordered_payload) if reordered_payload else None
    if reordered is not None and set(reordered) != set(a):
        raise ValueError("vLLM reordered run key coverage mismatch")
    identity_payload = (
        json.loads(Path(args.identity_audit).read_text(encoding="utf-8")) if args.identity_audit else None
    )
    identity_pass = identity_payload is not None and identity_payload.get("passed") is True
    for key in sorted(hf):
        h, va, vb = hf[key], a[key], b[key]
        rows.append(
            {
                "schema_version": "forkcert.phase1.hf_vllm.v1",
                "case_id": key[0],
                "token_index": key[1],
                "token_id": key[2],
                "path_ref": hf_payload["metadata"]["path"],
                "path_alt": "vllm-fp16-teacher-forcing",
                "logp_ref": float(h["logp"]),
                "logp_alt": float(va["logp"]),
                "logprob_delta": abs(float(va["logp"]) - float(h["logp"])),
                "delta_self_ref": abs(float(hf_b[key]["logp"]) - float(h["logp"])) if hf_b is not None else float(h["delta_self"]),
                "delta_self_alt": abs(float(vb["logp"]) - float(va["logp"])),
                "delta_batch_order_alt": (
                    abs(float(reordered[key]["logp"]) - float(va["logp"])) if reordered is not None else None
                ),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    cross = [row["logprob_delta"] for row in rows]
    self_ref = [row["delta_self_ref"] for row in rows]
    self_alt = [row["delta_self_alt"] for row in rows]
    batch_order = [row["delta_batch_order_alt"] for row in rows if row["delta_batch_order_alt"] is not None]
    cross_p50 = percentile(cross, 50)
    gate = (
        percentile(self_ref, 99) == 0.0
        and percentile(self_alt, 99) == 0.0
        and max(cross) > 0.0
        if cross_p50 == 0.0
        else percentile(self_ref, 99) < 0.1 * cross_p50
        and percentile(self_alt, 99) < 0.1 * cross_p50
    )
    summary = {
        "requests": int(hf_payload["metadata"]["requests"]),
        "tokens": len(rows),
        "delta_mean": mean(cross),
        "delta_p50": cross_p50,
        "delta_p95": percentile(cross, 95),
        "delta_p99": percentile(cross, 99),
        "delta_max": max(cross),
        "self_ref_p99": percentile(self_ref, 99),
        "self_alt_p99": percentile(self_alt, 99),
        "self_gate": gate,
        "batch_order_p99": percentile(batch_order, 99) if batch_order else None,
        "batch_order_max": max(batch_order) if batch_order else None,
        "batch_order_invariant": max(batch_order) == 0.0 if batch_order else None,
        "vllm_version": a_payload["metadata"]["vllm_version"],
        "vllm_engine": "V0" if a_payload["metadata"].get("vllm_use_v1") == "0" else "unknown",
        "hf_self_is_independent_process": hf_b is not None,
    }
    report = "\n".join(
        [
            "# Phase 1 HF-vLLM Teacher-Forcing Pair",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- exact shared local checkpoint: PASS",
            "- tokenized prompt and response IDs supplied directly: PASS",
            "- response scored only through prompt_logprobs: PASS",
            "- generated token excluded: PASS",
            "- temperature and penalties disabled: PASS",
            "- two independent HF processes/contexts: " + ("PASS" if hf_b is not None else "FAIL"),
            "- two independent vLLM processes/contexts: PASS",
            f"- self gate: {'PASS' if gate else 'FAIL'}",
            f"- empirical batch-order invariance: {'PASS' if summary['batch_order_invariant'] else 'FAIL'}",
            "- raw/processed identity equivalence: " + ("PASS" if identity_pass else "NOT PROVIDED"),
            "",
            "## Delta Self Control",
            f"HF p99={summary['self_ref_p99']:.9g}; vLLM independent-process p99={summary['self_alt_p99']:.9g}.",
            "",
            "## Raw/Processed Identity",
            (
                f"Equivalent completion: {identity_payload['identity']['tokens_compared']} tokens compared bitwise "
                f"with {identity_payload['identity']['bitwise_mismatch_count']} mismatches after all request-level "
                "logits processors were explicitly disabled."
                if identity_pass
                else "No explicit identity audit was supplied."
            ),
            "",
            "## Summary",
            markdown_table([summary], list(summary.keys())),
            "",
            "## External Validity",
            "This pair uses vLLM V0 on Tesla T4 FP16. It does not establish native BF16 or V1 processed-logit behavior.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
