#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forkcert.detector import detect_clipping_fork
from forkcert.inject import DEFAULT_BUGS, BugInjection, inject_logprob_bug
from forkcert.io import read_jsonl, write_jsonl
from forkcert.report import markdown_table, write_phase_report


def key(row: dict) -> tuple[str, int]:
    return str(row["case_id"]), int(row["token_index"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 bug injection validation for three-way classifier.")
    parser.add_argument("--logprob-jsonl", required=True)
    parser.add_argument("--rollout-jsonl", required=True)
    parser.add_argument("--delta-bound", type=float, default=None)
    parser.add_argument(
        "--bounds-json",
        default=None,
        help="Optional Phase 2 analytic certificate; bug classification uses logprob_bound_worst.",
    )
    parser.add_argument("--scale-shift-to-bound", action="store_true")
    parser.add_argument("--shift-multiplier", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--out-jsonl", default="results/phase5_bug_certificates.jsonl")
    parser.add_argument("--report", default="reports/phase5.md")
    parser.add_argument("--require-rollout-token-id", action="store_true", help="Require rollout token_id to match Phase 1 token_id.")
    parser.add_argument("--fail-on-token-mismatch", action="store_true", help="Exit non-zero after writing report if rollout token alignment is missing or mismatched.")
    args = parser.parse_args()

    delta_bound = args.delta_bound
    if args.bounds_json:
        bounds = json.loads(Path(args.bounds_json).read_text(encoding="utf-8"))
        if bounds.get("certificate_kind") != "analytic_legal" or not str(bounds.get("decision", "")).startswith("GO"):
            raise SystemExit("Phase 5 refuses an uncertified Phase 2 bound; bug classification requires analytic_legal GO.")
        delta_bound = float(bounds["logprob_bound_worst"])
    if delta_bound is None:
        raise SystemExit("Phase 5 requires --delta-bound or --bounds-json.")

    base_rows = read_jsonl(args.logprob_jsonl)
    rollout = {key(row): row for row in read_jsonl(args.rollout_jsonl)}
    certs = []
    missing_rollout = 0
    missing_token_id = 0
    token_mismatch = 0
    for bug in DEFAULT_BUGS:
        if args.scale_shift_to_bound:
            sign = 1.0 if bug.logprob_shift >= 0 else -1.0
            shift = sign * max(abs(bug.logprob_shift), args.shift_multiplier * delta_bound)
            bug = BugInjection(bug.name, bug.description, shift, bug.token_selector)
        injected = inject_logprob_bug(base_rows, bug)
        for row in injected:
            state = rollout.get(key(row))
            if state is None:
                missing_rollout += 1
                continue
            rollout_token_id = state.get("token_id")
            phase1_token_id = row.get("token_id")
            token_id_match = None
            if rollout_token_id is None or phase1_token_id is None:
                missing_token_id += 1
                if args.require_rollout_token_id:
                    continue
            else:
                token_id_match = int(rollout_token_id) == int(phase1_token_id)
                if not token_id_match:
                    token_mismatch += 1
                    continue
            sign = state.get("advantage_sign")
            cert = detect_clipping_fork(
                case_id=f"{bug.name}:{row['case_id']}",
                token_index=int(row["token_index"]),
                token_id=row.get("token_id"),
                token_text=row.get("token_text"),
                path_ref=row.get("path_ref", "path_ref"),
                path_alt=f"{row.get('path_alt', 'path_alt')}+{bug.name}",
                logp_ref=float(row["logp_ref"]),
                logp_alt=float(row["logp_alt"]),
                old_logp=float(state["old_logp"]),
                advantage=float(state["advantage"]) if sign is None else None,
                advantage_sign_value=int(sign) if sign is not None else None,
                eps=args.eps,
                delta_self_ref=row.get("delta_self_ref"),
                delta_self_alt=row.get("delta_self_alt"),
                delta_bound_legal=delta_bound,
                metadata={
                    "phase": "phase5_bug_injection",
                    "bug": bug.to_json_dict(),
                    "rollout_alignment": {
                        "rollout_token_id": rollout_token_id,
                        "phase1_token_id": phase1_token_id,
                        "token_id_match": token_id_match,
                    },
                },
            )
            certs.append(cert.to_json_dict())
    write_jsonl(args.out_jsonl, certs)

    total = len(certs)
    bug_count = sum(1 for cert in certs if cert["region"] == "bug")
    summary = {
        "n_certificates": total,
        "expected_bug": total,
        "classified_bug": bug_count,
        "bug_recall": bug_count / total if total else 0,
        "false_non_bug": total - bug_count,
        "missing_rollout_rows": missing_rollout,
        "missing_rollout_token_id_rows": missing_token_id,
        "token_id_mismatch_rows": token_mismatch,
    }
    write_phase_report(
        args.report,
        title="Phase 5 Bug Injection",
        confound_checklist={
            "synthetic_injection_only": True,
            "legal_delta_bound_supplied": True,
            "rollout_token_id_aligned": token_mismatch == 0 and (not args.require_rollout_token_id or missing_token_id == 0),
            "kernel_level_bug_injection": False,
            "posthoc_shift_smoke_only": True,
        },
        delta_self_summary="Uses Phase 1 delta_self fields if present; bug injection intentionally exceeds legal bound.",
        summary="Post-hoc logprob shifts validate classifier wiring only; they are not kernel-level bug evidence.",
        sections={"Confusion Matrix": markdown_table([summary], list(summary.keys()))},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_token_mismatch and (token_mismatch or (args.require_rollout_token_id and missing_token_id)):
        print(
            f"Phase 5 rollout token gate failed: missing_token_id={missing_token_id}, token_mismatch={token_mismatch}.",
            file=sys.stderr,
        )
        raise SystemExit(26)


if __name__ == "__main__":
    main()
