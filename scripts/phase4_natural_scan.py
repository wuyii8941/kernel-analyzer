#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forkcert.detector import detect_clipping_fork
from forkcert.io import read_jsonl, write_jsonl
from forkcert.report import markdown_table, write_phase_report
from forkcert.schema import SCHEMA_VERSION
from forkcert.stats import mean, percentile


def make_key(row: dict) -> tuple[str, int]:
    return str(row["case_id"]), int(row["token_index"])


def token_class(text: str | None) -> str:
    value = text or ""
    if not value or value.isspace():
        return "whitespace"
    stripped = value.strip()
    if stripped.isdigit():
        return "digit"
    if stripped.isalpha():
        return "alphabetic"
    if all(not char.isalnum() and not char.isspace() for char in stripped):
        return "punctuation"
    return "mixed_or_other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 natural scan using Phase 1 logprobs and rollout old_logp/advantage.")
    parser.add_argument("--logprob-jsonl", required=True)
    parser.add_argument("--rollout-jsonl", required=True, help="JSONL keyed by case_id/token_index with old_logp and advantage/sign.")
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--delta-bound", type=float, default=None)
    parser.add_argument(
        "--bounds-json",
        default=None,
        help="Optional Phase 2 analytic certificate; regions use the deterministic logprob_bound_worst.",
    )
    parser.add_argument("--calibration-json", default=None, help="Optional Phase 3 empirical fork-rate model.")
    parser.add_argument("--attribution-json", default=None, help="Optional Phase 1.5 attribution summary for source context.")
    parser.add_argument("--allow-uncertified-bound", action="store_true", help="Debug only: allow a non-GO Phase 2 bound. Regions remain unknown.")
    parser.add_argument("--out-jsonl", default="results/phase4_certificates.jsonl")
    parser.add_argument("--report", default="reports/phase4.md")
    parser.add_argument("--fail-on-missing-rollout", action="store_true", help="Exit non-zero after writing outputs if any Phase 1 row lacks rollout old_logp/advantage.")
    parser.add_argument("--require-rollout-state", default=None, help="Require every matched rollout row to have this state label, e.g. final.")
    parser.add_argument("--require-rollout-token-id", action="store_true", help="Require matched rollout rows to carry the same token_id as Phase 1 rows.")
    args = parser.parse_args()

    delta_bound = args.delta_bound
    delta_bound_prob = None
    if args.bounds_json:
        bounds = json.loads(Path(args.bounds_json).read_text(encoding="utf-8"))
        certified = str(bounds.get("certificate_kind")) == "analytic_legal" and str(bounds.get("decision", "")).startswith("GO")
        if not certified and not args.allow_uncertified_bound:
            raise SystemExit("Phase 4 refuses an uncertified Phase 2 bound; obtain analytic_legal GO or omit --bounds-json.")
        delta_bound = float(bounds["logprob_bound_worst"]) if certified else None
        delta_bound_prob = float(bounds["logprob_bound_prob"]) if certified else None

    logprob_rows = read_jsonl(args.logprob_jsonl)
    rollout_rows = read_jsonl(args.rollout_jsonl)
    rollout = {make_key(row): row for row in rollout_rows}
    certs = []
    missing = 0
    wrong_state = 0
    missing_token_id = 0
    token_mismatch = 0
    for row in logprob_rows:
        key = make_key(row)
        state = rollout.get(key)
        if state is None:
            missing += 1
            continue
        if args.require_rollout_state is not None and state.get("state") != args.require_rollout_state:
            wrong_state += 1
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
        sign_value = int(sign) if sign is not None else (1 if float(state["advantage"]) > 0 else -1 if float(state["advantage"]) < 0 else 0)
        common_metadata = {
            "phase": "phase4_natural_scan",
            "rollout": args.rollout_jsonl,
            "logprobs": args.logprob_jsonl,
            "rollout_advantage": state.get("advantage"),
            "phase1_metadata": row.get("metadata", {}),
            "tokenization": {
                key: row.get(key)
                for key in [
                    "prompt_token_hash",
                    "response_token_hash",
                    "full_token_hash",
                    "prompt_token_count",
                    "response_token_count",
                    "full_token_count",
                ]
                if key in row
            },
            "rollout_alignment": {
                "rollout_token_id": rollout_token_id,
                "phase1_token_id": phase1_token_id,
                "token_id_match": token_id_match,
                "rollout_state": state.get("state"),
            },
            "token_metrics": {
                "entropy_ref": row.get("entropy_ref"),
                "entropy_alt": row.get("entropy_alt"),
                "entropy_delta": row.get("entropy_delta"),
                "token_class": token_class(row.get("token_text")),
            },
            "bound_metadata": {
                "primary_bound_kind": "deterministic_worst",
                "delta_bound_worst": delta_bound,
                "delta_bound_prob": delta_bound_prob,
                "probability_region_is_not_bug_proof": True,
            },
        }
        if sign_value == 0:
            certs.append(
                {
                    "case_id": str(row["case_id"]),
                    "token_index": int(row["token_index"]),
                    "token_id": row.get("token_id"),
                    "token_text": row.get("token_text"),
                    "path_ref": row.get("path_ref", "path_ref"),
                    "path_alt": row.get("path_alt", "path_alt"),
                    "logp_ref": float(row["logp_ref"]),
                    "logp_alt": float(row["logp_alt"]),
                    "old_logp": float(state["old_logp"]),
                    "advantage_sign": 0,
                    "eps": args.eps,
                    "logprob_delta": abs(float(row["logp_alt"]) - float(row["logp_ref"])),
                    "delta_self_ref": row.get("delta_self_ref"),
                    "delta_self_alt": row.get("delta_self_alt"),
                    "clip_boundary": None,
                    "clip_margin": None,
                    "clip_ref": False,
                    "clip_alt": False,
                    "delta_bound_legal": delta_bound,
                    "region": "not_applicable",
                    "fork_possible": False,
                    "actual_fork": False,
                    "grad_contribution_ref": 0.0,
                    "grad_contribution_alt": 0.0,
                    "grad_contribution_diff": 0.0,
                    "metadata": common_metadata,
                    "schema_version": SCHEMA_VERSION,
                }
            )
            continue
        cert = detect_clipping_fork(
            case_id=str(row["case_id"]),
            token_index=int(row["token_index"]),
            token_id=row.get("token_id"),
            token_text=row.get("token_text"),
            path_ref=row.get("path_ref", "path_ref"),
            path_alt=row.get("path_alt", "path_alt"),
            logp_ref=float(row["logp_ref"]),
            logp_alt=float(row["logp_alt"]),
            old_logp=float(state["old_logp"]),
            advantage=float(state["advantage"]) if sign is None else None,
            advantage_sign_value=sign_value,
            eps=args.eps,
            delta_self_ref=row.get("delta_self_ref"),
            delta_self_alt=row.get("delta_self_alt"),
            delta_bound_legal=delta_bound,
            metadata=common_metadata,
        )
        certs.append(cert.to_json_dict())
    write_jsonl(args.out_jsonl, certs)

    total = len(certs)
    applicable = [cert for cert in certs if int(cert.get("advantage_sign", 0)) != 0]
    applicable_total = len(applicable)
    region_counts: dict[str, int] = {}
    for cert in certs:
        region_counts[cert["region"]] = region_counts.get(cert["region"], 0) + 1
    deltas = [c["logprob_delta"] for c in applicable]
    margins = [c["clip_margin"] for c in applicable]
    sample_count = len({c["case_id"] for c in certs})
    actual_count = sum(1 for c in applicable if c["actual_fork"])
    summary = {
        "n_certificates": total,
        "n_applicable_decisions": applicable_total,
        "n_zero_advantage_not_applicable": total - applicable_total,
        "n_samples": sample_count,
        "missing_rollout_rows": missing,
        "wrong_rollout_state_rows": wrong_state,
        "missing_rollout_token_id_rows": missing_token_id,
        "token_id_mismatch_rows": token_mismatch,
        "mean_logprob_delta": mean(deltas) if applicable_total else 0,
        "p95_logprob_delta": percentile(deltas, 95) if applicable_total else 0,
        "p99_logprob_delta": percentile(deltas, 99) if applicable_total else 0,
        "mean_clip_margin": mean(margins) if applicable_total else 0,
        "p1_clip_margin": percentile(margins, 1) if applicable_total else 0,
        "p5_clip_margin": percentile(margins, 5) if applicable_total else 0,
        "actual_fork_rate": actual_count / applicable_total if applicable_total else 0,
        "fork_possible_rate": sum(1 for c in applicable if c["fork_possible"]) / applicable_total if applicable_total else 0,
        "forks_per_1k_tokens": 1000.0 * actual_count / applicable_total if applicable_total else 0,
        "forks_per_1k_samples": 1000.0 * actual_count / sample_count if sample_count else 0,
        **{
            f"region_rate_{k}": v / applicable_total if applicable_total and k != "not_applicable" else (0.0 if k != "not_applicable" else v / total if total else 0.0)
            for k, v in sorted(region_counts.items())
        },
    }
    if args.calibration_json:
        calibration = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))
        predicted = float(calibration.get("predicted_fork_rate_late", 0.0))
        summary["predicted_fork_rate"] = predicted
        summary["observed_minus_predicted"] = summary["actual_fork_rate"] - predicted
    actual_forks = [c for c in certs if c["actual_fork"]]
    fragile = [c for c in applicable if c.get("region") == "fragile"]
    fragile_by_position: dict[str, int] = {}
    fragile_by_sign: dict[str, int] = {}
    fragile_by_class: dict[str, int] = {}
    fragile_entropies = []
    for cert in fragile:
        position_start = (int(cert["token_index"]) // 32) * 32
        position_key = f"{position_start}-{position_start + 31}"
        fragile_by_position[position_key] = fragile_by_position.get(position_key, 0) + 1
        sign_key = "positive" if int(cert["advantage_sign"]) > 0 else "negative"
        fragile_by_sign[sign_key] = fragile_by_sign.get(sign_key, 0) + 1
        metrics = (cert.get("metadata") or {}).get("token_metrics") or {}
        class_key = str(metrics.get("token_class", "unknown"))
        fragile_by_class[class_key] = fragile_by_class.get(class_key, 0) + 1
        if metrics.get("entropy_ref") is not None:
            fragile_entropies.append(float(metrics["entropy_ref"]))
    fragile_summary = {
        "fragile_tokens": len(fragile),
        "entropy_ref_p50": percentile(fragile_entropies, 50) if fragile_entropies else None,
        "entropy_ref_p95": percentile(fragile_entropies, 95) if fragile_entropies else None,
        "entropy_ref_mean": mean(fragile_entropies) if fragile_entropies else None,
    }
    fragile_breakdown = [
        {"dimension": "token_position", "value": key, "count": value}
        for key, value in sorted(fragile_by_position.items())
    ] + [
        {"dimension": "advantage_sign", "value": key, "count": value}
        for key, value in sorted(fragile_by_sign.items())
    ] + [
        {"dimension": "token_class", "value": key, "count": value}
        for key, value in sorted(fragile_by_class.items())
    ]
    attribution_context = []
    if args.attribution_json:
        attribution_payload = json.loads(Path(args.attribution_json).read_text(encoding="utf-8"))
        attribution_context = sorted(
            attribution_payload.get("rows", []),
            key=lambda item: abs(float(item.get("final_logprob_delta", 0.0))),
            reverse=True,
        )
    min_case = min(actual_forks, key=lambda c: c["logprob_delta"]) if actual_forks else None
    min_case_text = json.dumps(min_case, indent=2, sort_keys=True) if min_case else "_No actual fork found._"
    write_phase_report(
        args.report,
        title="Phase 4 Natural Scan",
        confound_checklist={
            "fixed_response_tokens": True,
            "real_old_logp_present": missing == 0,
            "rollout_state_matches": wrong_state == 0,
            "rollout_token_id_matches": token_mismatch == 0 and (not args.require_rollout_token_id or missing_token_id == 0),
            "advantage_sign_present_or_zero_marked_not_applicable": True,
            "same_token_comparison": True,
            "actual_forks_need_manual_confounds": "required for every actual_fork before claim",
        },
        delta_self_summary="Uses delta_self fields from Phase 1; inspect Phase 1 report for the hard gate.",
        summary="Natural scan generated v2 certificates with clipping branch decisions.",
        sections={
            "Rates": markdown_table([summary], list(summary.keys())),
            "Minimum Actual Fork Case": f"```json\n{min_case_text}\n```",
            "Fragile Set Entropy": markdown_table([fragile_summary], list(fragile_summary.keys())),
            "Fragile Set Breakdown": markdown_table(
                fragile_breakdown,
                ["dimension", "value", "count"],
            ),
            "Global Attribution Context": markdown_table(
                attribution_context,
                list(attribution_context[0].keys()) if attribution_context else [],
            ),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.fail_on_missing_rollout and (missing or wrong_state or token_mismatch or (args.require_rollout_token_id and missing_token_id)):
        print(
            "Phase 4 rollout coverage gate failed: "
            f"missing={missing}, wrong_state={wrong_state}, missing_token_id={missing_token_id}, token_mismatch={token_mismatch}.",
            file=sys.stderr,
        )
        raise SystemExit(24)


if __name__ == "__main__":
    main()
