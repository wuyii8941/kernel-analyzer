#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.report import markdown_table, write_phase_report
from forkcert.stats import percentile


def phase1_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase1_present": False}
    rows = read_jsonl(path)
    if not rows:
        return {"phase1_present": True, "phase1_nonempty": False}
    deltas = [float(row["logprob_delta"]) for row in rows]
    self_ref = [float(row.get("delta_self_ref", 0.0)) for row in rows]
    self_alt = [float(row.get("delta_self_alt", 0.0)) for row in rows]
    p50 = percentile(deltas, 50)
    sample_count = len({str(row["case_id"]) for row in rows})
    ref_p99 = percentile(self_ref, 99)
    alt_p99 = percentile(self_alt, 99)
    metadata = rows[0].get("metadata") or {}
    fingerprint_ref = metadata.get("model_artifact_fingerprint_ref") or {}
    fingerprint_alt = metadata.get("model_artifact_fingerprint_alt") or {}
    weights_verified = (
        fingerprint_ref.get("verified_local_files") is True
        and fingerprint_alt.get("verified_local_files") is True
        and bool(fingerprint_ref.get("aggregate_sha256"))
        and fingerprint_ref.get("aggregate_sha256") == fingerprint_alt.get("aggregate_sha256")
    )
    return {
        "phase1_present": True,
        "phase1_nonempty": True,
        "phase1_tokens": len(rows),
        "phase1_samples": sample_count,
        "phase1_scale_gate": 100 <= sample_count <= 500 and len(rows) >= 50_000,
        "phase1_weights_verified": weights_verified,
        "delta_p50": p50,
        "delta_p99": percentile(deltas, 99),
        "delta_self_ref_p99": ref_p99,
        "delta_self_alt_p99": alt_p99,
        "delta_self_ref_gate": ref_p99 < 0.1 * p50 if p50 > 0 else False,
        "delta_self_alt_gate": alt_p99 < 0.1 * p50 if p50 > 0 else False,
    }


def phase1_manifest_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase1_manifest_present": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = data.get("pairs") or []
    return {
        "phase1_manifest_present": True,
        "phase1_manifest_pair_count": len(pairs),
        "phase1_required_pairs_gate": data.get("required_pairs_gate") is True,
        "phase1_optional_vllm_gate": data.get("optional_vllm_gate") is True,
        "phase1_debug_excluded_from_claim": data.get("debug_results_excluded_from_claim") is True,
    }


def phase0_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase0_present": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data.get("overall", {})
    late = data.get("late_minibatches", overall)
    near = float(late.get("P(margin<0.01)", 0.0))
    real_training = bool((data.get("provenance") or {}).get("canonical_real_training", False))
    determinism = data.get("determinism") or {}
    deterministic_metadata = bool(determinism.get("metadata_present")) and bool(determinism.get("warn_messages_recorded"))
    return {
        "phase0_present": True,
        "phase0_tokens": overall.get("n", 0),
        "phase0_late_tokens": late.get("n", 0),
        "phase0_p_margin_lt_1e_2_overall": overall.get("P(margin<0.01)", 0.0),
        "phase0_p_margin_lt_1e_2_late": near,
        "phase0_real_training": real_training,
        "phase0_deterministic_metadata": deterministic_metadata,
        "phase0_go": near >= 0.001 and real_training and deterministic_metadata,
    }


def phase4_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase4_present": False}
    rows = read_jsonl(path)
    if not rows:
        return {"phase4_present": True, "phase4_nonempty": False}
    applicable = [row for row in rows if int(row.get("advantage_sign", 0)) != 0]
    actual = sum(1 for row in applicable if row.get("actual_fork"))
    possible = sum(1 for row in applicable if row.get("fork_possible"))
    region_counts: dict[str, int] = {}
    for row in rows:
        region_counts[row.get("region", "missing")] = region_counts.get(row.get("region", "missing"), 0) + 1
    out = {
        "phase4_present": True,
        "phase4_nonempty": True,
        "phase4_certificates": len(rows),
        "phase4_applicable_decisions": len(applicable),
        "phase4_not_applicable": len(rows) - len(applicable),
        "phase4_missing_rollout_rows": "see Phase 4 report",
        "phase4_actual_fork_rate": actual / len(applicable) if applicable else 0.0,
        "phase4_fork_possible_rate": possible / len(applicable) if applicable else 0.0,
    }
    out.update({f"phase4_region_{key}": value for key, value in sorted(region_counts.items())})
    return out


def phase2_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase2_present": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    decision = str(data.get("decision", "UNKNOWN"))
    return {
        "phase2_present": True,
        "phase2_decision": decision,
        "phase2_certificate_kind": data.get("certificate_kind", "unverified"),
        "phase2_bound_prob": data.get("logprob_bound_prob"),
        "phase2_bound_worst": data.get("logprob_bound_worst"),
        "phase2_empirical_delta_p99": data.get("empirical_delta_p99"),
        "phase2_empirical_delta_max": data.get("empirical_delta_max"),
        "phase2_classifier_usable": decision.startswith("GO") and data.get("certificate_kind") == "analytic_legal",
    }


def phase3_gate(path: Path, calibration_path: Path | None = None) -> dict:
    if not path.exists():
        return {"phase3_present": False}
    rows = read_jsonl(path)
    if not rows:
        return {"phase3_present": True, "phase3_nonempty": False}
    actual = sum(1 for row in rows if row.get("actual_fork"))
    possible = sum(1 for row in rows if row.get("fork_possible"))
    calibration = None
    if calibration_path is not None and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_ok = bool(
        calibration
        and calibration.get("model_kind") == "empirical_independent_margin_delta_convolution"
        and int(calibration.get("margin_count", 0)) > 0
        and int(calibration.get("delta_count", 0)) > 0
    )
    return {
        "phase3_present": True,
        "phase3_nonempty": True,
        "phase3_certificates": len(rows),
        "phase3_fork_possible": possible,
        "phase3_actual_forks": actual,
        "phase3_actual_fork_rate": actual / len(rows),
        "phase3_detector_gate": actual > 0,
        "phase3_calibration_present": calibration is not None,
        "phase3_calibration_gate": calibration_ok,
        "phase3_predicted_fork_rate_late": calibration.get("predicted_fork_rate_late") if calibration else None,
    }


def phase15_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase15_present": False}
    rows = read_jsonl(path)
    levels = sorted({str(row.get("level")) for row in rows})
    required = {"L1", "L2", "L3", "L4", "L5", "L6"}
    missing = sorted(required - set(levels))
    return {
        "phase15_present": True,
        "phase15_nonempty": bool(rows),
        "phase15_rows": len(rows),
        "phase15_levels": ",".join(levels),
        "phase15_required_levels_present": not missing,
        "phase15_missing_levels": ",".join(missing),
    }


def phase5_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase5_present": False}
    rows = read_jsonl(path)
    if not rows:
        return {"phase5_present": True, "phase5_nonempty": False}
    expected_bug = [row for row in rows if (row.get("metadata") or {}).get("phase5_expected_bug") is True]
    expected_valid = [row for row in rows if (row.get("metadata") or {}).get("phase5_expected_bug") is False]
    unlabeled = len(rows) - len(expected_bug) - len(expected_valid)
    true_positive = sum(1 for row in expected_bug if row.get("region") == "bug")
    false_negative = len(expected_bug) - true_positive
    false_positive = sum(1 for row in expected_valid if row.get("region") == "bug")
    true_negative = len(expected_valid) - false_positive
    non_kernel = sum(
        1
        for row in expected_bug
        if ((row.get("metadata") or {}).get("bug") or {}).get("injection_kind") != "kernel_execution"
    )
    token_bad = 0
    token_missing = 0
    for row in rows:
        alignment = ((row.get("metadata") or {}).get("rollout_alignment") or {})
        if alignment.get("token_id_match") is True:
            continue
        if "token_id_match" not in alignment or alignment.get("token_id_match") is None:
            token_missing += 1
        else:
            token_bad += 1
    return {
        "phase5_present": True,
        "phase5_nonempty": True,
        "phase5_certificates": len(rows),
        "phase5_true_positive": true_positive,
        "phase5_false_negative": false_negative,
        "phase5_false_positive": false_positive,
        "phase5_true_negative": true_negative,
        "phase5_unlabeled": unlabeled,
        "phase5_bug_recall": true_positive / len(expected_bug) if expected_bug else 0.0,
        "phase5_non_kernel_injections": non_kernel,
        "phase5_kernel_injection_gate": bool(expected_bug) and non_kernel == 0,
        "phase5_confusion_gate": bool(expected_bug) and bool(expected_valid) and unlabeled == 0 and false_negative == 0 and false_positive == 0,
        "phase5_token_alignment_gate": token_bad == 0 and token_missing == 0,
        "phase5_token_bad": token_bad,
        "phase5_token_missing": token_missing,
    }


def phase6_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase6_present": False}
    rows = read_jsonl(path)
    if not rows:
        return {"phase6_present": True, "phase6_nonempty": False}
    actual = [row for row in rows if row.get("actual_fork")]
    missing_grad = sum(1 for row in actual if row.get("grad_contribution_diff") is None)
    zero_grad = sum(1 for row in actual if row.get("grad_contribution_diff") is not None and float(row.get("grad_contribution_diff", 0.0)) <= 0.0)
    autograd = sum(1 for row in actual if row.get("grad_contribution_mode") == "hf_autograd")
    proxy = sum(1 for row in actual if row.get("grad_contribution_mode") == "branch_proxy")
    unmarked = len(actual) - autograd - proxy
    return {
        "phase6_present": True,
        "phase6_nonempty": True,
        "phase6_certificates": len(rows),
        "phase6_actual_forks": len(actual),
        "phase6_missing_grad_diff": missing_grad,
        "phase6_zero_grad_diff": zero_grad,
        "phase6_grad_gate": missing_grad == 0 and zero_grad == 0,
        "phase6_autograd_actual_forks": autograd,
        "phase6_proxy_actual_forks": proxy,
        "phase6_unmarked_actual_forks": unmarked,
        "phase6_autograd_gate": autograd == len(actual),
    }


def phase6_twin_gate(path: Path) -> dict:
    if not path.exists():
        return {"phase6_twin_present": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    status = str(data.get("status", "missing"))
    return {
        "phase6_twin_present": True,
        "phase6_twin_status": status,
        "phase6_twin_optimizer_steps": int(data.get("optimizer_steps", 0)),
        "phase6_twin_weight_measurements": int(data.get("weight_measurements", 0)),
        "phase6_twin_fork_events": int(data.get("total_fork_events", 0)),
        "phase6_twin_backend_only": data.get("backend_only_difference") is True,
        "phase6_twin_exact_divergence": data.get("exact_weight_divergence") is True,
        "phase6_twin_gate": (
            status == "completed"
            and data.get("backend_only_difference") is True
            and data.get("exact_weight_divergence") is True
            and data.get("weight_scope") == "full_model"
            and int(data.get("optimizer_steps", 0)) >= 100
            and int(data.get("weight_measurements", 0)) >= 20
            and int(data.get("total_fork_events", 0)) > 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ForkCert phase outputs and gate decisions.")
    parser.add_argument("--phase0-summary", default="results/phase0_margin_summary.json")
    parser.add_argument("--phase1-logprobs", default="results/phase1_logprobs.jsonl")
    parser.add_argument("--phase1-manifest", default="results/phase1_pair_manifest.json")
    parser.add_argument("--phase15-measurements", default="results/phase15_measurements.jsonl")
    parser.add_argument("--phase2-bounds", default="results/phase2_bounds.json")
    parser.add_argument("--phase3-certificates", default="results/phase3_controlled_certificates.jsonl")
    parser.add_argument("--phase3-calibration", default="results/phase3_calibration.json")
    parser.add_argument("--phase4-certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--phase5-certificates", default="results/phase5_bug_certificates.jsonl")
    parser.add_argument("--phase6-certificates", default="results/phase6_grad_certificates.jsonl")
    parser.add_argument("--phase6-twin-summary", default="results/phase6_twin_summary.json")
    parser.add_argument("--report", default="reports/audit.md")
    parser.add_argument("--fail-on-review", action="store_true", help="Exit non-zero if the final audit decision is REVIEW/DOWNGRADE.")
    args = parser.parse_args()

    rows = [
        phase0_gate(Path(args.phase0_summary)),
        phase1_gate(Path(args.phase1_logprobs)),
        phase1_manifest_gate(Path(args.phase1_manifest)),
        phase15_gate(Path(args.phase15_measurements)),
        phase2_gate(Path(args.phase2_bounds)),
        phase3_gate(Path(args.phase3_certificates), Path(args.phase3_calibration)),
        phase4_gate(Path(args.phase4_certificates)),
        phase5_gate(Path(args.phase5_certificates)),
        phase6_gate(Path(args.phase6_certificates)),
        phase6_twin_gate(Path(args.phase6_twin_summary)),
    ]
    flat = {}
    for row in rows:
        flat.update(row)
    phase4_coverage_ok = True
    if flat.get("phase1_nonempty") and flat.get("phase4_nonempty"):
        phase4_coverage_ok = flat.get("phase1_tokens") == flat.get("phase4_certificates")
        flat["phase4_coverage_gate"] = phase4_coverage_ok
    phase6_coverage_ok = True
    if flat.get("phase4_nonempty") and flat.get("phase6_nonempty"):
        phase6_coverage_ok = flat.get("phase4_certificates") == flat.get("phase6_certificates")
        flat["phase6_coverage_gate"] = phase6_coverage_ok
    if flat.get("phase1_present") and (
        not flat.get("phase1_nonempty", False)
        or not flat.get("phase1_scale_gate", False)
        or not flat.get("delta_self_ref_gate", False)
        or not flat.get("delta_self_alt_gate", False)
        or not flat.get("phase1_weights_verified", False)
    ):
        decision = "REVIEW: Phase 1 requires 100-500 pairs, >=50k tokens, and both self-consistency gates."
    elif flat.get("phase1_present") and (
        not flat.get("phase1_manifest_present", False)
        or not flat.get("phase1_required_pairs_gate", False)
        or not flat.get("phase1_optional_vllm_gate", False)
        or not flat.get("phase1_debug_excluded_from_claim", False)
    ):
        decision = "REVIEW: Phase 1 path-pair manifest must include passing debug, eager-compile, and SDPA-Flash pairs; vLLM must be measured when installed."
    elif flat.get("phase15_present") and not flat.get("phase15_required_levels_present"):
        decision = "REVIEW: Phase 1.5 attribution measurements are incomplete; require all six one-variable ladder levels before Phase 2 work."
    elif not flat.get("phase15_present") and flat.get("phase2_present"):
        decision = "REVIEW: Phase 2 bounds exist but Phase 1.5 attribution measurements are missing; source coverage is unaudited."
    elif flat.get("phase3_present") and not flat.get("phase3_nonempty", True):
        decision = "REVIEW: Phase 3 controlled certificates are empty; detector sanity check is missing."
    elif flat.get("phase3_present") and not flat.get("phase3_detector_gate", True):
        decision = "REVIEW: Phase 3 controlled construction did not produce an actual fork; detector calibration failed."
    elif flat.get("phase3_present") and not flat.get("phase3_calibration_gate", False):
        decision = "REVIEW: Phase 3 empirical margin-delta calibration model is missing or empty."
    elif not flat.get("phase3_present") and flat.get("phase4_present"):
        decision = "REVIEW: Phase 4 certificates exist but Phase 3 detector sanity check is missing."
    elif flat.get("phase5_present") and not flat.get("phase5_nonempty", True):
        decision = "REVIEW: Phase 5 bug-injection certificates are empty; classifier validation is missing."
    elif flat.get("phase5_present") and (
        not flat.get("phase5_confusion_gate", False)
        or not flat.get("phase5_token_alignment_gate", True)
        or not flat.get("phase5_kernel_injection_gate", False)
    ):
        decision = "REVIEW: Phase 5 requires executed kernel-level bug injections; post-hoc logprob shifts are smoke tests only."
    elif flat.get("phase4_actual_fork_rate", 0) > 0 and not flat.get("phase6_present"):
        decision = "REVIEW: Phase 4 found actual forks but Phase 6 gradient contribution evidence is missing."
    elif flat.get("phase4_actual_fork_rate", 0) > 0 and (
        not flat.get("phase6_nonempty", False)
        or not phase6_coverage_ok
        or not flat.get("phase6_grad_gate", False)
        or not flat.get("phase6_autograd_gate", False)
    ):
        decision = "REVIEW: Phase 6 requires full autograd gradient evidence for every actual fork; proxy annotations are not claim-ready."
    elif flat.get("phase4_actual_fork_rate", 0) > 0 and (
        not flat.get("phase6_twin_present", False) or not flat.get("phase6_twin_gate", False)
    ):
        decision = "REVIEW: Phase 4 found natural forks, but the backend-only twin-training coupling experiment is missing or incomplete."
    elif not phase4_coverage_ok:
        decision = "REVIEW: Phase 4 certificates do not cover all Phase 1 logprob rows; rerun with matched rollout samples."
    elif flat.get("phase2_present") and not flat.get("phase2_classifier_usable"):
        decision = "REVIEW: Phase 2 bound is not usable as a classifier; do not make fragile/bug claim without refining B."
    elif flat.get("phase4_actual_fork_rate", 0) > 0:
        decision = "GO: natural or semi-natural actual forks were found; inspect confound checklist before claim."
    elif flat.get("phase0_go") and flat.get("phase1_present"):
        decision = "CONTINUE: near-boundary mass exists; scan more path pairs or increase samples."
    else:
        decision = "DOWNGRADE/REVIEW: no fork evidence yet; consider coverage certification or other decision types."
    write_phase_report(
        args.report,
        title="ForkCert Output Audit",
        confound_checklist={
            "phase0_summary_present": flat.get("phase0_present", False),
            "phase1_logprobs_present": flat.get("phase1_present", False),
            "phase1_pair_manifest_present": flat.get("phase1_manifest_present", False),
            "phase15_measurements_present": flat.get("phase15_present", False),
            "phase2_bounds_present": flat.get("phase2_present", False),
            "phase3_controlled_certificates_present": flat.get("phase3_present", False),
            "phase4_certificates_present": flat.get("phase4_present", False),
            "phase5_bug_certificates_present": flat.get("phase5_present", False),
            "phase6_grad_certificates_present": flat.get("phase6_present", False),
            "phase6_twin_summary_present": flat.get("phase6_twin_present", False),
        },
        delta_self_summary=(
            f"ref gate={flat.get('delta_self_ref_gate')}, alt gate={flat.get('delta_self_alt_gate')}, "
            f"delta p50={flat.get('delta_p50')}"
        ),
        summary=decision,
        sections={"Audit": markdown_table([flat], list(flat.keys()))},
    )
    print(json.dumps({"decision": decision, **flat}, indent=2, sort_keys=True))
    if args.fail_on_review and not (decision.startswith("GO") or decision.startswith("CONTINUE")):
        print(f"ForkCert audit gate failed: {decision}", file=sys.stderr)
        raise SystemExit(23)


if __name__ == "__main__":
    main()
