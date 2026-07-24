#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.stats import percentile


def check_phase0(path: Path) -> tuple[bool, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    late = data.get("late_minibatches", data.get("overall", {}))
    rate = float(late.get("P(margin<0.01)", 0.0))
    real_training = bool((data.get("provenance") or {}).get("canonical_real_training", False))
    determinism = data.get("determinism") or {}
    deterministic_metadata = (
        bool(determinism.get("metadata_present"))
        and bool(determinism.get("warn_messages_recorded"))
        and bool(determinism.get("settings_verified"))
    )
    ok = rate >= 0.001 and real_training and deterministic_metadata
    return ok, (
        f"Phase 0 late-policy-iteration P(margin<1e-2)={rate:.6g}, "
        f"canonical_real_training={real_training}, deterministic_metadata={deterministic_metadata}; "
        "require >=0.001, TRL GRPO provenance, and recorded deterministic warnings."
    )


def check_phase1(path: Path) -> tuple[bool, str]:
    rows = read_jsonl(path)
    if not rows:
        return False, "Phase 1 logprob file is empty."
    deltas = [float(row["logprob_delta"]) for row in rows]
    sample_count = len({str(row["case_id"]) for row in rows})
    self_ref = [float(row.get("delta_self_ref", 0.0)) for row in rows]
    self_alt = [float(row.get("delta_self_alt", 0.0)) for row in rows]
    cross_p50 = percentile(deltas, 50)
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
    scale_ok = 100 <= sample_count <= 500 and len(rows) >= 50_000
    ok = cross_p50 > 0 and ref_p99 < 0.1 * cross_p50 and alt_p99 < 0.1 * cross_p50 and scale_ok and weights_verified
    return (
        ok,
        "Phase 1 self gate: "
        f"samples={sample_count}, tokens={len(rows)}, cross_p50={cross_p50:.6g}, "
        f"ref_self_p99={ref_p99:.6g}, alt_self_p99={alt_p99:.6g}, weights_verified={weights_verified}; "
        "require 100-500 samples, >=50,000 tokens, identical local checkpoint hashes, and self p99 < 0.1 * cross p50.",
    )


def check_phase2(path: Path) -> tuple[bool, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    decision = str(data.get("decision", "UNKNOWN"))
    return decision.startswith("GO"), f"Phase 2 decision={decision}"


def check_phase3(path: Path, calibration_path: Path | None = None) -> tuple[bool, str]:
    certs = read_jsonl(path)
    if not certs:
        return False, "Phase 3 controlled certificate file is empty."
    actual = sum(1 for cert in certs if cert.get("actual_fork"))
    possible = sum(1 for cert in certs if cert.get("fork_possible"))
    calibration_ok = False
    if calibration_path is not None and calibration_path.exists():
        model = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_ok = (
            model.get("model_kind") == "empirical_independent_margin_delta_convolution"
            and int(model.get("margin_count", 0)) > 0
            and int(model.get("delta_count", 0)) > 0
        )
    ok = actual > 0 and calibration_ok
    return (
        ok,
        f"Phase 3 detector sanity gate: certificates={len(certs)}, "
        f"fork_possible={possible}, actual_forks={actual}, calibration_ok={calibration_ok}; "
        "require at least one actual_fork and a non-empty empirical calibration model.",
    )


def check_phase4(path: Path, logprob_path: Path | None, require_token_match: bool = False) -> tuple[bool, str]:
    if logprob_path is None:
        return False, "Phase 4 coverage check requires --logprob-jsonl."
    certs = read_jsonl(path)
    logprobs = read_jsonl(logprob_path)
    coverage_ok = len(certs) == len(logprobs)
    token_bad = 0
    token_missing = 0
    if require_token_match:
        for cert in certs:
            alignment = ((cert.get("metadata") or {}).get("rollout_alignment") or {})
            if "token_id_match" not in alignment or alignment.get("token_id_match") is None:
                token_missing += 1
            elif alignment.get("token_id_match") is not True:
                token_bad += 1
    ok = coverage_ok and (not require_token_match or (token_bad == 0 and token_missing == 0))
    return (
        ok,
        f"Phase 4 rollout coverage: certificates={len(certs)}, phase1_logprob_rows={len(logprobs)}, "
        f"token_bad={token_bad}, token_missing={token_missing}; require equality"
        + (" and token_id matches." if require_token_match else "."),
    )


def check_phase15(path: Path) -> tuple[bool, str]:
    rows = read_jsonl(path)
    levels = {str(row.get("level")) for row in rows}
    required = {"L1", "L2", "L3", "L4", "L5", "L6"}
    missing = sorted(required - levels)
    indexed = bool(rows) and all(row.get("residual_layer_indexed") is True for row in rows)
    ok = not missing and indexed
    return ok, (
        f"Phase 1.5 measured levels={sorted(levels)}, residual_layer_indexed={indexed}; "
        f"require {sorted(required)} and propagation measurements grouped by transformer layer depth."
    )


def check_phase5(path: Path, require_token_match: bool = False) -> tuple[bool, str]:
    certs = read_jsonl(path)
    if not certs:
        return False, "Phase 5 bug-injection certificate file is empty."
    expected_bug = [cert for cert in certs if (cert.get("metadata") or {}).get("phase5_expected_bug") is True]
    expected_valid = [cert for cert in certs if (cert.get("metadata") or {}).get("phase5_expected_bug") is False]
    unlabeled = len(certs) - len(expected_bug) - len(expected_valid)
    true_positive = sum(1 for cert in expected_bug if cert.get("region") == "bug")
    false_negative = len(expected_bug) - true_positive
    false_positive = sum(1 for cert in expected_valid if cert.get("region") == "bug")
    true_negative = len(expected_valid) - false_positive
    non_kernel = sum(
        1
        for cert in expected_bug
        if ((cert.get("metadata") or {}).get("bug") or {}).get("injection_kind") != "kernel_execution"
    )
    token_bad = 0
    token_missing = 0
    if require_token_match:
        for cert in certs:
            alignment = ((cert.get("metadata") or {}).get("rollout_alignment") or {})
            if "token_id_match" not in alignment or alignment.get("token_id_match") is None:
                token_missing += 1
            elif alignment.get("token_id_match") is not True:
                token_bad += 1
    ok = (
        bool(expected_bug)
        and bool(expected_valid)
        and unlabeled == 0
        and false_negative == 0
        and false_positive == 0
        and non_kernel == 0
        and (not require_token_match or (token_bad == 0 and token_missing == 0))
    )
    return (
        ok,
        f"Phase 5 confusion gate: certificates={len(certs)}, TP={true_positive}, FN={false_negative}, "
        f"FP={false_positive}, TN={true_negative}, unlabeled={unlabeled}, "
        f"non_kernel_injections={non_kernel}, token_bad={token_bad}, token_missing={token_missing}; "
        "require zero FN/FP, executed kernel-level positives, and legal-pair negative controls"
        + (" and token_id matches." if require_token_match else "."),
    )


def check_phase6(path: Path, phase4_path: Path | None = None, require_autograd: bool = False) -> tuple[bool, str]:
    rows = read_jsonl(path)
    if phase4_path is not None:
        phase4_rows = read_jsonl(phase4_path)
        coverage_ok = len(rows) == len(phase4_rows)
    else:
        phase4_rows = rows
        coverage_ok = True
    actual = [row for row in rows if row.get("actual_fork")]
    missing_grad = sum(1 for row in actual if row.get("grad_contribution_diff") is None)
    zero_grad = sum(1 for row in actual if row.get("grad_contribution_diff") is not None and float(row.get("grad_contribution_diff", 0.0)) <= 0.0)
    non_autograd = sum(1 for row in actual if row.get("grad_contribution_mode") != "hf_autograd")
    ok = coverage_ok and missing_grad == 0 and zero_grad == 0 and (not require_autograd or non_autograd == 0)
    return (
        ok,
        f"Phase 6 grad gate: certificates={len(rows)}, phase4_certificates={len(phase4_rows)}, "
        f"actual_forks={len(actual)}, missing_grad={missing_grad}, zero_grad_diff={zero_grad}, "
        f"non_autograd={non_autograd}; require coverage and nonzero grad diff for actual forks"
        + (" with hf_autograd evidence." if require_autograd else "."),
    )


def check_phase6_twin(path: Path, phase4_path: Path | None = None) -> tuple[bool, str]:
    if phase4_path is None:
        return False, "Phase 6 twin gate requires --phase4-certificates."
    phase4_rows = read_jsonl(phase4_path)
    natural_forks = sum(1 for row in phase4_rows if row.get("actual_fork"))
    data = json.loads(path.read_text(encoding="utf-8"))
    status = str(data.get("status", "missing"))
    if natural_forks == 0:
        ok = status == "not_triggered"
        return ok, f"Phase 6 twin gate: natural_forks=0, status={status}; require not_triggered."
    steps = int(data.get("optimizer_steps", 0))
    measurements = int(data.get("weight_measurements", 0))
    fork_events = int(data.get("total_fork_events", 0))
    ok = (
        status == "completed"
        and data.get("backend_only_difference") is True
        and data.get("exact_weight_divergence") is True
        and data.get("weight_scope") == "full_model"
        and steps >= 100
        and measurements >= 20
        and fork_events > 0
    )
    return (
        ok,
        f"Phase 6 twin gate: natural_forks={natural_forks}, status={status}, steps={steps}, "
        f"weight_measurements={measurements}, fork_events={fork_events}; require completed backend-only "
        "full-model lockstep training with >=100 steps, >=20 exact divergence measurements, and reproduced forks.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ForkCert hard gates against existing phase outputs.")
    parser.add_argument("--phase", choices=["phase0", "phase1", "phase15", "phase2", "phase3", "phase4", "phase5", "phase6", "phase6_twin"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--logprob-jsonl", default=None, help="Required for phase4 coverage check.")
    parser.add_argument("--phase4-certificates", default=None, help="Optional Phase 4 certificates for phase6 coverage check.")
    parser.add_argument("--calibration-json", default=None, help="Required for the complete Phase 3 gate.")
    parser.add_argument("--require-token-match", action="store_true", help="For phase4, require cert metadata to prove rollout token_id matches Phase 1 token_id.")
    parser.add_argument("--require-autograd", action="store_true", help="For phase6, reject branch-proxy gradient annotations.")
    args = parser.parse_args()

    checks = {
        "phase0": check_phase0,
        "phase1": check_phase1,
        "phase2": check_phase2,
        "phase3": check_phase3,
    }
    if args.phase == "phase4":
        ok, message = check_phase4(
            Path(args.input),
            Path(args.logprob_jsonl) if args.logprob_jsonl else None,
            require_token_match=args.require_token_match,
        )
    elif args.phase == "phase15":
        ok, message = check_phase15(Path(args.input))
    elif args.phase == "phase5":
        ok, message = check_phase5(Path(args.input), require_token_match=args.require_token_match)
    elif args.phase == "phase6":
        ok, message = check_phase6(
            Path(args.input),
            Path(args.phase4_certificates) if args.phase4_certificates else None,
            require_autograd=args.require_autograd,
        )
    elif args.phase == "phase6_twin":
        ok, message = check_phase6_twin(
            Path(args.input),
            Path(args.phase4_certificates) if args.phase4_certificates else None,
        )
    elif args.phase == "phase3":
        ok, message = check_phase3(
            Path(args.input),
            Path(args.calibration_json) if args.calibration_json else None,
        )
    else:
        ok, message = checks[args.phase](Path(args.input))
    print(message)
    if not ok:
        code = {"phase0": 20, "phase1": 21, "phase15": 25, "phase2": 22, "phase3": 28, "phase4": 24, "phase5": 26, "phase6": 27, "phase6_twin": 29}[args.phase]
        print(f"{args.phase} hard gate failed.", file=sys.stderr)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
