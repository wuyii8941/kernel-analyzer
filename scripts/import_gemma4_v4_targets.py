#!/usr/bin/env python3
"""Import compact results from the fresh in-process Gemma target runs.

The two runs were frozen before their 32-step consequence was evaluated.  This
adapter copies only small JSON summaries into the repository and records the
runtime capture digest.  It never turns a missing or malformed run into a
negative result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def one(source_dir: Path, target_dir: Path, case_id: str, target_region: str) -> dict[str, Any]:
    required = {name: source_dir / name for name in (
        "formation.json", "prediction.json", "consequence.json", "short_screen.json"
    )}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        return {"case_id": case_id, "status": "ABSTAIN_MISSING_RUN_FILES", "missing": missing}
    formation = read(required["formation.json"])
    prediction = read(required["prediction.json"])
    consequence = read(required["consequence.json"])
    short = read(required["short_screen.json"])
    if formation.get("status") != "COMPLETE" or prediction.get("status") != "PREDICTION_FROZEN_BEFORE_CONSEQUENCE":
        return {"case_id": case_id, "status": "ABSTAIN_INVALID_PRETRAJECTORY_RUN"}
    if consequence.get("status") != "COMPLETE" or consequence.get("steps") != 32:
        return {"case_id": case_id, "status": "ABSTAIN_INCOMPLETE_CONSEQUENCE"}
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, path in required.items():
        destination = target_dir / f"{case_id}.{name}"
        shutil.copyfile(path, destination)
        copied[name] = {
            "path": str(destination),
            "sha256": digest(destination),
        }
    stats = consequence.get("statistics", {})
    local = stats.get("local", {})
    feedback = stats.get("feedback", {})
    actual = stats.get("actual", {})
    local_path_energy = float(local.get("path_energy", 0.0) or 0.0)
    if local_path_energy <= 1e-20:
        row_status = "NOT_APPLICABLE_NO_OBSERVED_CARRIER_EFFECT"
    elif float(actual.get("coherence_amplification", 0.0) or 0.0) > 1.1:
        row_status = "COMPLETE_NEW_IMPL_FEEDBACK_CONTROL"
    else:
        row_status = "COMPLETE_NEW_IMPL_DIRECT_NEGATIVE"
    return {
        "case_id": case_id,
        "status": row_status,
        "model": "google/gemma-4-E2B",
        "target_region": target_region,
        "carrier": consequence.get("carrier"),
        "prediction": prediction.get("source_prediction"),
        "prediction_frozen_before_consequence": True,
        "short_screen": {
            "status": short.get("status"),
            "source": short.get("input", {}).get("case_id", case_id),
        },
        "formation": formation.get("prediction", {}).get("evidence"),
        "consequence": {
            "steps": consequence.get("steps"),
            "local_A32": local.get("coherence_amplification"),
            "feedback_A32": feedback.get("coherence_amplification"),
            "actual_A32": actual.get("coherence_amplification"),
            "local_resultant_l2": local.get("resultant_l2"),
            "feedback_resultant_l2": feedback.get("resultant_l2"),
            "actual_resultant_l2": actual.get("resultant_l2"),
            "final_drift_l2": consequence.get("final_drift_l2"),
        },
        "runtime_capture_sha256": digest(source_dir / "runtime_release" / "capture.json"),
        "source_dir": str(source_dir),
        "artifacts": copied,
        "claim_boundary": (
            "This is a fresh in-process Gemma implementation check. The source score was frozen "
            "before the paired trajectory. A negative direct score does not prove full training safety; "
            "a large actual score with local_A32 near one is reported as feedback-sustained."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data1/tzh/cache/kernel_analyzer/direct_persistence_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/property/direct_persistence_v4/heldout"))
    args = parser.parse_args()
    rows = [
        one(args.root / "gemma4_random_softmax_v3", args.output_dir,
            "gemma4_random_softmax_backward", "backward:1880/in_out_ptr0"),
        one(args.root / "gemma4_random_gelu_v3", args.output_dir,
            "gemma4_random_gelu_loss_backward", "backward:1401/out_ptr3"),
    ]
    v3 = args.root / "gemma4_random_gelu_backward1860_v4"
    if v3.is_dir():
        rows.append(one(
            v3,
            args.output_dir,
            "gemma4_random_gelu_backward1860",
            "backward:1860/in_out_ptr0",
        ))
    payload = {
        "schema": "kernel-analyzer-direct-persistence-v4-new-impl-targets-v2",
        "status": "COMPLETE_FRESH_IN_PROCESS_GEMMA_ROWS" if all(not row.get("status", "").startswith("ABSTAIN") for row in rows) else "PARTIAL_FAIL_CLOSED",
        "rows": rows,
        "selection": "Two targets were selected from the pre-frozen Gemma new-implementation pool before their consequence runs.",
        "metrics": {
            "eligible_rows": sum(row.get("status", "").startswith("COMPLETE") for row in rows),
            "confirmed_direct_positive": sum(row.get("status") == "COMPLETE_NEW_IMPL_DIRECT_POSITIVE" for row in rows),
            "confirmed_direct_negative": sum(row.get("status") == "COMPLETE_NEW_IMPL_DIRECT_NEGATIVE" for row in rows),
            "feedback_controls": sum(row.get("status") == "COMPLETE_NEW_IMPL_FEEDBACK_CONTROL" for row in rows),
            "not_applicable_no_observed_carrier_effect": sum(row.get("status") == "NOT_APPLICABLE_NO_OBSERVED_CARRIER_EFFECT" for row in rows),
            "recall": None,
            "auroc": None,
        },
        "claim_boundary": "These rows add NEW_IMPL controls or candidates. Recall and AUROC remain undefined unless the frozen pool contains both a confirmed direct positive and confirmed negatives.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "new_impl_targets_v2.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    confirmation = {
        "schema": "kernel-analyzer-direct-persistence-v4-new-impl-confirmation-v2",
        "status": payload["status"],
        "rows": rows,
        "counting": {
            "fresh_rows": len(rows),
            "fresh_direct_positive": payload["metrics"]["confirmed_direct_positive"],
            "fresh_direct_negative": payload["metrics"]["confirmed_direct_negative"],
            "fresh_feedback_controls": payload["metrics"]["feedback_controls"],
            "fresh_not_applicable": payload["metrics"]["not_applicable_no_observed_carrier_effect"],
        },
        "claim_boundary": payload["claim_boundary"],
    }
    confirmation_path = args.output_dir.parent / "heldout_confirmation_v2.json"
    confirmation_path.write_text(json.dumps(confirmation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(destination), "confirmation": str(confirmation_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
