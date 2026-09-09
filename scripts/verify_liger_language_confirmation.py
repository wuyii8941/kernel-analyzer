#!/usr/bin/env python3
"""Independently check recorded execution; do not change the frozen loss test."""
import hashlib
import json
import math
from pathlib import Path

from scripts.run_liger_language_confirmation import OUT, N
from scripts.run_training_numerical_v2 import ROOT, save_new


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    plan = json.loads((OUT / "plan.json").read_text())
    failures, records = [], []
    if digest(ROOT / "scripts/run_liger_language_confirmation.py") != plan["runner_sha256"]:
        failures.append("confirmation source changed")
    for pair in range(N):
        directory = OUT / f"pair{pair}"
        protocol = json.loads((directory / "protocol.json").read_text())
        if protocol["model"]["initialization_seed"] != plan["initialization_seeds"][pair]:
            failures.append(f"pair {pair}: initialization differs from plan")
        for filename, expected in protocol["source_sha256"].items():
            if digest(ROOT / filename) != expected:
                failures.append(f"pair {pair}: source changed: {filename}")
        for split in ("train", "validation"):
            if digest(directory / f"{split}.npy") != protocol["data"][split]["encoded_sha256"]:
                failures.append(f"pair {pair}: {split} data changed")
        for condition in ("candidate", "reference"):
            folder = directory / condition
            if not (folder / "status.json").exists():
                failures.append(f"pair {pair}/{condition}: incomplete")
                continue
            status = json.loads((folder / "status.json").read_text())
            rows = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
            valid = (status.get("status") == "COMPLETE_DEVELOPMENT_PILOT"
                     and len(rows) == plan["steps"]
                     and all(row["step"] == i + 1 and math.isfinite(row["loss"]) for i, row in enumerate(rows))
                     and [e["step"] for e in status.get("evaluations", [])]
                     == list(range(protocol["eval_every"], plan["steps"] + 1, protocol["eval_every"]))
                     and all(math.isfinite(e["shared_evaluation_loss"]) for e in status.get("evaluations", [])))
            if not valid:
                failures.append(f"pair {pair}/{condition}: incomplete or nonfinite measurements")
            checkpoint_ok = (folder / "final.pt").is_file() and digest(folder / "final.pt") == status.get("final_checkpoint_sha256")
            if not checkpoint_ok:
                failures.append(f"pair {pair}/{condition}: checkpoint digest mismatch")
            records.append({"pair": pair, "condition": condition,
                            "status_sha256": digest(folder / "status.json"),
                            "steps_sha256": digest(folder / "steps.jsonl"),
                            "checkpoint_verified": checkpoint_ok})
    result = {"status": "VERIFIED_RECORDED_EXECUTION" if not failures else "NOT_VERIFIED",
              "plan_sha256": digest(OUT / "plan.json"), "failures": failures, "records": records,
              "scope": "Recorded sources, data, finite losses, horizon and checkpoint integrity; not a proof of bias or reference semantics. The frozen summary additionally checks paired inputs and initialization."}
    save_new(OUT / "execution_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    verify()
