"""Summarize observed failure chronology without inferring an unmeasured cause."""
import argparse
import json
from pathlib import Path


def summarize(root):
    events = []
    steps = []
    for path in sorted(root.glob("step_*.json")):
        document = json.loads(path.read_text())
        step = document["step"]
        steps.append(step)
        for row in document["parameters"]:
            for field in ("parameter_before", "gradient", "effective_second", "next_second", "parameter_after"):
                value = row[field]
                if value["nonfinite"] or (field in ("effective_second", "next_second") and value["negative"]):
                    events.append({"step": step, "parameter": row["name"], "field": field,
                                   "compensated": row["compensated"], "statistics": value})
    terminal_path = root / "terminal.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    first_step = min((event["step"] for event in events), default=None)
    return {"schema": "key-only-observed-failure-chronology-v1",
            "observed_steps": steps, "terminal": terminal,
            "first_observed_anomaly_step": first_step,
            "first_observed_anomalies": [e for e in events if e["step"] == first_step],
            "all_anomalies": events,
            "boundary": "Earliest recorded anomaly, not necessarily earliest cause; observation starts at step 900.",
            "status": "TERMINAL_RECORD_PRESENT" if terminal or (root / "run.json").exists() else "PARTIAL"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to(repo):
        parser.error("output must stay in repository")
    result = summarize(args.root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
