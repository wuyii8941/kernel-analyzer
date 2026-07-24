#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.phase8_matched_step import state_distance


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge matched A/B/C trajectory arms and stream checkpoint distances.")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--c", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in [args.a, args.b, args.c]]
    if [payload["arm"] for payload in payloads] != ["A_reference", "B_alternative", "C_fusion_repair"]:
        raise ValueError("trajectory arm order mismatch")
    if len({payload["fork_id"] for payload in payloads}) != 1:
        raise ValueError("trajectory fork IDs differ")
    root = Path(args.out_dir)
    distances = []
    for step in [1, 5, 20]:
        dirs = {payload["arm"]: root / payload["arm"] / f"step_{step:02d}" for payload in payloads}
        ab = state_distance(dirs["A_reference"], dirs["B_alternative"])
        ac = state_distance(dirs["A_reference"], dirs["C_fusion_repair"])
        bc = state_distance(dirs["B_alternative"], dirs["C_fusion_repair"])
        distances.append({"step": step, "A_B": ab, "A_C": ac, "B_C": bc, "recovery_ratio_A_C_over_A_B": ac["l2"] / ab["l2"] if ab["l2"] else None})
    output = {"schema_version": "forkcert.trajectory.v1", "fork_id": payloads[0]["fork_id"], "arms": payloads, "distances": distances}
    Path(args.out).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fork_id": output["fork_id"], "distances": distances}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
