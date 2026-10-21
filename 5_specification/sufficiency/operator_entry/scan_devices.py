#!/usr/bin/env python3
"""Device of every tensor argument per recorded key (path A2 deviation D-A2-2).

    scan_devices.py --rec REC_DIR     (REC_DIR from record_workload.py; writes REC_DIR/devices.jsonl)

The A2 recordings saved values on the host without the original device. This reruns the workload's step under a
Recorder without values (same exclusion as run.json), takes each key's devices from its structural key, and checks that
the key numbering and operators equal the recording's keys.jsonl before writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import record as R  # noqa: E402

R.install_launch_hook()
import flag_gems  # noqa: E402
import record_workload as RW  # noqa: E402


def devices(struct):
    out = []

    def walk(x):
        if isinstance(x, tuple) and x and x[0] == "T":
            out.append(x[4])
        elif isinstance(x, tuple):
            for v in x[1:]:
                walk(v)
    walk(struct)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True)
    a = ap.parse_args()
    run = json.loads((a.rec / "run.json").read_text())
    flag_gems.enable(unused=list(run["exclude"]))
    step = RW.trigger_step(run["trigger"]) if run.get("trigger") else RW.model_step(run["model"], run["config"])
    step()
    rec = R.Recorder(keep_values=False)
    with rec:
        step()
    import torch
    torch.cuda.synchronize()
    recorded = [json.loads(l) for l in open(a.rec / "keys.jsonl")]
    by_index = {i: k for k, i in rec.keys.items()}
    if len(by_index) != len(recorded) or any(by_index[k["key"]][0] != k["op"] for k in recorded):
        raise SystemExit("key numbering or operators differ from the recording; devices not written")
    with open(a.rec / "devices.jsonl", "x") as fh:
        for k in recorded:
            op, sa, sk = by_index[k["key"]]
            fh.write(json.dumps({"key": k["key"], "op": op, "args": devices(sa), "kwargs": devices(sk)}) + "\n")
    print(json.dumps({"keys": len(recorded),
                      "with_cpu_tensors": sum(1 for k in recorded if "cpu" in devices(by_index[k["key"]][1])
                                              + devices(by_index[k["key"]][2]))}))


if __name__ == "__main__":
    main()
