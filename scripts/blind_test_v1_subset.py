#!/usr/bin/env python3
"""Data subset for the reviewer's independent check (phase-2 release, section 6).

Programs detected under R2 or R3 in phase 1; seeds 0 and 1; one .npz per program with the input tensors,
the runtime scalars and the output K of both seeds (keys seed{0,1}__<name>, seed{0,1}__K).  Families F1,
F3 and F5 go to the priority set.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "programs"))
    inputs = importlib.import_module("inputs")
    report = json.loads(args.phase1.read_text())
    chosen = sorted(p for p, r in report.items() for o in r.get("outputs", []) if isinstance(o.get("rules"), dict)
                    and any(str(o["rules"].get(k, {}).get("final_verdict", "")).startswith("DETECTED") for k in ("R2", "R3")))
    index = []
    for pid in chosen:
        mod = importlib.import_module(pid)
        arrays, scalars = {}, {}
        for seed in (0, 1):
            inp = inputs.make_inputs(mod.FAMILY, seed)
            y = mod.launch(inp)
            torch.cuda.synchronize()
            for key, v in inp.items():
                if torch.is_tensor(v):
                    arrays[f"seed{seed}__{key}"] = v.detach().cpu().numpy()
                else:
                    arrays[f"seed{seed}__scalar__{key}"] = np.array(v, dtype=np.float64)
                    scalars[f"seed{seed}__{key}"] = repr(v)
            arrays[f"seed{seed}__K"] = y.detach().cpu().numpy()
        group = "priority_F1_F3_F5" if mod.FAMILY in ("F1", "F3", "F5") else "others"
        target = args.out / group / f"{pid}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        index.append({"program": pid, "family": mod.FAMILY, "file": f"{group}/{pid}.npz", "scalars_repr": scalars,
                      "size_bytes": target.stat().st_size})
        print(pid, mod.FAMILY, group, target.stat().st_size, flush=True)
    (args.out / "index.json").write_text(json.dumps({
        "selection": "programs detected under R2 or R3 in phase 1 (Holm over program x rule)", "seeds": [0, 1],
        "keys": "seed{0,1}__<input tensor>, seed{0,1}__scalar__<name> (float64 of the Python value), seed{0,1}__K",
        "programs": index}, indent=2) + "\n")


if __name__ == "__main__":
    main()
