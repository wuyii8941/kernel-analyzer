#!/usr/bin/env python3
"""Item A3 (docs/protocol_essential_bugs_phase2_20261007.md): is the source of an FR "interface / constant" item verified?

For every Inductor condition whose phase-1 FR had interface items (e_sem excluding zero below the action threshold),
re-capture one seed with detector 2.2 and inventory the kernels' compile-time float constants and runtime scalars
(`kernel_analyzer.reference_eval.interface.inventory`).  "Verified" means the captured kernels contain at least one
rounded compile-time constant or rounded runtime scalar (e.g. float32(eps), float32(eps / C), float32(1 / divisor)),
listed per condition; otherwise "not verified".  Writes results/essential/phase2a/a3_constants.json.

    python scripts/essential/a3_constants.py
"""
import gzip
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fr_stage as FS  # noqa: E402
import run_phase1 as R  # noqa: E402

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.interface import inventory  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

ROOT = common.ROOT
OUT = ROOT / "results/essential/phase2a"


def targets():
    """(family, candidate, condition id) with interface items in the phase-1 FR assessment."""
    out = []
    for fam in ("ce", "pool", "index"):
        d = json.load(gzip.open(ROOT / f"results/essential/phase1/classification_{fam}.json.gz", "rt"))
        for r in d["records"]:
            for cand in ("inductor_cuda32", "inductor_cuda_bf16"):
                fr = (r["candidates"].get(cand) or {}).get("FR") or {}
                if fr.get("status") != "ok":
                    continue
                if any((o.get("readings") or {}).get(list(o["readings"])[0], {}) and
                       o["readings"][list(o["readings"])[0]]["interface_elements"] > 0
                       for o in fr["outputs"].values() if o.get("readings")):
                    out.append((fam, cand, r["condition"]["id"]))
    return out


def main():
    import torch._inductor.config as ic
    ic.use_static_cuda_launcher = False
    TritonLaunchRecorder.install_hook()
    rows = []
    tg = targets()
    print("targets", len(tg), flush=True)
    for k, (fam, cand, cid) in enumerate(tg):
        cond = {c["id"]: c for c in R.FAMILIES[fam][0]()}[cid]
        try:
            setup, tensors, launch = FS._case(fam, cond, cand)
            setup()
            t = tensors(0)
            rec = TritonLaunchRecorder()
            with rec:
                launch(t)
                torch.cuda.synchronize()
            consts, scalars = [], []
            for l in rec.launches:
                inv = inventory(l, parse_ttir(l.asm["ttir"]))
                consts += [{"kernel": l.kernel_name[:60], **c} for c in inv["rounded_compile_time_constants"]]
                scalars += [{"kernel": l.kernel_name[:60], **s} for s in inv["runtime_scalars"] if s.get("exact") is False]
            rows.append({"family": fam, "candidate": cand, "condition": cid, "verified": bool(consts or scalars),
                         "rounded_compile_time_constants": consts[:12], "rounded_runtime_scalars": scalars[:12]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"family": fam, "candidate": cand, "condition": cid, "verified": None,
                         "error": f"{type(exc).__name__}: {exc}"[:300]})
        if (k + 1) % 25 == 0:
            print(k + 1, "done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a3_constants.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    v = [r["verified"] for r in rows]
    print("verified", v.count(True), "not verified", v.count(False), "error", v.count(None))


if __name__ == "__main__":
    main()
