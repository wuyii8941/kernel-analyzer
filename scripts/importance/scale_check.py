#!/usr/bin/env python3
"""Second-scale stability of rho (docs/importance_calibration_protocol_20261006.md section 10).

    python scripts/importance/scale_check.py train --ref R      # reference runs with saved states (scale 2)
    python scripts/importance/scale_check.py train --ref Rp
    python scripts/importance/scale_check.py measure --items A1,A2  # single-step effects at scale 2
    python scripts/importance/scale_check.py compare             # rho at both scales -> compare.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ITEMS, REFS, STATE_STEPS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/importance/scale_check"
CKPT = ROOT / ".cache/importance/ckpt_scale2"
SIZE = {"d": 512, "layers": 8, "heads": 8, "ffn": 1376}
CHECK = ("A1", "A2", "A3", "X1", "X2", "X3")


def train(ref):
    import small_lm as S

    out = OUT / f"{ref}__s0.json"
    if out.exists():
        return
    c = S.Config(seed=0, **REFS[ref], **SIZE, checkpoints=STATE_STEPS)
    r = S.train(c, out_dir=CKPT / f"{ref}_s0")
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=1, default=float) + "\n")


def measure(items):
    import single_step as SS
    import small_lm as S

    for name in items:
        out = OUT / f"single_step_{name}.json"
        if out.exists():
            continue
        kw, ref = ITEMS[name]
        S._COMPILED.clear()
        ck = [CKPT / f"{ref}_s0" / f"ckpt_{t:05d}.pt" for t in STATE_STEPS]
        rows, n_dev, extra = SS.measure(ck, S.Config(**REFS[ref], **SIZE), S.Config(**kw, **SIZE))
        eff = SS.effects(rows, n_dev, extra)
        eff.update(item=name, scale=SIZE)
        out.write_text(json.dumps(eff, indent=1, default=float) + "\n")
        print(name, eff["gamma"]["mean"], eff["b"]["mean"], flush=True)


def compare():
    from scipy.stats import t as tdist

    def load(path):
        return json.loads(path.read_text())

    def ci975(e):
        q975, q95 = tdist.ppf(1 - 0.0125, e["n"] - 1), tdist.ppf(1 - 0.025, e["n"] - 1)
        h = (e["hi"] - e["mean"]) * q975 / q95
        return e["mean"] - h, e["mean"] + h

    s1 = {n: load(ROOT / "results/importance/single_step" / f"{n}.json") for n in CHECK}
    s2 = {n: load(OUT / f"single_step_{n}.json") for n in CHECK}
    res = {}
    for name in CHECK:
        for den in ("A1", "A2"):
            if name == den:
                continue
            for k in ("gamma", "b"):
                row = {}
                for tag, s in (("scale1", s1), ("scale2", s2)):
                    a0, a1 = ci975(s[den][k])
                    row[tag] = None if a0 <= 0 <= a1 else s[name][k]["mean"] / s[den][k]["mean"]
                if row["scale1"] is None or row["scale2"] is None:
                    row["verdict"] = "undefined (denominator not established)"
                else:
                    q = row["scale2"] / row["scale1"] if row["scale1"] else float("inf")
                    row["ratio_scale2_over_scale1"] = q
                    row["verdict"] = "stable" if 0.5 <= q <= 2 else "unstable"
                res[f"{name}/{den} {k}"] = row
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    for k, v in res.items():
        print(f"{k:16s} {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "measure", "compare"])
    ap.add_argument("--ref", default="R")
    ap.add_argument("--items", default=",".join(CHECK))
    a = ap.parse_args()
    if a.command == "train":
        train(a.ref)
    elif a.command == "measure":
        sys.path.insert(0, str(ROOT / "src"))
        measure([x for x in a.items.split(",") if x])
    else:
        compare()


if __name__ == "__main__":
    main()
