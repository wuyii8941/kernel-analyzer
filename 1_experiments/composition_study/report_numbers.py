"""Every count the README quotes, generated from the result files (results/numbers.json).

Sources: the cell sweeps (strict status, cells.rescore), the recheck of the partial cells (reasons), the library
corpus, the predictor validation (held-out programs) and the TF32 modification.
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"


def cells(path):
    from cells import rescore
    rows = [rescore(json.loads(l)) for l in gzip.decompress(path.read_bytes()).decode().splitlines() if l.strip()]
    el = Counter()
    for r in rows:
        for L in r.get("launches", []) or []:
            el.update({k: v for k, v in (L.get("elements") or {}).items() if isinstance(v, int)})
    return {"cells": len(rows), "status": dict(Counter(r["status"] for r in rows)), "elements": dict(el)}


def partial_reasons(path):
    if not path.exists():
        return None
    rows = [json.loads(l) for l in gzip.decompress(path.read_bytes()).decode().splitlines() if l.strip()]
    why, category = Counter(), Counter()
    undefined = ("undefined", "poison", "out of range", "domain")
    for r in rows:
        ks = set()
        for L in r.get("launches", []) or []:
            ks |= set(L.get("not_established_reasons") or {})
        for k in ks or {"partial-sum overflow in a reduce / scan (outside the model, no reason recorded)"}:
            why[k] += 1
        cat = ("no declared value" if any(w in k for k in ks for w in undefined) else
               ("reduce / scan overflow" if not ks else "other"))
        category[f"{r['status']}: {cat}"] += 1
    return {"cells": len(rows), "status": dict(Counter(r["status"] for r in rows)), "cells_by_category": dict(category),
            "cells_by_reason": dict(why)}


def corpus(path):
    d = json.loads(path.read_text())
    s = d["summary"]
    keys = ("calls", "distinct_kernels", "status", "elements", "kernels_without_k1", "k2_pairs_by_status",
            "kernels_all_pairs_bit_identical", "k1_kernels")
    out = {k: s.get(k) for k in keys}
    out["mismatch_kernels"] = sorted({(k["call"], k["kernel"]) for k in d["kernels"] if k["status"] == "mismatch"})
    out["not_established_kernels"] = sorted({(k["call"], k["kernel"], (k.get("not_emulable") or k.get("abort_reasons")
                                                                      or [""])[0][:60])
                                             for k in d["kernels"] if k["status"] == "not_established"})
    out["call_failures"] = [c["call"] for c in s.get("call_failures", [])]
    return out


def validation(path):
    if not path.exists():
        return None
    V = json.loads(path.read_text())
    out = {"programs": len(V), "verdict": dict(Counter(v["verdict"] for v in V)),
           "null_verdict": dict(Counter(v.get("null_verdict") for v in V)),
           "verdict_x_null": {f"{a} / null {b}": n for (a, b), n in
                              Counter((v["verdict"], v.get("null_verdict")) for v in V).items()},
           "by_form": {}}
    for v in V:
        out["by_form"].setdefault(v["form"], Counter())[v["verdict"]] += 1
    out["by_form"] = {k: dict(c) for k, c in out["by_form"].items()}
    return out


def main():
    cache = HERE.parents[1] / ".cache/composition_study"
    numbers = {
        "cells_default": cells(R / "cells_default.jsonl.gz"),
        "cells_options": cells(R / "cells_options.jsonl.gz"),
        "partial_recheck": partial_reasons(R / "cells_partial_recheck.jsonl.gz"),
        "corpus": corpus(R / "corpus.json"),
        "validation_heldout_v1": validation(R / "predictor_heldout_verdicts.json"),
        "validation_heldout_v2": validation(R / "predictor_heldout2_verdicts.json"),
        "validation_development_dev0": validation(R / "predictor_dev_verdicts.json"),
        "validation_development_dev1": validation(R / "predictor_dev1_verdicts.json"),
    }
    for key in ("validation_heldout_v1", "validation_heldout_v2"):
        V = json.loads((R / ("predictor_heldout_verdicts.json" if key.endswith("v1")
                             else "predictor_heldout2_verdicts.json")).read_text())
        strict = Counter()
        for v in V:
            if v["verdict"] == "成立":
                ok = all(abs(o["difference"]) <= 3 * (o["se_pred"] ** 2 + o["se_meas"] ** 2) ** 0.5
                         + 0.1 * abs(o["predicted"]) for o in v["outputs"].values())
                strict["holds without the second-order term" if ok else "holds only with it"] += 1
        numbers[key]["post_hoc_strict_tolerance"] = dict(strict)
    tf = R / "tf32_modification_verdicts.json"
    if tf.exists():
        d = json.loads(tf.read_text())
        P = [json.loads(l) for l in (R / "tf32_modification_predictions.jsonl").read_text().splitlines() if l.strip()]
        M = [json.loads(l) for l in (R / "tf32_modification_measurements.jsonl").read_text().splitlines() if l.strip()]
        strict = Counter()
        for v in d["verdicts"]:
            key = (v["variant"], v["layer"], v["matrix"])
            p = [r["c_ptr"]["alignment"] for r in P if (r["variant"], r["layer"], r["matrix"]) == key]
            m = [r["tiles"] for r in M if (r["variant"], r["layer"], r["matrix"]) == key]

            def se(x):
                mu = sum(x) / len(x)
                return (sum((a - mu) ** 2 for a in x) / (len(x) - 1)) ** 0.5 / len(x) ** 0.5
            se_m = max(se([r["alignment"] for r in m]), sum(r["se_iid"] ** 2 for r in m) ** 0.5 / len(m))
            tol = 3 * (se(p) ** 2 + se_m ** 2) ** 0.5 + 0.1 * abs(sum(p) / len(p))
            ok = abs(sum(r["alignment"] for r in m) / len(m) - sum(p) / len(p)) <= tol
            strict[f"{v['variant']} {'成立' if ok else '不成立'}"] += 1
        numbers["tf32_modification"] = {"verdict": dict(Counter(f"{v['variant']} {v['verdict']}"
                                                                for v in d["verdicts"])),
                                        "post_hoc_strict_tolerance": dict(strict), "effects": d["effects"]}
    (R / "numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(numbers, indent=1, ensure_ascii=False, default=str)[:6000])


if __name__ == "__main__":
    main()
