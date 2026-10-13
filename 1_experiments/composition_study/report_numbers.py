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
        "validation_heldout": validation(R / "predictor_heldout_verdicts.json"),
        "validation_development": validation(R / "predictor_dev_verdicts.json"),
    }
    tf = R / "tf32_modification_verdicts.json"
    if tf.exists():
        d = json.loads(tf.read_text())
        numbers["tf32_modification"] = {"verdict": dict(Counter((v["variant"], v["verdict"]) for v in d["verdicts"])),
                                        "effects": d["effects"]}
    (R / "numbers.json").write_text(json.dumps(numbers, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(numbers, indent=1, ensure_ascii=False, default=str)[:6000])


if __name__ == "__main__":
    main()
