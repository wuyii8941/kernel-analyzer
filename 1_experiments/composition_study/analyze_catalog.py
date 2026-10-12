"""Summary of the primitive catalog (RQ1): one class per primitive, counts per class, and the primitives whose local
residual has a systematic mean (the B1 sites).

Classes (float results; ulp = spacing of the result format at the exact value):
  correctly rounded            every finite comparison equals the declared value rounded in the declared mode, and
                               nan / inf / signed-zero classes agree with IEEE float64 behaviour;
  correctly rounded, specials  the same for finite values, but some special-value classes differ;
  faithful                     not always correctly rounded, every error below 1 ulp;
  approximate                  larger errors (the bound is reported);
  flags: flushes subnormals; systematic mean error (|mean| > 4 standard errors and > 0.02 ulp in some distribution).
Exact results (integers, predicates, bit casts, selects): exact / exact except subnormal inputs flushed / mismatch.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

FLOAT_FAMILIES = ("real", "real_mixed", "convert_ff", "convert_if")


def compile_reason(err: str, diag=()) -> str:
    err = " ".join([err] + list(diag or []))
    if "failed to legalize" in err:
        return "not legalized by the NVIDIA backend (the frontend never emits it)"
    if "compute capability >= 89" in err:
        return "fp8 e4m3 (nv) needs sm_89"
    if "Unsupported bitwidth" in err:
        return "LLVM: unsupported bit width"
    if "Parse MLIR" in err:
        return "TTIR parse failure"
    return err[:120]


def classify(r) -> dict:
    fam, st = r.get("family"), r.get("status")
    out = {"pid": r["pid"], "family": fam}
    if st != "measured":
        out["class"] = st
        if r.get("error") or r.get("diagnostics"):
            out["reason"] = compile_reason(r.get("error", ""), r.get("diagnostics"))
        return out
    if fam == "literal":
        lit = r.get("literal", {})
        out["class"] = "literal equals RN(decimal)" if lit.get("equal_rn", True) else "literal differs from RN(decimal)"
        out["relative_rounding"] = lit.get("relative_rounding")
        return out
    if fam == "scalar":
        out["class"] = "exact" if r.get("roundtrip_equal") == r.get("n") else "mismatch"
        return out
    per = r.get("per_distribution") or {}
    if fam == "dot" or (fam == "mx" and any("cr" in v for v in per.values())):
        out["class"] = "dot"
        out["mean_relative_error"] = {d: v.get("mean_rel") for d, v in per.items()}
        out["correctly_rounded_share"] = {d: v.get("cr_share") for d, v in per.items()}
        out["max_abs_ulp"] = max((v.get("max_abs_ulp", 0) for v in per.values()), default=0)
        return out
    if fam in FLOAT_FAMILIES:
        n = sum(v.get("n", 0) for v in per.values())
        cr = sum(v.get("cr", 0) for v in per.values())
        mx = max((v.get("max_abs_ulp", 0) for v in per.values()), default=0.0)
        flush = sum(v.get("flushed_to_zero", 0) for v in per.values())
        sp_bad = sum(v.get("special_disagree", 0) for v in per.values())
        sp_n = sum(v.get("special_n", 0) for v in per.values())
        nonfin = sum(v.get("nonfinite_where_finite", 0) for v in per.values())
        bias, smooth = {}, ("symmetric", "same-sign", "heavy-tailed", "wide")
        for d, v in per.items():
            m, se = v.get("mean_ulp"), v.get("se_mean_ulp")
            if m is not None and se is not None and abs(m) > max(4 * se, 0.02) and v.get("n", 0) >= 100:
                bias[d] = round(m, 4)
        if cr == n and sp_bad == 0 and nonfin == 0:
            cls = "correctly rounded"
        elif cr == n and nonfin == 0:
            cls = "correctly rounded, specials differ"
        elif mx < 1 and nonfin == 0:
            cls = "faithful"
        else:
            cls = "approximate"
        out.update({"class": cls, "n": n, "correctly_rounded_share": cr / n if n else None, "max_abs_ulp": mx,
                    "flushes_subnormals": flush > 0, "flushed": flush, "special_disagree": sp_bad,
                    "special_checked": sp_n, "nonfinite_where_finite": nonfin, "systematic_mean_ulp": bias,
                    "special_examples": [e for v in per.values() for e in v.get("special_examples", [])][:3]})
        return out
    n = sum(v.get("n", 0) for v in per.values() if v.get("n") is not None)
    eq = sum(v.get("eq", 0) for v in per.values() if v.get("eq") is not None)
    ftz = sum(v.get("mismatch_explained_by_input_flush", 0) for v in per.values())
    und = Counter()
    for v in per.values():
        und.update(v.get("undefined", {}))
    if eq == n:
        cls = "exact"
    elif eq + ftz == n:
        cls = "exact except subnormal inputs flushed"
    else:
        cls = "mismatch"
    out.update({"class": cls, "n": n, "equal": eq, "flush_explained": ftz, "undefined_inputs": dict(und),
                "examples": [e for v in per.values() for e in v.get("examples", [])][:3]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(HERE / "results/catalog.json"))
    ap.add_argument("--out", default=str(HERE / "results/catalog_summary.json"))
    a = ap.parse_args()
    cat = json.loads(Path(a.catalog).read_text())
    rows = [classify(r) for r in cat["records"]]
    by = defaultdict(Counter)
    for r, raw in zip(rows, cat["records"]):
        src = raw.get("source") or raw["pid"].split("|")[0]
        by[src][r["class"]] += 1
    reasons = Counter(r.get("reason", "") for r in rows if r["class"] == "not compiled")
    not_legal_ops = sorted({r["pid"].split("|")[1] for r in rows if r["class"] == "not compiled"
                            and "legalized" in r.get("reason", "")})
    smooth = ("symmetric", "same-sign", "heavy-tailed", "wide")
    b1 = [{"pid": r["pid"], "class": r["class"], "max_abs_ulp": r.get("max_abs_ulp"),
           "systematic_mean_ulp": {d: v for d, v in r["systematic_mean_ulp"].items() if d in smooth}}
          for r in rows if r.get("class") in ("approximate", "faithful")
          and any(d in smooth for d in r.get("systematic_mean_ulp", {}))]
    data_structured = [{"pid": r["pid"], "systematic_mean_ulp": r["systematic_mean_ulp"]} for r in rows
                       if r.get("class", "").startswith("correctly rounded") and r.get("systematic_mean_ulp")]
    out = {"schema": "composition-study-catalog-summary-v1", "source": Path(a.catalog).name,
           "primitives": len(rows), "classes_by_source": {k: dict(v) for k, v in by.items()},
           "not_compiled_reasons": dict(reasons), "registered_but_not_legalized": not_legal_ops,
           "systematic_mean_sites": b1, "correctly_rounded_with_data_structured_mean": data_structured,
           "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: out[k] for k in ("primitives", "classes_by_source", "not_compiled_reasons")}, indent=1,
                     ensure_ascii=False))
    print(len(b1), "approximate / faithful primitives with a systematic mean error in a smooth distribution;",
          len(data_structured), "correctly rounded primitives with a data-structured mean")


if __name__ == "__main__":
    main()
