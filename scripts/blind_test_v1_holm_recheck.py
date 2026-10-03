#!/usr/bin/env python3
"""Post-submission check of the frozen blind_test_v1 Holm decisions under the corrected conservative p-value.

The submitted runs used p = max(p_l, p_h) (two two-sided t-test p-values of the projection endpoints).  For
a box straddling zero with E[l] < 0 < E[h] both significant this is small although the interval verdict is
NOT_CONFIRMED, and Holm counted such tests as rejections, which loosens the thresholds of the others.  The
corrected p-value, 2 min(p(E[l] > 0), p(E[h] < 0)), is at least 1 for those tests and equals the old one on
every cell whose interval excludes zero (the weaker endpoint carries the decision).  So the recheck sets the
affected tests to p = 1, keeps the other p-values, reruns Holm and compares the final verdicts.

    python scripts/blind_test_v1_holm_recheck.py --out results/reference_eval/blind_test_v1/holm_pvalue_recheck.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "reference_eval" / "blind_test_v1"
ALPHA = 0.05


def cells(report, mode=None):
    out = []
    for pid, rep in report.items():
        outputs = rep["modes"][mode] if mode else rep.get("outputs", [])
        for o in outputs:
            if isinstance(o.get("rules"), dict):
                for rule, x in o["rules"].items():
                    if isinstance(x, dict) and "p" in x:
                        out.append((pid, o["output"], rule, x))
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    rej = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= ALPHA / (len(ps) - rank):
            rej[i] = True
        else:
            break
    return rej


def recheck(cs):
    affected = [c for c in cs if c[3].get("holm_reject") and c[3]["verdict"] == "NOT_CONFIRMED"]
    ps = [1.0 if (x.get("holm_reject") and x["verdict"] == "NOT_CONFIRMED") else x["p"] for *_, x in cs]
    rej = holm(ps)
    old = [str(x.get("final_verdict", "")).startswith("DETECTED") for *_, x in cs]
    new = [rej[i] and cs[i][3]["verdict"].startswith("DETECTED") for i in range(len(cs))]
    detected_p = [cs[i][3]["p"] for i in range(len(cs)) if old[i]]
    return {"tests": len(cs), "straddling_tests_counted_as_holm_rejections": len(affected),
            "final_detections_submitted": sum(old), "final_detections_recomputed": sum(new),
            "lost": [list(cs[i][:3]) for i in range(len(cs)) if old[i] and not new[i]],
            "gained": [list(cs[i][:3]) for i in range(len(cs)) if new[i] and not old[i]],
            "largest_p_among_detections": max(detected_p) if detected_p else None,
            "strictest_holm_threshold": ALPHA / len(cs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    p1 = json.loads((RES / "phase1_report.json").read_text())
    rep = json.loads((RES / "replication_96_191" / "phase1_report.json").read_text())
    p2 = json.loads((RES / "phase2" / "phase2_report.json").read_text())
    out = {"definition": __doc__.strip().split("\n\n")[1].replace("\n", " "),
           "phase1_seeds_0_95": recheck(cells(p1)), "phase1_seeds_96_191": recheck(cells(rep)),
           "phase2_published_scalars": recheck(cells(p2, "given")), "phase2_float32_scalars": recheck(cells(p2, "fp32"))}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {key: v[key] for key in ("tests", "straddling_tests_counted_as_holm_rejections",
                                              "final_detections_submitted", "final_detections_recomputed")},
                  "lost", v["lost"], "gained", v["gained"])


if __name__ == "__main__":
    main()
