#!/usr/bin/env python3
"""Recount the blind-test summary (results/reference_eval/blind_test_records/blind_records/errata.md) from the files:

1. independent recomputation scale: output elements, projection values and interval violations from the setter's
   three verification reports;
2. known positives: our verdict matrices frozen before unsealing (blind_test_v1/phase1_report_v1.1.json,
   blind_test_v2/phase1_verdict_matrix_v2.csv) joined with the rule and expected sign of each positive as listed in the
   setter's final scoring (v1/v2 *_final_scoring.md, section "已知阳性").  A positive counts as found when the listed
   rule is detected with the expected sign in both runs (seeds 0-95 and 96-191).

    python scripts/blind_records_recount.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "results/reference_eval/blind_test_records/blind_records"

# rule and expected sign of the detection, transcribed from the final scoring tables (section 2 / 3)
POSITIVES_V1 = {"prog_21": ("R3", "-"), "prog_26": ("R1", "+"), "prog_06": ("R1", "+"), "prog_14": ("R2", "+"),
                "prog_05（rstd）": ("R1", "-"), "prog_16": ("R4", "+"), "prog_31": ("R3", "+"), "prog_32": ("R3", "+")}
POSITIVES_V2 = {"prog_11": ("R1", "+"), "prog_17": ("R3", "-"), "prog_24": ("R3", "+"), "prog_27": ("R1", "-"),
                "prog_30": ("R3", "-"), "prog_06": ("R1", "+"), "prog_36": ("R3", "+"), "prog_13": ("R3", "-")}
PROBE_V2 = {"prog_08"}  # declared a sensitivity probe before unsealing (errata 6)


def scale():
    out = {}
    v1a = json.loads((REC / "v1/blind_test_v1_verification_report.json").read_text())
    n1 = sum(e["n"] for e in v1a.values())
    p1 = sum(len(m) for e in v1a.values() for m in e.get("projections", {}).values())
    viol1 = sum(e.get("tool_interval_violations", 0) for e in v1a.values())
    v1b = json.loads((REC / "v1/blind_test_v1_phase2_verification_report.json").read_text())
    n2 = sum(e["n"] for e in v1b.values())
    proj2 = [r for e in v1b.values() for m in e["modes"].values() for o in m.get("projections", {}).values() for r in o.values()]
    p2 = len(proj2)
    viol2 = sum(e.get("KR_tool_interval_violations", 0) + sum(m.get("f_tool_interval_violations", 0)
                                                               + m.get("e_sem_tool_interval_violations", 0)
                                                               for m in e["modes"].values()) for e in v1b.values())
    viol2 += sum(1 for r in proj2 if not (r.get("inside_directed", True) and r.get("inside_pipeline", True)))
    v2 = json.loads((REC / "v2/blind_test_v2_verification_report.json").read_text())
    n3 = sum(e["n"] for e in v2["entries"].values())
    p3 = len(v2["projection_checks"])
    viol3 = sum(e.get("KR_violations", 0) + e.get("f_violations", 0) + sum(u.get("violations", 0) for u in e.get("u", {}).values())
                for e in v2["entries"].values()) + sum(1 for c in v2["projection_checks"] if not c[-1])
    out["elements"] = {"v1_phase1": n1, "v1_phase2": n2, "v2": n3, "total": n1 + n2 + n3}
    out["projections"] = {"v1_phase1": p1, "v1_phase2": p2, "v2": p3, "total": p1 + p2 + p3}
    out["violations"] = viol1 + viol2 + viol3
    return out


def found_v1():
    d = json.loads((ROOT / "results/reference_eval/blind_test_v1/phase1_report_v1.1.json").read_text())
    res = {}
    for prog, (rule, sign) in POSITIVES_V1.items():
        cell = d[prog]["cells"][rule]
        want = "DETECTED_POSITIVE" if sign == "+" else "DETECTED_NEGATIVE"
        res[prog] = all(cell[k]["verdict"] == want for k in ("seeds_0_95", "seeds_96_191"))
    return res


def found_v2():
    rows = {r["program"]: r for r in csv.DictReader(open(ROOT / "results/reference_eval/blind_test_v2/phase1_verdict_matrix_v2.csv"))}
    res = {}
    for prog, (rule, sign) in POSITIVES_V2.items():
        want = "DETECTED_POSITIVE" if sign == "+" else "DETECTED_NEGATIVE"
        r = rows[prog]
        res[prog] = all(r[f"{rule}_verdict_seeds_{s}"] == want for s in ("0_95", "96_191"))
    probe = {p: any(rows[p][c].startswith("DETECTED") for c in rows[p] if c.endswith(("seeds_0_95", "seeds_96_191")) and "verdict" in c)
             for p in PROBE_V2}
    return res, probe


def main():
    s = scale()
    v1 = found_v1()
    v2, probe = found_v2()
    report = {"scale": s, "positives_v1": v1, "positives_v2": v2, "probe_v2_detected": probe,
              "summary": {"v1": f"{sum(v1.values())}/{len(v1)}", "v2": f"{sum(v2.values())}/{len(v2)}",
                          "all_including_probe": f"{sum(v1.values()) + sum(v2.values())}/{len(v1) + len(v2) + len(probe)}"}}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    (REC.parent / "recount.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
