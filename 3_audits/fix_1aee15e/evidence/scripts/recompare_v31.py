"""Recompute the v3.1 comparison field of each job of a regression run from the job's own outputs, with the
comparison of regression_v11.py as it is now (fields v3.1 wrote, recursively).  The jobs were written by the same
script before this refinement of the comparison; nothing else in a job changes."""
import json, sys
from pathlib import Path
R = Path("/data1/tzh/kernel-analyzer")
sys.path[:0] = [str(R / "2_tool/scripts/dsl_v2"), str(R / "2_tool/src"), str(R / "2_tool/scripts/acceptance"), str(R / "2_tool/scripts/general")]
import regression_v11 as RV
d = R / "1_experiments/dsl_v2/regression_v11" / sys.argv[1]
for p in sorted(d.glob("prog_*.json")):
    job = json.loads(p.read_text())
    if job.get("status") != "ok":
        continue
    job["v31_comparison"] = RV.compare_v31(job["program"], job["outputs"])
    job["v31_comparison_note"] = ("recomputed after the run from these outputs: fields v3.1 wrote, recursively "
                                  "(later additions are not differences)")
    p.write_text(json.dumps(job, indent=1, default=str) + "\n")
    print(p.stem, {n: (r.get("reference_equal"), r.get("e_num_record_equal")) for n, r in job["v31_comparison"].items()})
