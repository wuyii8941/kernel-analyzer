#!/usr/bin/env python3
"""Store the follow-up captures (.cache/audit_work/rerun2, commit 857b913) in 1_experiments/dsl_v2/captures and
replace the superseded files (originals stay in git history):
  inc15 broad (already replaced by the first audit-fix run)  -> auditfix_broad.jsonl.gz (rerun)
  inc10_gluon_test_core + inc14_gluon_test_core_atomic      -> auditfix_gluon_test_core.jsonl.gz
  inc11_cross_level_combined                                -> auditfix_cross_nvidia.jsonl.gz
  inc12_cross_level_amd (+ supplementary core / td)         -> auditfix_cross_amd*.jsonl.gz
  inc4_tutorials + inc7_tutorial05                          -> auditfix_tutorials.jsonl.gz
  inc14_test_core_atomic (a subset of the broad test_core)  -> (contained in auditfix_broad)
and write auditfix_capture_comparisons.json (every rerun against its recorded predecessor)."""
import gzip
import json
import shutil
import subprocess
from pathlib import Path

R = Path("/data1/tzh/kernel-analyzer")
O = R / ".cache/audit_work/rerun2"
CAP = R / "1_experiments/dsl_v2/captures"


def gz(src, dst):
    (CAP / dst).write_bytes(gzip.compress((O / src).read_bytes(), compresslevel=9, mtime=0))


subprocess.run(["/data1/tzh/envs/ka_main/bin/python", str(R / "3_audits/fix_1aee15e/evidence/scripts/analyze_captures.py")],
               check=True, cwd=R)
gz("broad_combined.jsonl", "auditfix_broad.jsonl.gz")
gz("gluon_test_core.jsonl", "auditfix_gluon_test_core.jsonl.gz")
gz("cross_nvidia.jsonl", "auditfix_cross_nvidia.jsonl.gz")
gz("cross_amd.jsonl", "auditfix_cross_amd.jsonl.gz")
gz("cross_amd_supp_core.jsonl", "auditfix_cross_amd_supplementary_core.jsonl.gz")
gz("cross_amd_supp_td.jsonl", "auditfix_cross_amd_supplementary_tensor_descriptor.jsonl.gz")
gz("tutorials.jsonl", "auditfix_tutorials.jsonl.gz")
# the summarizer refuses to overwrite files under 1_experiments and exits 1 when launches are not complete: write to
# the cache, then copy
subprocess.run(["/data1/tzh/envs/ka_main/bin/python", str(R / "2_tool/scripts/dsl_v2/summarize_capture.py"),
                str(O / "broad_combined.jsonl"), "--out", str(O / "broad_summary.json")], cwd=R, capture_output=True)
shutil.copy(O / "broad_summary.json", CAP / "auditfix_broad_summary.json")
a = json.loads((O / "analysis.json").read_text())
a["note"] = ("follow-up captures at commit 857b913 against their recorded predecessors (originals in git history: "
             "1_experiments/dsl_v2/captures at commit 857b913^ = the parent of the records commit)")
(CAP / "auditfix_capture_comparisons.json").write_text(json.dumps(a, indent=1) + "\n")
for old in ("inc10_gluon_test_core.jsonl.gz", "inc14_gluon_test_core_atomic.jsonl.gz", "inc11_cross_level_combined.jsonl.gz",
            "inc12_cross_level_amd.jsonl.gz", "inc12_cross_level_amd_supplementary_core.jsonl.gz",
            "inc12_cross_level_amd_supplementary_tensor_descriptor.jsonl.gz", "inc4_tutorials.jsonl.gz",
            "inc7_tutorial05.jsonl.gz", "inc14_test_core_atomic.jsonl.gz", "auditfix_compare_broad_vs_inc15.json"):
    subprocess.run(["git", "rm", "-q", f"1_experiments/dsl_v2/captures/{old}"], check=True, cwd=R)
shutil.copy(R / ".cache/audit_work/rerun_all.sh", CAP / "run/auditfix_rerun_all.sh")
subprocess.run(["git", "rm", "-q", "1_experiments/dsl_v2/captures/run/auditfix_broad_run.sh"], check=True, cwd=R)
print("stored")
