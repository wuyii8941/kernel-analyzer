#!/usr/bin/env python3
"""Ledger 3a (audit M01): every commit of dsl-v2 after main (4156422) up to the audited 1aee15e, with its full SHA,
parent, changed files, the tests it touched, the result records it wrote, and the code of the seven findings it
touched (functions found in the diff hunks).  Run from the repository root: python 3_audits/fix_1aee15e/commit_table.py
Writes commit_table.json and commit_table.md next to this script."""
import collections
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE, HEAD = "4156422", "1aee15e"
FINDINGS = {
    "F01": {"_int_op", "_cmpi", "_unsigned", "_wrap", "_op_atomic_cas", "_int_to_float"},
    "F02": {"_scaled_dot", "_op_dot_scaled", "_scaled_upcast", "_op_scaled_upcast_fp8", "_op_scaled_upcast_fp4",
            "_op_tc_gen5_mma_scaled"},
    "F03": {"_cas_reverse_order", "_check_scan_bracketing", "_generic_scan", "element_classes"},
    "F04": {"torch_intermediates", "run_level"},
    "F05": {"run", "evaluate_sequence", "_float_atomics"},
    "F06": {"expand", "make_inputs"},
    "F07": {"_execution_status", "_within_input_mean_residual"},
}
FILES = {"F01": "ttir_eval.py", "F02": "ttir_eval.py", "F03": "ttir_eval.py", "F04": ("check.py", "measure.py"),
         "F05": ("check.py", "ttir_eval.py"), "F06": "measure.py", "F07": "check.py"}


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout


def touched_functions(sha):
    """(file, function) pairs from the diff hunk headers and the def lines added or removed."""
    out = set()
    cur = None
    for line in git("show", "-U0", "--format=", sha, "--", "*.py").splitlines():
        if line.startswith("+++ "):
            cur = line[6:] if line.startswith("+++ b/") else None
            continue
        if cur is None:
            continue
        m = re.match(r"@@ [^@]* @@\s*(?:async\s+)?(?:def|class)\s+(\w+)", line)
        if m:
            out.add((cur, m.group(1)))
        m = re.match(r"[+-]\s*(?:async\s+)?def\s+(\w+)\s*\(", line)
        if m:
            out.add((cur, m.group(1)))
    return out


rows = []
for sha in git("rev-list", "--reverse", f"{BASE}..{HEAD}").split():
    parents = git("show", "-s", "--format=%P", sha).split()
    subject = git("show", "-s", "--format=%s", sha).strip()
    date = git("show", "-s", "--format=%ad", "--date=short", sha).strip()
    files = [l for l in git("show", "--format=", "--name-status", sha).splitlines() if l.strip()]
    paths = [l.split("\t")[-1] for l in files]
    groups = collections.Counter(p.split("/")[0] if "/" in p else "(root)" for p in paths)
    funcs = touched_functions(sha)
    hits = {}
    for f, names in FINDINGS.items():
        want = FILES[f] if isinstance(FILES[f], tuple) else (FILES[f],)
        found = sorted({fn for (path, fn) in funcs if fn in names and path.endswith(want)})
        if found:
            hits[f] = found
    rows.append({"sha": git("rev-parse", sha).strip(), "parents": parents, "date": date, "subject": subject,
                 "files_changed": len(paths), "by_top_level_dir": dict(groups),
                 "tests_touched": sorted(p for p in paths if re.search(r"(^|/)tests?/test_[^/]*\.py$", p)),
                 "result_records": sorted(p for p in paths if p.startswith(("results/", "docs/dsl_v2/")))[:12],
                 "finding_code_touched": hits})
(HERE / "commit_table.json").write_text(json.dumps({"base": git("rev-parse", BASE).strip(),
                                                    "head": git("rev-parse", HEAD).strip(),
                                                    "count": len(rows), "commits": rows}, indent=1) + "\n")
md = [f"# 提交逐项表（{BASE}..{HEAD}，{len(rows)} 个提交）", "",
      "由 `commit_table.py` 从 git 生成；文件路径是提交当时（整理前）的路径，当前位置见 `CURRENT.json`。"
      "「发现相关代码」按 diff 块所在函数匹配七项发现涉及的函数，是定位线索，不是逐行审阅结论。", "",
      "| # | 提交 | 日期 | 说明 | 改动文件 | 测试文件 | 发现相关代码 |", "| --- | --- | --- | --- | --- | --- | --- |"]
for i, r in enumerate(rows, 1):
    hits = "; ".join(f"{f}: {', '.join(v)}" for f, v in r["finding_code_touched"].items()) or "—"
    md.append(f"| {i} | `{r['sha'][:10]}` | {r['date']} | {r['subject'].replace('|', '/')} | {r['files_changed']} "
              f"({', '.join(f'{k} {v}' for k, v in sorted(r['by_top_level_dir'].items()))}) | {len(r['tests_touched'])} | "
              f"{hits} |")
(HERE / "commit_table.md").write_text("\n".join(md) + "\n")
print(len(rows), "commits")
