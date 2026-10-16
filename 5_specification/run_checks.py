#!/usr/bin/env python3
"""Run every check of the specification folder; exit 1 if any required check fails.

Each check is its own process (no shell pipes), so a failing test cannot hide behind a later command (review finding
F2 of 2026-10-15, item 5 of 2026-10-16). Scripts that write logs run in a temporary directory and the regenerated logs
are compared with the committed ones; nothing in the repository is written.

Needs Python >= 3.10 with numpy, scipy, sympy, mpmath and pyyaml (on this machine: /data1/tzh/envs/liger/bin/python).

    python run_checks.py            # required checks
    python run_checks.py --ort      # also the ONNX Runtime cross-check (optional; needs onnxruntime)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
SPEC = HERE / "specification_v2"
ALG = SPEC / "algorithm_specs"
REVIEW = HERE / "reviews/2026-10-15/evidence/handoff"


def run(argv, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    t = time.time()
    r = subprocess.run([sys.executable, *map(str, argv)], cwd=cwd, env=env, capture_output=True, text=True,
                       timeout=3600)
    return r.returncode, r.stdout + r.stderr, time.time() - t


def tests_passed(out):
    """'Ran 9 tests ... OK' (unittest) or '105/105 checks passed' (the phase test scripts)."""
    m = re.search(r"Ran (\d+) tests?.*?\n\s*OK", out, re.S)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)/(\d+) checks passed", out)
    return int(m.group(1)) if m and m.group(1) == m.group(2) else None


def same_json(a, b, drop=()):
    x, y = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    for k in drop:
        x.pop(k, None), y.pop(k, None)
    return x == y


@dataclass
class Check:
    group: str
    name: str
    argv: list
    cwd: Path
    accept: Callable = lambda rc, out, tmp: (rc == 0, "")     # -> (ok, note)
    required: bool = True


def execute(checks, tmp):
    """Run the checks one by one; returns (rows, failed). A required check that exits non-zero, crashes or whose
    acceptance test fails makes the whole run fail."""
    rows, failed = [], False
    for c in checks:
        try:
            rc, out, s = run(c.argv, c.cwd)
            ok, note = c.accept(rc, out, tmp)
        except Exception as e:  # noqa: BLE001
            ok, note, s = False, f"{type(e).__name__}: {e}", 0.0
        if not ok and not note:
            note = f"exit {rc}: " + out.strip()[-300:]
        failed |= c.required and not ok
        state = "通过" if ok else ("失败" if c.required else "未通过（可选）")
        rows.append((c.group, c.name, state, f"{s:.1f}s", note))
        print(f"[{state}] {c.group} / {c.name} ({s:.1f}s) {note}", flush=True)
    return rows, failed


def counted(expected_total=None):
    def accept(rc, out, tmp):
        n = tests_passed(out)
        ok = rc == 0 and n is not None and (expected_total is None or n == expected_total)
        return ok, f"{n} 项" if ok else ""
    return accept


def regenerated(made, committed):
    def accept(rc, out, tmp):
        ok = rc == 0 and (tmp / made).is_file() and same_json(tmp / made, committed)
        return ok, "与提交的记录一致" if ok else ""
    return accept


def required_checks(tmp):
    return [
        # 1 executable references v2.4: 9 + 4 + 105 + 89 + 15 = 222
        Check("参照测试", "关系实例", [SPEC / "test_relations.py"], SPEC, counted(9)),
        Check("参照测试", "舍入 oracle", [SPEC / "test_oracle.py"], SPEC, counted(4)),
        Check("参照测试", "phase1", [ALG / "phase1/test_specs.py"], ALG / "phase1", counted(105)),
        Check("参照测试", "phase2", [ALG / "phase2/test_specs_phase2.py"], ALG / "phase2", counted(89)),
        Check("参照测试", "phase3", [ALG / "phase3/test_specs_phase3.py"], ALG / "phase3", counted(15)),
        # 2 derived logs: regenerated in tmp and compared with the committed copies
        Check("派生记录复现", "70 条关系检查", [SPEC / "derive.py"], tmp,
              regenerated("derivation_log.json", SPEC / "derivation_log.json")),
        Check("派生记录复现", "角色目录与计数", [SPEC / "catalogue.py"], tmp,
              regenerated("catalogue.json", SPEC / "catalogue.json")),
        Check("派生记录复现", "26 条关系 × 200 组随机验证", [SPEC / "fuzz_relations.py"], tmp,
              regenerated("fuzz_log.json", SPEC / "fuzz_log.json")),
        Check("派生记录复现", "phase1 独立复算（1000 组）", [ALG / "phase1/independent_check.py", "1000"], tmp,
              regenerated("independent_check_results.json", ALG / "phase1/independent_check_results.json")),
        # 3 contracts: the single checker, its failure-injection tests, the reviewer's witnesses
        Check("合同", "合同检查（结构、例子、读法、变异）", [HERE / "tools/check_contracts.py"], HERE),
        Check("合同", "检查器与注册入口的测试（含注入失败 A–I）", ["-m", "unittest", "discover", "-s", "tests"], HERE,
              counted()),
        Check("合同", "审阅方的例子对照（22 例、58 个标量）",
              [HERE / "tools/check_witnesses.py", "--source-root", HERE, "--output", tmp / "witness.json"], HERE,
              lambda rc, out, t: ((ok := rc == 0 and same_json(t / "witness.json", REVIEW / "reviewed_witness_checks.json")),
                                  "与审阅方的结果一致" if ok else "")),
        # 4 evidence (does not decide clause status)
        Check("证据", "v1 误读重建（SGD、BN）", [HERE / "evidence/historical_misreadings/meaning_relations.py"], HERE,
              lambda rc, out, t: ((ok := rc == 0 and out.count("relation holds: True") == 2
                                   and out.count("relation holds: False") == 2), "正确版本满足关系，v1 版本被抓到" if ok else "")),
        Check("证据", "THEORY 撤回部分的核对", [HERE / "evidence/theory_checks/checks.py"], HERE),
        Check("证据", "Liger CE 声明公式对照", [HERE / "evidence/declared_semantics/liger_ce_crosscheck.py"], HERE),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ort", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        checks = required_checks(tmp)
        if a.ort:
            checks.append(Check("证据", "ONNX Runtime 对照", [HERE / "evidence/ort_witness/battery.py"], HERE,
                                required=False))
        rows, failed = execute(checks, tmp)
    print("全部必需检查通过" if not failed else "有必需检查失败")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
