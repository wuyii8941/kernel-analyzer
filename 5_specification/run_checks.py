#!/usr/bin/env python3
"""Run every check of the specification folder; exit 1 if any required check fails.

Each check is its own process (no shell pipes), so a failing test cannot hide behind a later command (the defect of
the old run_all.sh, review finding F2). Scripts that write logs run in a temporary directory and the regenerated logs
are compared with the committed ones; nothing in the repository is written.

Needs Python >= 3.10 with numpy, scipy, sympy, mpmath and pyyaml (on this machine: /data1/tzh/envs/liger/bin/python).

    python run_checks.py            # required checks
    python run_checks.py --ort      # also the ONNX Runtime cross-check (needs onnxruntime; writes
                                    # evidence/ort_witness/battery_rows.json)
    python run_checks.py --draft    # also print the report of the superseded draft checker (not a check: it reports
                                    # false greens, review findings F1-F5)
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = HERE / "specification_v2"
ALG = SPEC / "algorithm_specs"
REVIEW = HERE / "reviews/2026-10-15/evidence/handoff"


def run(*argv, cwd):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    t = time.time()
    r = subprocess.run([sys.executable, *map(str, argv)], cwd=cwd, env=env, capture_output=True, text=True,
                       timeout=1800)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ort", action="store_true")
    ap.add_argument("--draft", action="store_true")
    a = ap.parse_args()
    rows, failed = [], False

    def record(group, name, ok, seconds, note, required=True):
        nonlocal failed
        state = "通过" if ok is True else "失败" if ok is False else "未运行"
        failed |= required and ok is not True
        rows.append((group, name, state, f"{seconds:.1f}s", note))
        print(f"[{state}] {group} / {name} ({seconds:.1f}s) {note}", flush=True)

    # 1 executable references v2.4: 9 + 4 + 105 + 89 + 15 = 222
    total = 0
    for name, script, cwd in [("关系实例", SPEC / "test_relations.py", SPEC),
                              ("舍入 oracle", SPEC / "test_oracle.py", SPEC),
                              ("phase1", ALG / "phase1/test_specs.py", ALG / "phase1"),
                              ("phase2", ALG / "phase2/test_specs_phase2.py", ALG / "phase2"),
                              ("phase3", ALG / "phase3/test_specs_phase3.py", ALG / "phase3")]:
        rc, out, s = run(script, cwd=cwd)
        n = tests_passed(out)
        total += n or 0
        record("参照测试", name, rc == 0 and n is not None, s, f"{n} 项" if n else out.strip()[-300:])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 2 derived logs: regenerate in tmp, compare with the committed copies
        for name, script, args, made, committed in [
                ("70 条关系检查", SPEC / "derive.py", (), "derivation_log.json", SPEC / "derivation_log.json"),
                ("角色目录与计数", SPEC / "catalogue.py", (), "catalogue.json", SPEC / "catalogue.json"),
                ("26 条关系 × 200 组随机验证", SPEC / "fuzz_relations.py", (), "fuzz_log.json", SPEC / "fuzz_log.json"),
                ("phase1 独立复算（1000 组）", ALG / "phase1/independent_check.py", ("1000",),
                 "independent_check_results.json", ALG / "phase1/independent_check_results.json")]:
            rc, out, s = run(script, *args, cwd=tmp)       # catalogue.py reads the derivation_log.json made just before
            ok = rc == 0 and (tmp / made).is_file() and same_json(tmp / made, committed)
            record("派生记录复现", name, ok, s, "与提交的记录一致" if ok else out.strip()[-300:])

        # 3 reviewed contracts (structure, validator regressions, witnesses against the references)
        rc, out, s = run(HERE / "tools/validate_contracts.py", cwd=HERE)
        record("合同", "结构与引用校验", rc == 0, s, "3 份合同、15 条条款、29 个例子" if rc == 0 else out[-300:])
        rc, out, s = run("-m", "unittest", "discover", "-s", "tests", cwd=HERE)
        n = tests_passed(out)
        record("合同", "校验器回归", rc == 0 and n is not None, s, f"{n} 项" if n else out[-300:])
        rc, out, s = run(HERE / "tools/check_witnesses.py", "--source-root", str(HERE), "--output",
                         str(tmp / "witness.json"), cwd=HERE)
        ok = rc == 0 and same_json(tmp / "witness.json", REVIEW / "reviewed_witness_checks.json")
        record("合同", "例子对照参照（22 例、58 个标量）", ok, s,
               "与审阅方的结果一致；7 个非法输入 / 未支持例待适配器实现" if ok else out[-300:])
        rc, out, s = run(HERE / "tools/validate_contracts.py", "--require-runtime-ready", cwd=HERE)
        record("合同", "未启用（准入门槛应返回 2）", rc == 2, s, "没有任何合同被自动启用" if rc == 2 else f"返回 {rc}")

        # 4 evidence (does not decide clause status)
        rc, out, s = run(HERE / "evidence/historical_misreadings/meaning_relations.py", cwd=HERE)
        ok = rc == 0 and out.count("relation holds: True") == 2 and out.count("relation holds: False") == 2
        record("证据", "v1 误读重建（SGD、BN）", ok, s, "正确版本满足关系，v1 版本被抓到" if ok else out[-300:])
        rc, out, s = run(HERE / "evidence/theory_checks/checks.py", cwd=HERE)
        record("证据", "THEORY 撤回部分的核对", rc == 0, s, "" if rc == 0 else out[-300:])
        rc, out, s = run(HERE / "evidence/declared_semantics/liger_ce_crosscheck.py", cwd=HERE)
        record("证据", "Liger CE 声明公式对照", rc == 0, s, "" if rc == 0 else out[-300:])
        try:
            import onnxruntime  # noqa: F401
            have_ort = True
        except ImportError:
            have_ort = False
        if a.ort and have_ort:
            rc, out, s = run(HERE / "evidence/ort_witness/battery.py", cwd=HERE)
            record("证据", "ONNX Runtime 对照", rc == 0, s, out.strip()[-200:], required=False)
        else:
            record("证据", "ONNX Runtime 对照", None, 0.0,
                   "未安装 onnxruntime" if not have_ort else "需要 --ort", required=False)

    if a.draft:
        rc, out, s = run(HERE / "contracts/draft/contract_check.py", cwd=HERE)
        print("\n---- 草案检查器的报告（不作为检查：会给出假绿，见审阅 F1–F5）----\n" + out[-1500:])

    print(f"\n参照测试合计 {total} 项（应为 222）")
    print("全部必需检查通过" if not failed else "有必需检查失败")
    return 1 if failed or total != 222 else 0


if __name__ == "__main__":
    sys.exit(main())
