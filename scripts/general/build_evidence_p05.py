#!/usr/bin/env python3
"""Evidence item P05 (task book section 7): the blind-test v1 / v2 scoring files and the calibration records, as the
existing evidence that "the reference is reliable" and "the detection is reliable".  Writes
results/general/evidence_P05.json (path, sha256, bytes, what it supports) and docs/general/evidence_P05.md."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BR = "results/reference_eval/blind_test_records/blind_records"
ITEMS = [
    # (path, supports, note)
    (f"{BR}/v1/blind_test_v1_final_scoring.md", "reference + detection", "blind test v1 final scoring"),
    (f"{BR}/v1/blind_test_v1_verification_report.json", "reference", "independent recomputation of K_R (v1)"),
    (f"{BR}/v1/blind_test_v1_phase2_verification_report.json", "reference + detection", "v1 phase 2 verification"),
    (f"{BR}/v1/blind_test_v1_answer_key_UNSEALED.json", "detection", "v1 answer key (unsealed after scoring)"),
    (f"{BR}/v2/blind_test_v2_final_scoring.md", "reference + detection", "blind test v2 final scoring"),
    (f"{BR}/v2/blind_test_v2_verification_report.json", "reference", "independent recomputation of K_R (v2)"),
    (f"{BR}/v2/blind_test_v2_answer_key_UNSEALED.json", "detection", "v2 answer key"),
    (f"{BR}/v2/blind_test_v2_answer_key_update_layer_UNSEALED.json", "detection", "v2 update-layer answer key"),
    (f"{BR}/SHA256SUMS", "integrity", "checksums of the blind records (verified: sha256sum -c)"),
    (f"{BR}/errata.md", "integrity", "errata of the blind records"),
    ("results/reference_eval/blind_test_records/recount.json", "reference + detection",
     "recount from the frozen verdict matrix: 16,908,160 elements and 974 projections, 0 violations"),
    ("results/reference_eval/blind_test_v1/coverage_check.json", "reference", "v1: every program has a complete reference"),
    ("results/reference_eval/blind_test_v2/coverage_check.json", "reference", "v2: every program has a complete reference"),
    ("results/reference_eval/calibration_equivalence.json", "detection", "t / bootstrap-t coverage, TOST (4000 per cell)"),
    ("results/reference_eval/detector_calibration_v2.json", "detection", "default detector v2 false-positive calibration"),
    ("results/reference_eval/decision_rule_calibration.json", "detection", "direction-rule false-positive and family-wise rates"),
    ("docs/statistics_calibration_20261006.md", "detection", "calibration report (source of S0, N0, n_min)"),
    ("docs/tool_validation.md", "reference", "bit-exact emulation and counterfactual checks"),
]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


rows = []
for path, supports, note in ITEMS:
    p = ROOT / path
    rows.append({"path": path, "exists": p.exists(), "sha256": sha(path) if p.exists() else None,
                 "bytes": p.stat().st_size if p.exists() else None, "supports": supports, "note": note})
chk = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=ROOT / BR, capture_output=True, text=True)
out = {"item": "P05", "statement": "blind tests v1 / v2 scoring files and calibration records; existing evidence that the "
       "reference is reliable and the detection is reliable (frozen; not recomputed in this round)",
       "blind_records_checksums_ok": chk.returncode == 0, "files": rows}
(ROOT / "results/general/evidence_P05.json").write_text(json.dumps(out, indent=1) + "\n")
lines = ["# 证据 P05：盲测 v1 / v2 计分与校准记录（通用能力轮，2026-10-08）", "",
         "作为「参照可靠」「检测可靠」的既有证据入库索引（文件本身早已入库，冻结不改）。盲测记录的 SHA256SUMS 校验："
         + ("通过" if out["blind_records_checksums_ok"] else "**失败**") + "。", "",
         "| 文件 | 支撑 | 说明 | sha256（前 16 位） |", "|---|---|---|---|"]
lines += [f"| `{r['path']}` | {r['supports']} | {r['note']} | `{(r['sha256'] or 'missing')[:16]}` |" for r in rows]
(ROOT / "docs/general/evidence_P05.md").write_text("\n".join(lines) + "\n")
print("P05", sum(r["exists"] for r in rows), "/", len(rows), "files; checksums ok:", out["blind_records_checksums_ok"])
