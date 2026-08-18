#!/usr/bin/env python3
"""Run strict complete-coordinate coherent-carrier T3 for four RMSNorm survivors."""

from __future__ import annotations

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/tzh/miniconda3/envs/pt_nightly/bin/python")

CASES = (
    ("forward:16:in_out_ptr0", "model.layers.0.self_attn.q_proj.weight", "rsqrt3_l0_qproj"),
    ("forward:21:in_out_ptr1", "model.layers.0.mlp.gate_proj.weight", "rsqrt4_l0_gateproj"),
    ("forward:34:in_out_ptr1", "model.layers.1.self_attn.q_proj.weight", "rsqrt7_l1_qproj"),
    ("forward:39:in_out_ptr1", "model.layers.1.mlp.gate_proj.weight", "rsqrt8_l1_gateproj"),
)


def main() -> None:
    for task_id, carrier, stem in CASES:
        output = ROOT / "results/coverage/cases/full_coordinate" / f"qwen_seq128_{stem}_t3_gram.json.gz"
        if output.exists():
            continue
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts/run_qwen128_rsqrt13_carrier_gram.py"),
            "--device", "cuda:0", "--task-id", task_id, "--carrier", carrier,
            "--spool-dir", f"/data1/tzh/cache/kernel_analyzer_spool/qwen128_{stem}_t3",
            "--output", str(output),
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
