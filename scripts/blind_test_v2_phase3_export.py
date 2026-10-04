#!/usr/bin/env python3
"""Verification material for blind_test_v2 (requested with the phase 3 submission): seeds 0 and 32.

Per family and seed: the inputs, the specification f's enclosure, theta_s and the AdamW history state (the common
state of phase 3).  Per program and seed: K (the kernel output), the K_R interval, and u for the three
optimizers in the scoring definition (ideal response: the real optimizer in interval arithmetic, u = step(K) -
step(K_R)); u is recomputed with the same function as the phase 3 run (deterministic).

    python scripts/blind_test_v2_phase3_export.py --package .cache/blind/blind_test_v2 --date 20261004
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import blind_test_v2_phase3 as p3  # noqa: E402

SEEDS = (0, 32)
CACHE = ROOT / ".cache"

README = """# blind_test_v2 核实材料（seed 0 与 32，{date}）

被测方：kernel-analyzer（提交 `{head}`）。数组均按输出的行优先展平顺序（与 `output_flat_index.npy` 一致时即 0..n−1）。

| 路径 | 内容 |
|---|---|
| `family/<家族>/seed{{000,032}}/inputs.npz`、`scalars.json` | `make_inputs(family, seed)` 的输入张量与运行时标量 |
| `family/<家族>/seed{{000,032}}/f_lo.npy`、`f_hi.npy` | 规格 f 的严格包围（阶段 2） |
| `family/<家族>/seed{{000,032}}/theta.npy` | θ_s（float32，seed 5000 + s） |
| `family/<家族>/seed{{000,032}}/adamw_history_state.npz` | AdamW 历史状态：exp_avg、exp_avg_sq（float32）、step = 16；由 RN32(f) 在历史 seed 上经同一 FP32 AdamW 16 步得到 |
| `programs/<程序>/seed{{000,032}}/K.npy` | kernel 输出（float32） |
| `programs/<程序>/seed{{000,032}}/KR_lo.npy`、`KR_hi.npy` | 工具的 K_R 区间（float64） |
| `programs/<程序>/seed{{000,032}}/u_<optimizer>_lo.npy`、`_hi.npy` | 计分口径的 u（理想响应）：sgd、adamw_history（t = 17）、adamw_zero（t = 1），lr = 2⁻¹⁰，β = (0.9, 0.999)，ε = 2⁻²⁷ |

u = step(K) − step(K_R)：optimizer 的实数语义在区间算术中对 K（点）与 K_R（区间）各求一步（`update_layer.py`），θ 在差中抵消；
状态 (m, v) 取上表的 float32 值作为精确给定值。seed 32 是 0–95 一轮的确认 seed，它在每条规则下的投影上下界见阶段 3 记录
（`phase3_report.json` 中各程序、各 optimizer、各口径的 `record.rules[*].per_unit_bounds`，确认 seed 按 32..95 排列，seed 32 为第一个）。
附加口径（实际 FP32 写入差）可由 θ、状态、K 与 K_R 用 `scripts/blind_test_v2_phase3.py` 重算，本包不重复存放。
"""


def main():
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    import torch

    sys.path.insert(0, str(args.package / "programs"))
    inputs = importlib.import_module("inputs")
    manifest = json.loads((args.package / "manifest.json").read_text())
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    target = CACHE / f"blind_test_v2_verification_{args.date}"
    if target.exists():
        shutil.rmtree(target)
    for fam in manifest["families"]:
        for seed in SEEDS:
            d = target / "family" / fam / f"seed{seed:03d}"
            d.mkdir(parents=True)
            inp = inputs.make_inputs(fam, seed, device="cpu")
            np.savez_compressed(d / "inputs.npz", **{k: v.numpy() for k, v in inp.items() if torch.is_tensor(v)})
            (d / "scalars.json").write_text(json.dumps({k: v for k, v in inp.items() if not torch.is_tensor(v)}))
            f = np.load(CACHE / "blind_v2_f" / f"{fam}_{seed:03d}.npz")
            np.save(d / "f_lo.npy", f["lo"])
            np.save(d / "f_hi.npy", f["hi"])
            st = np.load(p3.STATE / f"{fam}_{seed:03d}.npz")
            np.save(d / "theta.npy", p3.theta_for(seed, f["lo"].size))
            np.savez(d / "adamw_history_state.npz", exp_avg=st["exp_avg"], exp_avg_sq=st["exp_avg_sq"], step=st["step"])
    for e in manifest["programs"]:
        pid, fam = e["id"], e["family"]
        for seed in SEEDS:
            d = target / "programs" / pid / f"seed{seed:03d}"
            d.mkdir(parents=True)
            c = np.load(p3.ARRAYS / pid / f"seed{seed:03d}.npz")
            order = np.argsort(c["index"])
            np.save(d / "K.npy", c["k"][order])
            np.save(d / "KR_lo.npy", c["kr_lo"][order])
            np.save(d / "KR_hi.npy", c["kr_hi"][order])
            np.save(d / "output_flat_index.npy", c["index"][order])
            theta = p3.theta_for(seed, c["k"].size)
            for opt in p3.OPTIMIZERS:
                _, (i_lo, i_hi, _, _) = p3.measure_seed(opt, fam, seed, c, theta, None)
                np.save(d / f"u_{opt}_lo.npy", i_lo)
                np.save(d / f"u_{opt}_hi.npy", i_hi)
        print(pid, "exported", flush=True)
    (target / "README.md").write_text(README.format(date=args.date, head=head))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}"
             for p in sorted(target.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    out = Path(str(target) + ".tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(target, arcname=target.name)
    print(out, len(lines), "files", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
