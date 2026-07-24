#!/usr/bin/env python
"""Compare arm D (fork-aware) against A (eager), B (compile), C (fusion repair)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.phase8_matched_step import state_distance


def main() -> None:
    root = Path("results/trajectory_step5_fusion")
    arms = {}
    for name in ["A_reference", "B_alternative", "C_fusion_repair", "D_fork_aware", "E_fork_aware_unclipped"]:
        path = root / f"{name}.json"
        if path.exists():
            arms[name] = json.loads(path.read_text(encoding="utf-8"))

    print("=== Fork-Aware Trajectory Comparison ===\n")

    A = arms["A_reference"]["trajectory"]
    B = arms["B_alternative"]["trajectory"]
    C = arms["C_fusion_repair"]["trajectory"]
    extra_arms = {}
    for key in ["D_fork_aware", "E_fork_aware_unclipped"]:
        if key in arms:
            extra_arms[key] = arms[key]["trajectory"]
    all_trajs = [A, B, C] + list(extra_arms.values())
    n = min(len(t) for t in all_trajs)

    print("Target token clip agreement with A (eager reference):")
    for label, window in [("Steps 1-5", range(5)), ("Steps 6-10", range(5, 10)), ("Steps 11-20", range(10, min(n, 20)))]:
        ab = sum(1 for i in window if A[i]["target_clip_active"] == B[i]["target_clip_active"])
        ac = sum(1 for i in window if A[i]["target_clip_active"] == C[i]["target_clip_active"])
        ad = sum(1 for i in window if A[i]["target_clip_active"] == D[i]["target_clip_active"])
        size = len(list(window))
        print(f"  {label}: A==B {ab}/{size} ({100*ab/size:.0f}%),  A==C {ac}/{size} ({100*ac/size:.0f}%),  A==D {ad}/{size} ({100*ad/size:.0f}%)")

    total_ab = sum(1 for i in range(n) if A[i]["target_clip_active"] == B[i]["target_clip_active"])
    total_ac = sum(1 for i in range(n) if A[i]["target_clip_active"] == C[i]["target_clip_active"])
    total_ad = sum(1 for i in range(n) if A[i]["target_clip_active"] == D[i]["target_clip_active"])
    print(f"  Total:      A==B {total_ab}/{n} ({100*total_ab/n:.0f}%),  A==C {total_ac}/{n} ({100*total_ac/n:.0f}%),  A==D {total_ad}/{n} ({100*total_ad/n:.0f}%)")

    # 2. Batch clip deviation
    print("\nCumulative batch-clip deviation from A:")
    for label, arm, data in [("B", "B_alternative", B), ("C", "C_fusion_repair", C), ("D", "D_fork_aware", D)]:
        dev = sum(abs(A[i]["batch_clip_active_count"] - data[i]["batch_clip_active_count"]) for i in range(n))
        print(f"  Σ|A-{label}| = {dev}")

    # 3. Fork-aware mask stats
    if "fork_aware_masked_count" in D[0]:
        print("\nFork-aware masked tokens per step:")
        for step in D:
            print(f"  Step {step['step']:2d}: {step['fork_aware_masked_count']} / 512 masked ({100*step['fork_aware_masked_count']/512:.1f}%)")

    # 4. Parameter distances at checkpoints
    print("\nParameter distances (L2):")
    print(f"{'Step':>6} | {'A-B':>12} | {'A-C':>12} | {'A-D':>12} | {'Recovery B':>10} | {'Recovery C':>10} | {'Recovery D':>10}")
    print("-" * 95)
    for step in [1, 5, 20]:
        step_dir = f"step_{step:02d}"
        ab = state_distance(root / "A_reference" / step_dir, root / "B_alternative" / step_dir)
        ac = state_distance(root / "A_reference" / step_dir, root / "C_fusion_repair" / step_dir)
        ad = state_distance(root / "A_reference" / step_dir, root / "D_fork_aware" / step_dir)
        r_b = 1.0
        r_c = ac["l2"] / ab["l2"] if ab["l2"] else None
        r_d = ad["l2"] / ab["l2"] if ab["l2"] else None
        print(f"{step:>6} | {ab['l2']:.6e} | {ac['l2']:.6e} | {ad['l2']:.6e} | {r_b:>10.4f} | {r_c:>10.4f} | {r_d:>10.4f}")

    # 5. Training stats convergence
    print("\nFinal step (step 20) training stats:")
    for label, data in [("A eager", A), ("B compile", B), ("C fusion", C), ("D fork-aware", D)]:
        last = data[min(n, 20) - 1]
        print(f"  {label:>14}: loss={last['loss']:.6f}, grad_norm={last['full_gradient_norm']:.4f}, batch_clips={last['batch_clip_active_count']}")


if __name__ == "__main__":
    main()
