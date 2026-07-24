#!/usr/bin/env python3
"""Synthetic twin-trajectory illustration (not a matched-state Oracle experiment).

Runs three scenarios on a classification task (MLP without BatchNorm so bias
propagates directly through the network):

  A) Reference vs reference (identical) — control, should ACCEPT
  B) Candidate with systematic bias in fc2 — should REJECT, params diverge linearly
  C) Candidate with symmetric noise in fc2 — should ACCEPT, params diverge as √t

Because the two models freely evolve after the first step, hook differences mix
current perturbations with accumulated parameter differences. This script may
illustrate trajectory shapes, but it cannot establish operator B/H/N or safety.
"""

import sys
sys.path.insert(0, "src")

import copy
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from forkcert.oracle import (
    AcceptanceCriteria,
    TwinTrajectoryMonitor,
    Verdict,
)


# -----------------------------------------------------------------------
# Model and data
# -----------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP — no BatchNorm so bias propagates directly."""
    def __init__(self, d_in=32, d_hidden=64, d_out=5):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_dataset(n_samples=1000, d_in=32, n_classes=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, d_in).astype(np.float32)
    W_true = rng.randn(d_in, n_classes).astype(np.float32)
    logits = X @ W_true
    y = logits.argmax(axis=1)
    return torch.from_numpy(X), torch.from_numpy(y)


def train_step(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


# -----------------------------------------------------------------------
# Perturbation via hooks (preserves module structure)
# -----------------------------------------------------------------------

def install_bias_hook(model, module_name: str, bias_value: float):
    """Add a constant bias to a module's output via hook."""
    target = dict(model.named_modules())[module_name]
    def hook(mod, inp, out):
        return out + bias_value
    return target.register_forward_hook(hook)


def install_noise_hook(model, module_name: str, noise_std: float):
    """Add zero-mean noise to a module's output via hook."""
    target = dict(model.named_modules())[module_name]
    def hook(mod, inp, out):
        return out + torch.randn_like(out) * noise_std
    return target.register_forward_hook(hook)


# -----------------------------------------------------------------------
# Run experiment
# -----------------------------------------------------------------------

def run_scenario(
    name: str,
    ref_model: nn.Module,
    cand_model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 200,
    batch_size: int = 32,
    lr: float = 0.01,
    criteria: AcceptanceCriteria | None = None,
):
    print(f"\n{'='*70}")
    print(f"  Scenario: {name}")
    print(f"{'='*70}\n")

    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=lr)
    cand_opt = torch.optim.SGD(cand_model.parameters(), lr=lr)

    target_modules = ["fc1", "fc2", "fc3"]

    if criteria is None:
        criteria = AcceptanceCriteria(
            max_relative_bias=5e-4,
            max_heterogeneity_cv=0.1,
            training_steps=n_steps,
            learning_rate=lr,
        )

    oracle = TwinTrajectoryMonitor(
        ref_model, cand_model, criteria,
        target_modules=target_modules,
    )

    n_samples = X.shape[0]
    reject_step = None

    param_divs = []

    for step in range(n_steps):
        idx = torch.randint(0, n_samples, (batch_size,))
        x_batch = X[idx]
        y_batch = y[idx]

        oracle.begin_step()

        ref_loss = train_step(ref_model, ref_opt, x_batch, y_batch)
        cand_loss = train_step(cand_model, cand_opt, x_batch, y_batch)

        snapshot = oracle.end_step(step, ref_loss, cand_loss)
        param_divs.append(snapshot.param_divergence)

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  Step {step:4d}: ref_loss={ref_loss:.4f}, cand_loss={cand_loss:.4f}, "
                  f"loss_diff={snapshot.loss_diff:+.4e}, "
                  f"param_div={snapshot.param_divergence:.4e}")

        if snapshot.verdict == Verdict.REJECT and reject_step is None:
            reject_step = step

    oracle.detach()

    print()
    print(oracle.report())

    # Divergence growth analysis
    if len(param_divs) >= 20:
        t = np.arange(1, len(param_divs) + 1, dtype=np.float64)
        divs = np.array(param_divs)
        nonzero = divs > 0
        if nonzero.sum() >= 10:
            log_t = np.log(t[nonzero])
            log_d = np.log(divs[nonzero])
            slope, intercept = np.polyfit(log_t, log_d, 1)
            print(f"\n  Divergence growth exponent (log-log slope): {slope:.2f}")
            print(f"    slope ≈ 1.0 → linear growth (systematic bias)")
            print(f"    slope ≈ 0.5 → √t growth (random walk / benign noise)")

    if reject_step is not None:
        print(f"\n>>> REJECT detected at step {reject_step}")
    else:
        print(f"\n>>> No REJECT through {n_steps} steps")

    results = {
        "scenario": name,
        "n_steps": n_steps,
        "reject_step": reject_step,
        "final_param_divergence": param_divs[-1] if param_divs else None,
        "loss_bias": oracle.loss_stats.mean,
        "modules": {},
    }
    for mname, series in oracle.module_series.items():
        results["modules"][mname] = {
            "relative_bias": series.relative_bias,
            "heterogeneity": series.heterogeneity,
            "bias_std_err": series.bias_std_err,
            "n_steps": series.n_steps,
        }

    return results


def main():
    X, y = make_dataset()
    N_STEPS = 200
    LR = 0.01

    criteria = AcceptanceCriteria(
        max_relative_bias=5e-4,
        max_heterogeneity_cv=0.1,
        training_steps=N_STEPS,
        learning_rate=LR,
    )

    # --- Scenario A: identical (control) ---
    torch.manual_seed(42)
    ref_a = MLP()
    cand_a = MLP()
    cand_a.load_state_dict(ref_a.state_dict())
    result_a = run_scenario("A: Identical (control)", ref_a, cand_a, X, y,
                            n_steps=N_STEPS, lr=LR, criteria=criteria)

    # --- Scenario B: systematic bias in fc2 (+0.005) ---
    torch.manual_seed(42)
    ref_b = MLP()
    cand_b = MLP()
    cand_b.load_state_dict(ref_b.state_dict())
    bias_hook = install_bias_hook(cand_b, "fc2", bias_value=0.005)
    result_b = run_scenario("B: Systematic bias in fc2 (+0.005)", ref_b, cand_b, X, y,
                            n_steps=N_STEPS, lr=LR, criteria=criteria)
    bias_hook.remove()

    # --- Scenario C: symmetric noise in fc2 (std=0.005) ---
    torch.manual_seed(42)
    ref_c = MLP()
    cand_c = MLP()
    cand_c.load_state_dict(ref_c.state_dict())
    noise_hook = install_noise_hook(cand_c, "fc2", noise_std=0.005)
    result_c = run_scenario("C: Symmetric noise in fc2 (std=0.005)", ref_c, cand_c, X, y,
                            n_steps=N_STEPS, lr=LR, criteria=criteria)
    noise_hook.remove()

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print()
    print(f"{'Scenario':<50s} {'Reject?':>10s} {'ParamDiv':>12s} {'LossBias':>12s}")
    print("-" * 85)
    for r in [result_a, result_b, result_c]:
        reject = f"step {r['reject_step']}" if r["reject_step"] is not None else "no"
        pdiv = f"{r['final_param_divergence']:.4e}" if r["final_param_divergence"] else "N/A"
        lb = f"{r['loss_bias']:+.4e}"
        print(f"  {r['scenario']:<48s} {reject:>10s} {pdiv:>12s} {lb:>12s}")

    print()
    print("Interpretation limit:")
    print("  These are synthetic free-running trajectories, not matched-state B/H/N.")
    print("  A growth exponent does not by itself prove bias, benign noise, or safety.")

    results_path = "/data1/tzh/forkcert/results/training_oracle_experiment.json"
    with open(results_path, "w") as f:
        json.dump([result_a, result_b, result_c], f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
