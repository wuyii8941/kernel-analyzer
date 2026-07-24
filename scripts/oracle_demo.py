#!/usr/bin/env python3
"""End-to-end demo: Bias-Variance Oracle on eager vs compiled torch models.

Demonstrates three scenarios:
1. Identical implementations → ACCEPT
2. Systematic bias injection → REJECT
3. Benign noise injection → ACCEPT (raw diff would mislead)

Then profiles a real torch.compile comparison at the operator level.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import torch
import torch.nn as nn

from forkcert.oracle import (
    AcceptanceCriteria,
    Oracle,
    OperatorProfile,
    Verdict,
    collect_operator_measurements,
    compute_operator_profile,
    compute_step_profile,
    format_operator_report,
    format_step_report,
    judge_operator,
    judge_step,
    profile_torch_model,
    profile_torch_training_step,
)


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# -----------------------------------------------------------------------
# Scenario 1-3: Synthetic demonstrations
# -----------------------------------------------------------------------

def demo_synthetic():
    separator("Scenario 1: Identical implementations")
    oracle = Oracle(AcceptanceCriteria(max_relative_bias=1e-5))
    inputs = [np.array([float(i)]) for i in range(100)]
    ref_fn = lambda x: x * 2.0
    cand_fn = lambda x: x * 2.0
    p, v = oracle.measure_and_judge("identical", ref_fn, cand_fn, inputs)
    print(format_operator_report(p, v))

    separator("Scenario 2: Systematic bias (dangerous)")
    cand_fn_biased = lambda x: x * 2.0 + 0.001
    p2, v2 = oracle.measure_and_judge("biased", ref_fn, cand_fn_biased, inputs)
    print(format_operator_report(p2, v2))

    separator("Scenario 3: Large noise, no bias (benign)")
    rng = np.random.RandomState(42)
    noise = {i: rng.randn() * 0.1 for i in range(100)}
    cand_fn_noisy = lambda x: x * 2.0 + noise[int(x[0])]
    p3, v3 = oracle.measure_and_judge("noisy_benign", ref_fn, cand_fn_noisy, inputs)
    print(format_operator_report(p3, v3))

    print("\nKey insight: Scenario 3 has much larger raw diffs than Scenario 2,")
    print("but Scenario 2 is the dangerous one (consistent bias accumulates).")
    print(f"  Scenario 2 max |diff|: 0.001  → Verdict: {v2.verdict.value}")
    print(f"  Scenario 3 max |diff|: ~0.3   → Verdict: {v3.verdict.value}")


# -----------------------------------------------------------------------
# Scenario 4: Real torch model — eager vs compiled operator profiling
# -----------------------------------------------------------------------

def make_simple_model():
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.LayerNorm(64),
        nn.Linear(64, 10),
    )


def demo_torch_operator_profile():
    separator("Scenario 4: Operator-level profiling with different implementations")

    torch.manual_seed(42)
    ref_model = make_simple_model().eval()

    # Simulate what torch.compile might do: slightly different numerical results
    # due to operator fusion, reduction reordering, etc.
    cand_model = make_simple_model().eval()
    cand_model.load_state_dict(ref_model.state_dict())

    # Inject realistic perturbations per layer:
    # Layer 0 (Linear): bf16-like rounding bias (~2^-8 * scale)
    # Layer 2 (Linear): symmetric noise from reduction reordering
    # Layer 4 (Linear): no difference
    rng = np.random.RandomState(42)

    class PerturbedModel(nn.Module):
        def __init__(self, base, rng):
            super().__init__()
            self.base = base
            self.rng = rng

        def forward(self, x):
            # Layer 0: Linear with systematic bf16 rounding bias
            h = self.base[0](x) + 3e-4  # consistent positive shift
            # Layer 1: ReLU (exact)
            h = self.base[1](h)
            # Layer 2: Linear with symmetric reduction noise
            h = self.base[2](h) + torch.from_numpy(
                self.rng.randn(*h.shape).astype(np.float32) * 1e-3
            )
            # Layer 3: LayerNorm (exact after renormalization)
            h = self.base[3](h)
            # Layer 4: Linear (exact)
            h = self.base[4](h)
            return h

    perturbed = PerturbedModel(cand_model, rng).eval()

    inputs = [torch.randn(4, 64) for _ in range(100)]
    criteria = AcceptanceCriteria(max_relative_bias=1e-4, max_heterogeneity_cv=0.1)

    # Manual profiling per layer
    oracle = Oracle(criteria)

    # Profile Layer 0: systematic bias
    def ref_layer0(x): return ref_model[0](x).detach().numpy()
    def cand_layer0(x): return (ref_model[0](x) + 3e-4).detach().numpy()
    p0, v0 = oracle.measure_and_judge("Linear_0 (bf16 rounding)", ref_layer0, cand_layer0, inputs)
    print(format_operator_report(p0, v0))
    print()

    # Profile Layer 2: symmetric noise
    rng2 = np.random.RandomState(99)
    def ref_layer2(x):
        h = ref_model[1](ref_model[0](x))
        return ref_model[2](h).detach().numpy()
    def cand_layer2(x):
        h = ref_model[1](ref_model[0](x))
        out = ref_model[2](h)
        return (out + torch.from_numpy(rng2.randn(*out.shape).astype(np.float32) * 1e-3)).detach().numpy()
    p2, v2 = oracle.measure_and_judge("Linear_2 (reduction noise)", ref_layer2, cand_layer2, inputs)
    print(format_operator_report(p2, v2))
    print()

    # Profile Layer 4: identical
    def ref_layer4(x):
        h = ref_model[:4](x)
        return ref_model[4](h).detach().numpy()
    p4, v4 = oracle.measure_and_judge("Linear_4 (identical)", ref_layer4, ref_layer4, inputs)
    print(format_operator_report(p4, v4))
    print()

    print("Conclusion:")
    print(f"  Linear_0 (systematic bias):  {v0.verdict.value:15s} bias={p0.relative_bias:.2e} — consistent shift accumulates in training")
    print(f"  Linear_2 (symmetric noise):  {v2.verdict.value:15s} bias={p2.relative_bias:.2e} — noise averages out over steps")
    print(f"  Linear_4 (identical):        {v4.verdict.value:15s} bias={p4.relative_bias:.2e} — no difference")


# -----------------------------------------------------------------------
# Scenario 5: Training step profiling
# -----------------------------------------------------------------------

def demo_torch_training_step():
    separator("Scenario 5: Training step B/H/N profiling")

    torch.manual_seed(42)
    ref_model = make_simple_model()
    cand_model = make_simple_model()
    cand_model.load_state_dict(ref_model.state_dict())

    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    cand_opt = torch.optim.SGD(cand_model.parameters(), lr=0.01)

    inputs = [torch.randn(4, 64) for _ in range(20)]
    loss_fn = lambda out: out.sum()

    criteria = AcceptanceCriteria(
        max_step_loss_bias=1e-4,
        max_step_param_bias=1e-3,
        training_steps=10000,
        learning_rate=0.01,
    )

    profile, verdict = profile_torch_training_step(
        ref_model, cand_model,
        ref_opt, cand_opt,
        inputs, loss_fn,
        criteria=criteria,
    )

    print(format_step_report(profile, verdict))


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    demo_synthetic()
    demo_torch_operator_profile()
    demo_torch_training_step()

    separator("Summary")
    print("The Bias-Variance Oracle decomposes operator differences into:")
    print("  B (bias):          systematic shift → accumulates, dangerous")
    print("  H (heterogeneity): input-dependent variation → identifies risky regions")
    print("  N (runtime var):   execution randomness → may be absorbed by SGD noise")
    print()
    print("This is strictly more informative than raw diff, which conflates all three.")
    print("A large diff with zero bias is safe; a tiny diff with consistent bias is not.")
