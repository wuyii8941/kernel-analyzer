#!/usr/bin/env python3
"""Render the evidence-bounded persistence-property development report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/persistence_v1"


def read(relative: str) -> dict:
    return json.loads((BASE / relative).read_text())


def main() -> None:
    orbit = read("orbits/comparison.json")
    schedule = read("interventions/phi_schedule.json")
    sr = read("interventions/phi_sr.json")
    feedback = read("feedback_matched_floor.json")
    phi, qwen = orbit["cases"]
    sr_values = [row["coherence_amplification"] for row in sr["sr_vs_fp32"]]
    report = f"""# Persistence property v1: development result

## Result

The surviving candidate is **transported conditional-mean persistence**:

> An implementation difference produces source-persistent parameter drift when
> its conditional mean over a semantics-preserving arithmetic orbit has a
> temporally non-canceling component, and the real F+B/optimizer transport
> preserves or amplifies that component. Orbit variance without this transported
> temporal mean is diffusive under the tested protocol.

For an endpoint residual $\\epsilon_{{t,\\pi}}$ under equivalent arithmetic
variants $\\pi$, define

$$m_t=\\mathbb{{E}}_\\pi[\\epsilon_{{t,\\pi}}],\\qquad
q_{{t,\\pi}}=\\epsilon_{{t,\\pi}}-m_t.$$

The property is not $\\|m_t\\|$ alone. It is the ordered persistence of the
transported mean $M_t m_t$, where $M_t$ is the actual reference F+B and optimizer
map. This separates bias generation from the later feedback term.

## Decisive development pair

| Case | orbit-mean energy | source $A$ | effective-update $A$ |
|---|---:|---:|---:|
| Phi lm_head dX | {phi['orbit_mean_energy_fraction']:.6f} | {phi['default_source_temporal_amplification']:.6f} | {phi['effective_update_temporal_amplification']:.6f} |
| Qwen128 v_proj | {qwen['orbit_mean_energy_fraction']:.6f} | {qwen['default_source_temporal_amplification']:.6f} | {qwen['effective_update_temporal_amplification']:.6f} |

Qwen has the larger orbit-mean energy fraction but no effective-update
persistence. Therefore orbit-mean magnitude is falsified as the property.
Phi has a small but significant temporally shared source component, which its
backward transport amplifies; Qwen's source directions largely re-draw across
states.

## Matched interventions on the real Phi backward MM

Randomizing the exact GEMM K-axis ordering preserves real $GW$ semantics. It
changed amplification from
`{schedule['natural_vs_fp32']['coherence_amplification']:.6f}` to
`{schedule['random_schedule_vs_fp32']['coherence_amplification']:.6f}`
(ratio `{schedule['amplification_ratio']:.6f}`). Fixed schedule is therefore not
the main anchor: most bias is shared by the reduction orbit.

Replacing deterministic low-precision materialization by unbiased stochastic
rounding produced four amplifications `{', '.join(f'{value:.6f}' for value in sr_values)}`;
their mean/natural ratio is `{sr['sr_to_natural_amplification_ratio']:.6f}`.
The SR endpoint residual norm is larger than the deterministic endpoint
difference, so the loss of persistence cannot be explained by smaller error.

## Feedback boundary

`A_B>1` alone is not a feedback mechanism certificate. The bmm hard control has
feedback amplification `{feedback['comparisons'][0]['bmm_floor']:.6f}`. Saved-P
does not exceed that floor, Qwen128 exceeds it by only about 7%, and SiLU by
about 54%; all remain unresolved until an RMS-matched perturbation null is run.

## Current claim and boundary

Supported now:

* semantic-orbit mean magnitude alone is not predictive;
* fixed reduction order is not necessary for Phi persistence;
* removing conditional mean with SR suppresses Phi persistence while increasing
  local error magnitude;
* source temporal structure and F+B transport are both required in the observed
  Phi/Qwen pair.

Not yet supported:

* universality across operator families;
* a source-free predictor for arbitrary kernels;
* feedback-sustained bias as a distinct mechanism;
* long-horizon loss failure (M7).

## Frozen confirmation

The next confirmation must not reuse Phi or Qwen128 for threshold selection.
Use Liger fused CE as a reduction/accumulation positive and one independently
bound attention or state-space endpoint as a negative or second positive. The
prediction is frozen before those values are measured:

1. a temporally persistent orbit mean transported into the declared parameter
   predicts effective-update persistence;
2. an orbit mean whose direction re-draws predicts diffusion even when its norm
   is large;
3. stochastic centering must reduce persistence without requiring lower RMS.

Until that confirmation is complete, the result is a causally supported
development property, not a universal oracle.
"""
    (BASE / "property_report.md").write_text(report)
    summary = {
        "schema": "kernel-analyzer-persistence-property-summary-v1",
        "status": "SUPPORTED_DEVELOPMENT_PROPERTY_AWAITING_HELDOUT_CONFIRMATION",
        "name": "TRANSPORTED_CONDITIONAL_MEAN_PERSISTENCE",
        "schedule_anchor_hypothesis": "COUNTEREXAMPLE_FOUND",
        "orbit_mean_magnitude_hypothesis": "COUNTEREXAMPLE_FOUND",
        "stochastic_centering_intervention": "SUPPORTED_ON_PHI",
        "cross_kernel_development_pair": ["phi4_seq64_lmhead_dx", "qwen128_vproj_mm"],
        "heldout_confirmation_complete": False,
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(BASE / "property_report.md")


if __name__ == "__main__":
    main()
