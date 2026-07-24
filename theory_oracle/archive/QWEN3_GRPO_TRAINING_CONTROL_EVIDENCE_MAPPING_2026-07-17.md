# Qwen3 GRPO Training-Control Evidence Mapping — 2026-07-17

## Purpose

This record maps retained ForkCert Qwen3 GRPO evidence into the Training-Step
Oracle v0.2 ledgers. It does not retroactively turn exploratory analyses into a
preregistered correctness experiment.

## Subject

- implementation pair: eager PyTorch versus `torch.compile`/Inductor;
- model family: Qwen3-0.6B;
- precision/hardware: FP32 master weights, FP16 autocast, Tesla T4;
- training decision: sign-specific PPO/GRPO ratio clipping for nonzero-advantage
  response tokens;
- downstream endpoint: token loss derivative, full gradient and one identical SGD
  update for the replayed step-5 case.

Eager is the baseline, not mathematical truth.

## Ledger mapping

| Oracle ledger | Evidence | Status | Claim boundary |
|---|---|---|---|
| matched-state validity | shared weights, token IDs, old log-probs, advantages, masks, position semantics, seed and path controls | valid for audited cases | not a universal state reconstruction result |
| runtime repeatability | exact-zero self deltas on the canonical scan and exact repeated step-5 gradient norms | no variability observed under this protocol | two/few repeats do not prove universal determinism |
| average relative shift | full-scan signed log-prob mean close to zero with prompt-cluster interval spanning zero | descriptive | no universal directional compiler shift |
| state heterogeneity | nonzero deltas with varying signs/magnitudes and boundary exposure | instantiated | mechanism source not identified |
| semantic event | 5 audited natural clipping disagreements among 39,936 applicable token decisions | finite-scan compatibility evidence | discovery scan, not deployment prevalence or correctness |
| one-step impact | step-5 fork changes target `dLoss/dlogp`, gradient norm and next SGD parameters | positive scoped impact witness | one case; no long-run quality claim |
| numerical correctness | no independent legal floating envelope | `UNINSTANTIATED` | no bug label |
| operator attribution | graph barriers and hooks changed compilation context; identity-hook controls failed | invalid for unique operator causality | whole-region/intervention sensitivity only |

## Why this is a better training example than greedy argmax

The greedy Qwen confirmation is an application compatibility endpoint. Greedy
selection is not used by the teacher-forced cross-entropy update in that experiment.

The GRPO clipping case is different: the discrete event is inside the actual loss
definition. In the replayed step-5 witness:

```text
eager branch:    unclipped, nonzero target derivative
compiled branch: clipped, zero target derivative
repair arm:      compiled numerical value, eager branch decision
```

The repair arm reduced the eager/compiled one-step parameter distance but left a
substantial residual. Therefore the event has a direct causal contribution to this
one-step endpoint, while continuous backend discrepancy and possible interactions
remain. “The fork is the whole error” is rejected.

## Direction and disagreement

The five natural cases include both clipping directions. The overall signed
log-prob mean is near zero. This is a concrete counterexample to treating global
mean numerical shift as the semantic Oracle: boundary-conditioned event disagreement
can be nonzero when the global signed mean is negligible.

The retained reports do not establish a stable population directional clipping
shift. The defensible positive result is disagreement/existence under the frozen
scan and a one-step effect for one replayed case.

## Attribution interpretation

The branch repair is a valid intervention on the clipping decision and estimates:

> the change in the chosen one-step endpoint when that one decision is set to the
> eager branch while the compiled numerical log-prob is retained.

It is not an operator repair. Disabling or hooking attention, MLP, RMSNorm or the
LM head altered graph partition/context. These interventions cannot establish that
four operators independently caused the original discrepancy.

## Authoritative evidence

- natural scan: `reports/phase4.md`;
- audited cases: `reports/fork_cases.md`;
- gradient-clipping pilot/control: `reports/phase7_gradclip.md`;
- step-5 branch repair and update effect: `reports/phase8_case_step5.md`;
- matched-step artifact:
  `results/matched_step/clip-step5-grpo_000001_2817771126c0-t80.json`;
- held-out zero-exposure record: `reports/r1_heldout.partial.md`.

## Remaining confirmation need

A new population claim requires a frozen checkpoint/prompt/trajectory sampling
design before inspecting margins and signed crossings. Boundary-enriched stress
states may be used to test mechanism sensitivity, but their event frequency must
not be reported as natural prevalence.

