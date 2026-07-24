# Qwen3 GRPO Training-Control Confirmation Contract v0.2 — 2026-07-17

## Status and reason for revision

Frozen before executing v0.2 trajectories or inspecting their signed crossings.

The v0.1 execution completed 20 rollout states and found five descriptive clipping
disagreements, all in states with valid tracked candidate calls. However, Dynamo's
default `recompile_limit=8` caused the final two shape specializations in each
trajectory to fall back outside the tracked Inductor wrapper. The predeclared
all-state identity gate correctly made the whole v0.1 confirmation `INVALID`.

No v0.1 row is deleted and the gate is not relaxed. This revision fixes the
realization protocol and uses new prompt-disjoint states, so it is not a rescore of
the observed v0.1 bank.

## Incorporated contract

All definitions, state fields, clipping semantics, direction/disagreement
estimands, strict finite-bank endpoint, correctness abstention, one-step follow-up
rule and kill criteria from
`QWEN3_GRPO_TRAINING_CONTROL_CONFIRMATION_CONTRACT_V0_1_2026-07-17.md` apply unless
explicitly replaced below.

## Replacement state strata

| Trajectory | Initial model state | Prompt slice | Seed |
|---|---|---|---:|
| A2 | `data/r1_from240_step242_pre` | built-in arithmetic `[320,384)` | 20260720 |
| B2 | `data/r1_from270_step272_pre` | built-in arithmetic `[384,448)` | 20260721 |

Both use a new empty optimizer/scaler state and run 30 eager-anchored GRPO steps.
Ten policy-iteration-2 pre-minibatch states per trajectory are scored. These prompt
slices must be disjoint from all discovery, R1, v0.1 and greedy-impact banks.

## Candidate identity repair

When online compiled scoring is enabled, the executor fixes Dynamo
`recompile_limit=64` before model construction. Multiple shape-specialized tracked
graphs are permitted and reported. The limit changes compilation coverage only; it
does not alter the event rule or acceptance threshold.

Every warmup and both measured compiled calls in all twenty states must invoke at
least one tracked Inductor graph. Exact eager and compiled self-repeat equality is
still required. Any untracked state again makes the complete confirmation
`INVALID`.

## Frozen endpoint

The endpoint is unchanged:

```text
INVALID  if any state/candidate/self/token gate fails
REJECT   if mechanics are valid and at least one applicable token changes clipping
ACCEPT   if mechanics are valid and no applicable token changes clipping
```

Numerical correctness remains `UNINSTANTIATED`; no disagreement is called a compiler
bug or deployment prevalence estimate.

