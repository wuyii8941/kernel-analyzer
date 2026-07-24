# Qwen3 GRPO Boundary-Conditioning Incremental-Value Contract v0.1 — 2026-07-18

## Purpose and scope

Test, on the already frozen v0.4 grad-enabled finite bank, whether the semantic
boundary transformation is nonredundant with ranking tokens by raw absolute
implementation delta.

This is a retrospective diagnostic on a fixed bank. It is not held-out evidence,
a population claim, a new event discovery rule, or a correctness test.

## Population

Use every v0.4 token with nonzero advantage, applying its sign-specific GRPO
clipping boundary. Keep all twenty rollout-state clusters and both trajectories.
Do not resample tokens as if independent states.

## Frozen quantities

For token `j`, let

```text
d_j = logp_candidate - logp_reference
b_j = log(1.2) if advantage_sign > 0 else log(0.8)
q_j = advantage_sign * (b_j - (logp_reference - old_logp))
```

For either sign, the reference clips iff `q_j < 0`. Define:

```text
raw_score_j = |d_j|

boundary_score_j =
  max(0,  advantage_sign*d_j) / q_j       if q_j > 0
  max(0, -advantage_sign*d_j) / (-q_j)    if q_j < 0
  +infinity                 if q_j = 0 and d_j != 0
  0                         if q_j = 0 and d_j = 0.
```

`boundary_score` is a descriptive crossing-pressure transform: it incorporates
direction and reference distance. A value above one is algebraically equivalent
to crossing this binary boundary, apart from exact-boundary policy. It is not a
learned predictor and must not be advertised as one.

## Required reports

1. stable semantic event count and identities;
2. each event's descending rank under `raw_score` and `boundary_score`, with
   deterministic tie handling;
3. maximum raw score among non-events and its boundary distance;
4. minimum raw score among events and its boundary distance;
5. whether raw and boundary rankings are identical;
6. finite-bank average precision for both rankings, labelled descriptive because
   events and scores come from the same fixed bank;
7. state-cluster identities for every event and every reported counterexample.

## Interpretation and kill rule

- If rankings are identical and event/non-event ordering is identical, this bank
  provides no incremental ranking evidence over raw delta.
- If a semantic event ranks materially differently or large raw non-events outrank
  it, boundary conditioning is nonredundant on this bank.
- Because `boundary_score` contains the exact event geometry, superior event
  ranking is construct validation, not evidence of out-of-sample prediction.
- No conclusion about correctness, prevalence, update harm or operator cause is
  licensed.
