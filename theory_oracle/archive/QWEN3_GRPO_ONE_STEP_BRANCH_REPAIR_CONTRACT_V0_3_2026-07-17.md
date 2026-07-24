# Qwen3 GRPO One-Step Branch-Repair Contract v0.3 — 2026-07-17

## Status

Frozen before execution. This contract supersedes the unexecuted v0.2 contract.
The v0.1 execution is invalid because it failed endpoint realization; v0.2 is
withdrawn because its proposed clipped-branch intervention was not functionally
the reference branch.

## Parent evidence and selected event

The parent is the valid Qwen3 GRPO training-control confirmation v0.2. Its
predeclared event-ordering rule selected trajectory A, optimizer step 11,
rollout batch 3, case `grpo_000003_dc31e4bafb66`, token index 100 (flat index
228), token id 18395, negative advantage. The reference path is clipped and the
compiled path is unclipped.

The selection was not based on discrepancy magnitude or an expected repair
outcome.

## Frozen state and execution realization

All arms reload the same reconstructed pre-minibatch model and use a new empty
SGD state (`lr=1e-5`, no momentum, `foreach=False`). The four responses and all
512 token objectives remain in their original batch order.

The scorer must reproduce the exact Trainer realization:

- identical left-padded prompt/response tensors and attention masks;
- `use_cache=False`;
- Qwen `logits_to_keep = completion_length + 1`;
- removal of the final next-token logits and retention of completion positions;
- TRL `selective_log_softmax` at temperature 1;
- identical FP16 autocast and SDPA MATH context.

The selected A log-probability must exactly equal the parent eager anchor. The
selected B and C log-probabilities must exactly equal the parent compiled anchor.
Any failure makes the run `INVALID`.

## Arms and intervention

| Arm | Numerical path | Selected-token objective branch |
|---|---|---|
| A | eager | ordinary reference decision |
| B | tracked Inductor | ordinary compiled decision |
| C | same tracked Inductor scores as B | reference branch functional form |

All non-selected objectives are ordinary GRPO objectives. B and C must have
identical hashes of the complete 512-element log-probability tensor, identical
old log-probabilities and advantages, and matching compiled graph identities.

“Reference branch functional form” is defined before evaluation at the compiled
ratio:

- if the reference branch is unclipped, use `ratio * advantage`;
- if the reference branch is clipped with positive advantage, use
  `(1 + epsilon) * advantage`;
- if the reference branch is clipped with negative advantage, use
  `(1 - epsilon) * advantage`.

For the selected negative-advantage event, C therefore uses the constant lower
boundary objective. Its selected `dLoss/dlogp` must be exactly zero, while B's
ordinary in-range branch must have nonzero gradient. This is a validity gate,
not an empirical hypothesis.

## Estimands and endpoints

Let `T_A`, `T_B`, and `T_C` be the realized next parameter states. Report:

- selected-token `dLoss/dlogp`, loss, and full-gradient norm;
- `d(A,B)`, `d(A,C)`, and `d(B,C)` in parameter L2 distance;
- signed reference-directed repair effect `d(A,B) - d(A,C)`;
- residual ratio `d(A,C) / d(A,B)` when `d(A,B) > 0`.

`d(B,C)` is the realized one-step effect of changing this one objective branch
under the compiled numerical scores. A positive reference-directed effect means
this intervention moves the compiled next state toward A; zero or negative is a
valid possible result and must not be hidden.

## Claim boundary

This estimates an intervention-dependent selected-branch contribution for one
frozen state and one SGD probe. It does not establish source-operator causality,
necessity, sufficiency, long-run harm, or compiler incorrectness. Numerical
correctness remains uninstantiated for this Qwen transition.

