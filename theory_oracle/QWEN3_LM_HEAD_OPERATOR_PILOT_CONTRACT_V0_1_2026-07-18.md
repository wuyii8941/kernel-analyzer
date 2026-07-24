# Qwen3 `lm_head` operator pilot contract v0.1

## Question

At the frozen Qwen3-0.6B GRPO step-29 matched state, how much of the eager/compiled
selected-token log-probability discrepancy is changed by intervening on the final
`lm_head` linear operator?

This is an operator feasibility pilot. It is not a correctness test and it does
not presuppose that eager is mathematical truth.

## Frozen state and observable

- State: `data/qwen3_grpo_heldout_transport_v01_b_step29_transition`.
- Input: the captured step-29 minibatch.
- Precision/runtime protocol: training mode, FP16 autocast, SDPA math backend,
  gradient checkpointing enabled, deterministic algorithms requested.
- Observable: the 4 x 128 selected-token log-probability tensor produced by
  TRL `selective_log_softmax`.
- Reference anchor SHA256: `742468b7b182ea8e70fec4733f702dbcc71ebb64fa3f4aec5e9fbc2450a29806`.
- Candidate anchor SHA256: `1107b4ac9c2662b34572cee3b4b4e1bf454a4b6d0a6def0c427d84f9944a09f2`.

## Arms

- `whole_eager`: the original Accelerate-wrapped eager model.
- `split_EE`: eager Qwen decoder body followed by eager `lm_head`.
- `split_CC`: separately compiled decoder body followed by separately compiled
  `lm_head`.
- `repair_CE`: compiled body followed by eager `lm_head`.
- `injection_EC`: eager body followed by compiled `lm_head`.

The target operator is exactly the final `torch.nn.Linear` module invocation
(`lm_head`), not an arbitrary compiler region. The body is everything upstream.

## Fail-closed validity gates

1. `whole_eager` must reproduce the frozen eager scorer anchor bit-for-bit.
2. `split_EE` must reproduce `whole_eager` bit-for-bit. Otherwise the split does
   not preserve reference realization.
3. `split_CC` must reproduce the frozen whole-compiled scorer anchor bit-for-bit.
   Otherwise introducing a body/head compilation boundary changed the candidate
   realization, and repair/injection cannot be attributed to the original
   candidate.
4. Every measured arm must be bit-exact over two target-state repetitions.
5. Inputs, parameters, training mode, attention backend and precision protocol
   must be shared across arms.

If gate 3 fails, numerical contrasts may be retained only as diagnostics of the
partitioned realization. They are not evidence that `lm_head` caused the original
whole-compiled discrepancy.

## Estimands if valid

For distance `d` on selected-token log-probabilities:

- total discrepancy: `d(split_EE, split_CC)`;
- head repair effect: change from `split_CC` to `repair_CE` and residual
  `d(split_EE, repair_CE)`;
- head injection effect: change from `split_EE` to `injection_EC`;
- interaction/non-additivity: whether the head contrast depends on whether its
  input came from the eager or compiled body.

These are selected-state, intervention-dependent effects. Even a valid result is
not yet a population operator effect or root-cause claim.
