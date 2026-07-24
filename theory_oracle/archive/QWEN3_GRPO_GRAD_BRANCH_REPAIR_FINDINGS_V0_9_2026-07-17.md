# Qwen3 GRPO Grad-Branch Repair Findings v0.9 — 2026-07-17

## Verdict

The frozen v0.9 execution and independent safetensors audit are **VALID** for the
declared single-state controlled branch-intervention query.

The unified result is `COMPLETE` for its required clipping-impact, controlled
update-impact, variability and branch-functional attribution ledgers. Numerical
correctness remains `UNINSTANTIATED`.

## What was established

At the preselected trajectory-A step-14 event, eager is unclipped and the tracked
candidate is clipped for one negative-advantage token. All repeated complete
scorer tensors exactly reproduce the v0.4 event bank.

The three controlled arms realize the intended derivative gate:

- eager ordinary branch: selected-logp derivative is nonzero;
- candidate ordinary clipped branch: selected-logp derivative is zero;
- candidate scorer with reference branch repair: derivative is nonzero.

An independent verifier loaded the three saved next-parameter vectors and found
nonzero A–B and B–C distances. C is substantially closer to A than B is, but C
does not equal A. Therefore the selected clipping branch has a context-specific
contribution to the controlled update, while the residual shows that it is not
the whole eager/candidate difference.

## The more important Oracle result

Model parameters, batch, RNG and implementation labels were not a complete
matched state for this subject. The candidate at step 14 depended on the prior
shape-specialization history:

- cold compilation at the target shape produced a different compiled scorer;
- replaying the frozen `[4,168] -> [4,167]` specialization sequence reproduced
  the original graph hashes, node counts and all 512 candidate values exactly.

Compiler cache/specialization/autotuning state must therefore be included in the
state or fixed by the implementation/randomness protocol whenever it changes the
realized candidate. It is neither automatically a fixed implementation shift nor
within-state runtime noise. Which category it occupies depends on whether the
query conditions on, randomizes over, or leaves that state uncontrolled.

The failed v0.5–v0.8 attempts remain useful validity controls: wrong tensor-hash
geometry, cold compiler realization, non-native graph wrapper and leaked GPU
resource lifetime were all rejected rather than converted into zero effects.

## Claim boundary

This result does **not** show:

- eager or compiled is mathematically correct;
- the event is common under a target training distribution;
- the controlled SGD magnitude equals a natural GRPO/AdamW update effect;
- the clipping branch is an operator or kernel root cause;
- repair is uniquely necessary or sufficient;
- any long-run convergence, reward or quality consequence.

The attribution level is `INTERVENTION_DEPENDENT`. Injection, interactions and
held-out replication are absent, and the intervened object is a boundary-conversion
branch function rather than a source operator.

## Evidence

- contract and frozen manifest:
  `QWEN3_GRPO_GRAD_BRANCH_REPAIR_CONTRACT_V0_9_2026-07-17.md`,
  `QWEN3_GRPO_GRAD_BRANCH_REPAIR_MANIFEST_V0_9.json`;
- executor result and independent audit:
  `results/training_step_oracle/qwen3_grpo_grad_branch_repair_v0_9/result.json`,
  `results/training_step_oracle/qwen3_grpo_grad_branch_repair_v0_9/audit.json`;
- unified query/evidence/result:
  `QWEN3_GRPO_GRAD_BRANCH_REPAIR_UNIFIED_QUERY_V0_9.json`,
  `QWEN3_GRPO_GRAD_BRANCH_REPAIR_UNIFIED_EVIDENCE_V0_9.json`,
  `results/training_step_oracle/qwen3_grpo_grad_branch_repair_v0_9/unified_oracle_result_v0_9.json`.
