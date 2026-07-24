# Qwen3 operator coverage contract v0.1

## Why this contract exists

“All operators” is ambiguous. This contract separates four denominators so that
an inventory cannot be presented as causal coverage:

1. **operator type**: e.g. `aten.mm`, `aten.mean`;
2. **operator invocation**: e.g. layer 7 `q_proj`, layer 19 input RMSNorm;
3. **fusion context / kernel family**: the generated treatment containing one or
   more operator invocations;
4. **training domain**: model forward, scorer/post-processing, loss and decision,
   model backward, gradient control, AMP and optimizer update.

## Frozen first coverage universe

The first concrete universe is the Qwen3-0.6B GRPO step-29 matched transition
under the already validated FP16/SDPA-math/Inductor protocol. It includes:

- both observed forward specializations;
- the dynamic model backward graph;
- scorer `selective_log_softmax`;
- GRPO loss and clipping decision;
- gradient clipping, AMP/GradScaler, AdamW and scheduler transition.

Coverage of this universe does not imply coverage of other shapes, states,
compiler configurations, attention backends or models.

## Coverage states

Every declared unit must be in exactly one state:

- `OBSERVED_ONLY`: present in a validated graph or execution path;
- `MAPPED_NOT_INTERVENED`: correspondence to generated kernel is known;
- `VALID_NULL_EFFECT`: valid repair/injection found no implementation-specific
  effect in the tested scope;
- `VALID_EFFECT`: valid intervention changed the Oracle endpoint;
- `BARRIER_CONDITIONED`: only a matched barrier-control effect is identifiable;
- `INVALID_TREATMENT`: intervention changed undeclared compilation context;
- `JUSTIFIED_EQUIVALENCE`: covered by a declared equivalence class whose
  transport has been validated across required layer positions and states;
- `UNINSTANTIATED`: required realization or intervention is absent.

## When a family is covered

Sharing an ATen name or generated kernel does not prove equivalence. An operator
family can replace per-invocation intervention only after:

1. shape, dtype, semantic role and fusion context are declared;
2. early, middle and late layer representatives are tested when applicable;
3. at least the declared matched-state distribution is sampled;
4. repair and injection, or an explicit reason one is not identifiable, are
   reported;
5. treatment-integrity gates pass;
6. failures and null effects are both retained.

## Completion claims

- **Descriptive forward coverage** requires every forward node to be present in
  the validated FX/IR/kernel ledger.
- **Forward causal coverage** requires every forward invocation or justified
  equivalence class to have a valid coverage state beyond observation/mapping.
- **Training-step causal coverage** additionally requires scorer, loss/decision,
  backward, gradient-control, AMP and optimizer domains.
- **Population coverage** additionally requires replication over the declared
  matched-state distribution.

The discrepancy Oracle can be statistically well-defined before operator causal
coverage is complete. What is prohibited is using a valid Oracle to imply that
operator root cause or full training coverage has already been established.
