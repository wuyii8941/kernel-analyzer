# Qwen3 functional-operator barrier batch contract v0.1

## Scope

At the frozen Qwen3-0.6B step-29 state, instrument layers 0, 14 and 27 for:

- MLP SiLU;
- MLP activated-gate multiplication;
- attention residual add;
- MLP residual add;
- paired q/k rotary application;
- the high-level SDPA call.

This gives 18 selected invocations.  SDPA remains a composite functional
operator: it does not separately cover its decomposed qk-bmm, safe-softmax and
probability-value-bmm primitives.

## Treatment

The installed forward methods reproduce the exact installed Transformers Qwen3
logic, replacing only the selected functional call with a persistent
`torch.compiler.disable` boundary.  Each boundary selects an eager or separately
compiled implementation of the same operation.

As in the named-module batch, compare all-eager reference vs one-target
injection and all-compiled barrier candidate vs one-target repair.  Two exact
repeats, exact eager-anchor reproduction, exact mode-call counts and no outer
recompile are mandatory.

Passing rows receive only `BARRIER_CONDITIONED` coverage unless the barrier
candidate exactly reproduces the frozen whole-compiled candidate.
