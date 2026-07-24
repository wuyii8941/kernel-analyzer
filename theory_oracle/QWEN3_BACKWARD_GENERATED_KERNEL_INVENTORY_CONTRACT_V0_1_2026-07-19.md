# Qwen3 backward generated-kernel inventory contract v0.1

The source is the already audited generated Qwen3-0.6B GRPO step-29 AOT
backward trace. The source inventory compiled but did not execute backward, so
the descriptive denominator includes every static generated Triton definition
and external-kernel call site appearing in that trace. It must reconcile with
the previously audited static 9,471 backward ATen/prims nodes across 40 target
types.

The parser must find exactly one `model__1_backward` trace, 39 generated Triton
families whose definitions all have static call sites, and the observed
external call-site counts of 563 `mm` and 168 `bmm`. A generated family is a treatment candidate,
not a proven semantic equivalence class; same-name calls cannot replace
per-invocation evidence until role, fusion, shape and state transport pass.

This artifact supplies only a static backward generated-treatment denominator.
It does not prove these paths execute for the target loss and grants no repair,
injection, causal, population or correctness coverage.
