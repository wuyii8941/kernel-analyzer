# Qwen3 backward runtime metadata contract v0.1

Use the same frozen compiled natural GRPO step-29 transition and the validated
39-Triton plus two-external backward denominator. Immediately before the single
backward call, wrap every family with a delegate-only observer.

For every one of the 1,857 calls, record only ordered call index and argument
metadata: tensor shape, stride, dtype, device, storage offset and
`requires_grad`, plus scalar values and keyword types. Do not retain tensor
values, raw storage or numerical summaries. Require exact family-by-family and
external call counts, all 39 generated families resolved, one backward hook,
the original scorer anchor and the complete natural-transition validity gates.

The purpose is to partition repeated same-name families into candidate
shape/layout roles before selecting interventions. Metadata equality alone does
not prove semantic equivalence. The observer delegates unchanged arithmetic
and grants no repair, injection, causal, population or correctness credit.
