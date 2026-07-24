# Phase 2 Compile Graph Audit

## Claim Scope
Execution-path mismatch is relevant only when the training algorithm's own discrete boundary amplifies it into an optimization-semantic fork. A fork is called fragile or bug only under a validated analytic legal bound; raw numerical mismatch alone is not a claim.

## Confound Checklist
- exact step-5 model artifact: PASS
- same input and MATH SDPA context: PASS
- compile warm-up discarded: PASS
- FX graphs persisted: PASS
- Inductor trace artifacts hashed: PASS
- causal numerical injection points fully enumerated: FAIL / pending

## Delta Self Control
Eager logits self equal: True; warmed compiled logits self equal: True.

## Summary
| input_tokens | eager_self_equal | compiled_self_equal_after_warmup | eager_compile_logits_max_abs_delta | dynamo_graph_count | dynamo_graph_break_count | dynamo_op_count | causal_injection_points_enumerated | analytic_legal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 166 | True | True | 0.0625 | 1 | 0 | 93 | False | False |

## Remaining Requirement
Generated fusion kernels and external GEMM calls must be mapped to eager materialization/rounding boundaries with per-kernel arithmetic contracts before they can seed a legal difference bound.

## External Validity
Artifacts are specific to this PyTorch/Inductor build, T4 target, FP16 autocast, sequence shape, and Qwen3-0.6B snapshot. Other shapes or hardware can select different kernels.
