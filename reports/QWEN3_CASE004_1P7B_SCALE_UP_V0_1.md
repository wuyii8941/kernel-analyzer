# Qwen3 case 004 — 1.7B-flavor scale-up

Case 001 was repeated with the Qwen3 1.7B flavor dimensions from TorchTitan:
2048 hidden size, 16 query heads, 8 KV heads, and 128 head dimension.  The
input remains a short two-document attention probe so memory and provenance
stay controlled.

The blind locator again found exact upstream boundaries through
`inner_attention`, the first changed named boundary at `wo`, the same generic
operation sequence mismatch (`transpose` versus `view`), and exact repeated
buggy executions.  After the fixed commit was revealed, the changed source
path and transpose mechanism were covered by the same score as the small
calibration case.

This validates that the vertical slice is not tied to the toy 64-dimensional
configuration.  It does not yet validate a full Qwen3-1.7B checkpoint,
training-step endpoint, compiler-generated kernel provenance, or DeepSeek.

The patch-free package audit is recorded at
`results/operator_oracle/qwen3_case004_blind_protocol_audit.json`.
