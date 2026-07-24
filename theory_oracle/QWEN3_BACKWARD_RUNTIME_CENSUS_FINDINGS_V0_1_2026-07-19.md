# Qwen3 compiled backward runtime census findings v0.1

The frozen runtime census and independent audit are valid. The instrumented arm
reproduces the anchored compiled scorer and completes the same natural GRPO
loss, scaled backward, unscale, global clipping, captured AdamW, GradScaler and
scheduler transition as the previously validated natural-transition query.

All 39 statically inventoried Triton families execute. Their 1,126 runtime call
counts agree family-by-family with the generated code. External calls also
agree exactly: 563 `mm` and 168 `bmm`. The selected backward execution therefore
has a dynamically validated denominator of 41 generated/external family names
and 1,857 calls.

This upgrades the backward evidence from a static AOT census to a live
one-state denominator; it does not make any family causally covered. Every
counting proxy delegates unchanged arithmetic, and gradient-checkpoint forward
recomputation inside backward remains part of the observed generated backward
execution. Repair/injection, cross-state transport and correctness remain
absent.
