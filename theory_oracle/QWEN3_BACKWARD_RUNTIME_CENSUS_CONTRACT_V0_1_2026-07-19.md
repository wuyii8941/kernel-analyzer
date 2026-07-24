# Qwen3 compiled backward runtime census contract v0.1

Run one fresh compiled arm of the already validated step-29 natural GRPO
transition. Preserve the v0.2 construction gates, scorer anchor, real loss,
scaled backward, unscale, global clipping, captured AdamW, GradScaler and
scheduler suffix.

Immediately before the single `Tensor.backward()` call, replace every resolved
generated backward kernel object and the shared external `mm`/`bmm` callables
with counting proxies that delegate unchanged to the original call. Restore all
objects after backward. Require every one of the 39 statically inventoried
Triton families to resolve, at least one to execute, both external operations to
execute, exactly one backward hook call, and the complete base transition and
candidate identity gates to remain valid.

This is an outcome-independent runtime census. It identifies which static
generated families are live in this selected transition and their call counts.
It changes no declared arithmetic and gives no repair, injection, causal,
population or correctness coverage.
