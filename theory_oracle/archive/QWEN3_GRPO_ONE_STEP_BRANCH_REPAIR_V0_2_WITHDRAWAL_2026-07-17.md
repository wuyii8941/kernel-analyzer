# Qwen3 GRPO One-Step Branch-Repair v0.2 — Withdrawal

## Status

`WITHDRAWN BEFORE EXECUTION` on 2026-07-17. There is no v0.2 result artifact.

The Trainer-realization correction in v0.2 was necessary but insufficient. A
static audit found that the inherited intervention implemented “force clipped”
as the expression

```text
clamp(compiled_ratio, 1 - epsilon, 1 + epsilon) * advantage
```

evaluated at the compiled ratio. For the selected event, the compiled ratio is
inside the clipping interval. The clamp therefore has a nonzero derivative and
does not reproduce the flat clipped branch selected by the reference path.

This is not evidence about the branch effect. It is a defect in the proposed
intervention. v0.3 replaces it with the reference branch's functional form and
adds a zero-gradient gate plus a full-batch B/C log-probability identity gate.

