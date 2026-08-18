# Bias Formation Map

The project goal is no longer a property tournament.  It is to explain how an
implementation difference becomes training bias:

```text
implementation difference
  → local numerical variation
  → bias formation mechanism
  → effective optimizer update bias
  → SEUP persistence
  → parameter drift
```

Kernel Analyzer supplies exact forward/backward semantic ground truth and
repair/sham boundaries.  BiasFormation v2.1 measures the first centered-to-
directional transition in open-loop common states.  Existing SEUP is used only
after that transition to study persistence and drift.

The four frozen cases are Liger fused CE, Phi MM, Qwen saved-P softmax and Qwen
bmm.  No endpoint discovery, T1–T4 redesign, or universal property assumption
is introduced.  Current natural formation cells are intentionally `PENDING`.
The machine-readable protocol and deliverables are under
`results/property/bias_formation/`.

The population file retains the full existing endpoint denominator (1,562
invocation units) and 12 canonical strict/anchor records.  Legacy coherent,
normal, and unresolved roles are provenance-only; they do not become
`LOCAL_BIAS`, `GRADIENT_BIAS`, or `UPDATE_BIAS` until the v2.1 open-loop
formation measurement is run.
