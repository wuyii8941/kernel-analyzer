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
repair/sham boundaries.  BiasFormation v2.1 measures only a global,
state-comparable centered-to-directional transition in open-loop common
states.  BiasFormation v2.2 adds conditional and trajectory observation levels;
failure of the v2.1 global gate is no longer interpreted as absence of
training bias.  Existing SEUP is used only after formation to study persistence
and drift.

The original four-case pilot was Liger fused CE, Phi MM, Qwen saved-P softmax
and Qwen bmm.  It remains a historical open-loop/global measurement and is not
the current case denominator.  The current systematic audit uses eight unique
closed F+B paired-trajectory cases under
`results/property/bias_formation_systematic/`.
The machine-readable protocol and deliverables are under
`results/property/bias_formation/`.

The population file retains the full existing endpoint denominator (1,562
invocation units) and 12 canonical strict/anchor records.  Legacy coherent,
normal, and unresolved roles are provenance-only; they do not become
`LOCAL_BIAS`, `GRADIENT_BIAS`, or `UPDATE_BIAS` merely because an older T1–T4
or SEUP artifact exists. The current measured cells are recorded in
`bias_transition_matrix.csv`: Phi is
`LOCAL_CENTERED → GRADIENT_BIAS → UPDATE_BIAS`, saved-P is centered through all
three layers, and Liger remains unresolved on confirmation.

The final compact package is under
`results/property/bias_formation_final/`. It contains the fail-closed
population screening/matrix, Phi transport mechanism, Qwen layer-23 attention
state mechanism, anchor reports, intervention results, SEUP consequence
summary, taxonomy, and `SHA256SUMS`. Rows without a v2.1 capture are explicitly
`NOT_CAPTURED_EXISTING_ARTIFACT_ONLY`; legacy T1–T4 and SEUP evidence is never
promoted into formation labels. The two validated mechanisms are deliberately
kept at their evidence boundaries: Phi is an empirical composite transport
mechanism, while layer-23 is a closed semantic region rather than a single
kernel attribution.

The v2.1 global-scope interpretation is superseded for training-bias claims by
the v2.2 addendum and trajectory reclassification under
`results/property/bias_formation_v22/`. The strict audit reports eight complete
paired trajectory artifacts and eight semantic cases; six retain the older
fixed-direction evidence and two are trajectory-only observations. One additional
layer-23 key repair artifact is retained as an incomplete candidate because
its same-weight sham is not exact; it is not counted. P1–P6 mechanism
identification has now produced the two-channel effective-antithetic-symmetry
map:

```text
E[F(epsilon)|c] = integral p_s F_e + integral p_a F_o.
```

Liger/Phi give matched evidence for event/pairing asymmetry; saved-P/SiLU give
matched evidence for optimizer response rectification.  The remaining four
cases retain partial or unresolved boundaries.  This is a cross-case working
property, not yet a zero-shot predictor for unseen operators.
