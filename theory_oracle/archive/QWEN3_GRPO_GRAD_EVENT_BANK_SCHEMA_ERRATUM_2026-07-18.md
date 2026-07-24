# Qwen3 GRPO Grad-Event Bank Schema Erratum — 2026-07-18

The frozen v0.4 evaluator and evidence use all tokens whose advantage sign is
nonzero. The denominator is 4,608 and each token uses its sign-specific GRPO
clipping boundary.

The unified query/evidence endpoint identifier
`grad_context_negative_advantage_grpo_clipping` is therefore too narrow as a
label. It should be read as `grad_context_grpo_clipping_all_nonzero_advantages`.

This is a naming erratum, not a numerical correction:

- the token inclusion rule, denominator, event identities, directions and verdict
  remain unchanged;
- both observed events happen to have negative advantage;
- frozen v0.4 JSON artifacts are not rewritten post hoc;
- new contracts must use the corrected endpoint name and state the sign-specific
  boundary rule explicitly.
