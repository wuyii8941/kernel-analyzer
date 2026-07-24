# Case 002: blind view/storage-offset replay

This report records a bounded validation slice, not a compiler root-cause
claim.  The anonymous package is
`theory_oracle/blind_cases/case_002/`; the two independent replay results are
`results/operator_oracle/view_offset_case002_a.json` and `..._b.json`, audited
by `view_offset_case002_audit.json`.

## Evidence

- The complete compiled case disagrees with the declared reference on the
  target shape after the prescribed warm-shape calls.
- The first `view` boundary is an exact control on every target row.
- The second `view` region, given the exact eager first-view tensor, produces a
  repeatable discrepancy.  This is local same-input production evidence.
- Replaying the fixed second-view suffix on the compiled first-view boundary
  does not change the output relative to the eager boundary.  No separate
  boundary mediation effect is observed.
- Two independent processes agree row by row.
- Torch 2.7 emits a wrapper-level `reinterpret_tensor` artifact but no
  auditable Triton kernel/origin mapping for this case.
- The same anonymous input contract under Torch 2.11.0+cu126 on the same T4
  has an exact compiled/reference result on every target row and no local
  production signal.  This is a version-boundary validation, not evidence of
  a particular upstream patch.

## Maximum claim

`LOCAL_INJECTION_WITH_WRAPPER_STOP`.

The result supports a reproducible local production signal at the second-view
boundary and supports stopping at the wrapper/AOT level.  It does not identify
a unique faulty operator, generated kernel, compiler source line, or historical
patch.  In particular, a kernel-ranking tool would be overclaiming on this
case.

## Why this case matters

The original issue is known publicly as a silent `torch.compile` wrong-output
case, but this blind package intentionally excludes that context.  The old
release reproduces the witness while newer environments screened so far do
not.  The external issue page labels it a silent correctness issue and records
that the trigger involves non-zero storage offsets and shape-dependent
execution; those facts are used only in the post-hoc audit, not by the blind
locator.  See [PyTorch issue #155690](https://github.com/pytorch/pytorch/issues/155690).

The key methodological outcome is negative as well as positive: local replay
works, but provenance is insufficient for a kernel claim.  A credible pipeline
must report that boundary and stop rather than inventing a kernel root cause.
