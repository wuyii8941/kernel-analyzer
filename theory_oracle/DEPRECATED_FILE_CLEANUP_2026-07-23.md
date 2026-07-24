# Cleanup record — 2026-07-23

The repository contains a large historical experiment archive. This cleanup
does not delete evidence merely because it is old: reports, manifests and
scripts referenced by prior results remain available for reproducibility.

Removed in this pass:

- generated Python `__pycache__` directories;
- generated `.pytest_cache`;
- stale README links to files that no longer exist.

This follow-up pass also records the active subject boundary in
`QWEN3_SUBJECT_SCOPE_V0_1.md`.  No compact result, manifest, or historical
definition was deleted: the old Qwen/phase material is evidence, not an
active method.  The repository is intentionally cleaned by policy and
explicit deprecation labels rather than by deleting versioned artifacts whose
provenance may later be needed.

Kept deliberately:

- historical fork/mutation/phase reports;
- superseded Oracle definitions, because they document definition changes;
- old Qwen operator experiments, because their artifacts are useful when
  moving from the 0.6B calibration subject to a larger model;
- TVM historical cases and the generic locator, because TVM is the calibration
  subject for the harder Qwen/Megatron work.

Future deletion should be based on an explicit artifact dependency audit, not
on filename age or version suffix alone.
