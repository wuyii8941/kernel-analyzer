# Short persistence screening oracle

This is the low-cost screening layer after development property profiling.  It
does not replace exact matched F+B repair or a 32-step paired trajectory.

For each endpoint and each ordered reference state, the runtime adapter emits a
declared effective-update residual vector.  All endpoints share the same short
reference state sequence.  The screen immediately compresses each vector with a
fixed signed CountSketch and discards the raw vector.

For a projected path (x_1,ldots,x_T), it reports

\[
A(T)=\frac{\|\sum_t x_t\|}{\sqrt{\sum_t\|x_t\|^2}},
\]

prefix growth, short-lag correlation, and a sign-flip null that preserves each
per-state vector norm.  A risk candidate must exceed the frozen sign-flip 95th
percentile, have at least two positive short-lag correlations, and show late
prefix growth.  This is a screen, so every positive must be rechecked with the
exact repair protocol.

The screen is coordinate-free with respect to unrelated natural inputs: it
uses consecutive reference states, not a fixed direction learned from unrelated
texts.  It is also endpoint-agnostic: the adapter only needs the complete
F+B/optimizer effective-update residual and a declared parameter coordinate
set.

The current implementation is in
`src/kernel_analyzer/short_persistence.py` and
`scripts/run_short_persistence_screen.py`.  Development property profiling is
reported separately in `results/property/bias_property_search/`.

## Runtime adapter

Existing paired-trajectory runners can write a temporary geometry spool with
one tensor tree per state (for example, `--geometry-spool`).  The shared
adapter consumes that exact interface:

```bash
PYTHONPATH=src python scripts/run_short_persistence_from_spool.py \
  --spool /data1/tzh/tmp/case_geometry.pt \
  --phase evaluation --steps 8 --field effective_update \
  --output results/property/bias_property_search/short_screen_case.json
```

The declared `carrier_parameters` order is part of the input certificate and
must be identical across all selected states.  The adapter rejects scalar-only
trajectory JSON, missing parameters, coordinate changes, and nonfinite
tensors.  It stores only CountSketch paths, state IDs, parameter order, and
source digests; the temporary spool may then be removed under the project
resource policy.

The pre-held-out property scope is frozen in
`results/property/bias_property_search/property_freeze_v1.json`.  Source
asymmetry and source--transport coupling are conditional formation branches;
transport concentration is supporting-only because it overlaps centered
controls; carrier stability is the short-trajectory confirmation gate.

## Cost boundary

The intended production path is one shared 8--16-state reference capture for
all endpoints, fixed (k)-dimensional sketches, and exact trajectories only
for screen-positive endpoints plus a mechanically sampled set of negatives.
Missing effective-update vectors, changing coordinate sets, nonfinite values,
and unresolved F+B boundaries fail closed as `UNRESOLVED`; they are never
treated as safe.
