# FlagGems public sort binding gap — 2026-10-19

This review covers the installed FlagGems 5.0.2 public `torch.sort` path on one contiguous 32x32 float32 tensor. The
active mathematical task is the already reviewed PyTorch `SORT.finite_unique` profile: ascending dimension-one sort,
pairwise distinct values in every row, complete values and unique original indices, and unchanged input.

The installed distribution source, metadata and license are retained as `sources/FLAGGEMS502_SORT.py`,
`sources/FLAGGEMS502_INIT.py`, `sources/FLAGGEMS502_METADATA.txt` and `sources/FLAGGEMS502_LICENSE.txt`. The package
registers both `sort` overloads under `flag_gems.use_gems`; its public implementation performs a radix sort. The
review wrapper enables only `sort`, fixes dimension one and ascending order, checks the physical input and refuses
ties before entering the public call.

A GPU 3 run produced nine Triton launches: one global histogram followed by eight radix sweeps. Complete values and
indices were bitwise equal to public PyTorch, the indices were int64, and the input remained unchanged. A development
binding check also evaluated the complete independent SORT reference successfully. This is useful fixed-call
evidence, but it is not enough for formal admission.

The formal contract measurement was rejected before any bias conclusion. Both public outputs depend on internally
allocated arrays passed between the sweep launches. In particular, the captured dependencies include value and
index arrays plus exclusive-prefix buffers computed by non-Triton PyTorch operations between launches. The current
producer evidence cannot prove that every byte read from those buffers was created inside this call solely from the
declared input and exact constants. Treating their observed bytes as extra inputs would compare a different task,
while ignoring them would hide part of the implementation boundary.

The binding therefore remains `drafted`, with the current `SORT.finite_unique` profile and exact unresolved reason.
Its rejected formal report is `.cache/spec_trial_20261019_flaggems_sort/production_report_20261019.json`. Admission
requires both visibility into the actual public call and a general producer check for deterministic in-call
non-Triton intermediates, including complete byte coverage and dependency on declared inputs. A numeric agreement
test or a name match cannot replace that evidence.

The follow-up GPU probe in `flaggems_sort_trace_visibility_v1.json.gz` makes the first gap concrete. For the public
`torch.sort` entry the trace records `aten.sort.default` only after all nine launches; none of the internal `clone`,
`zeros`, `cumsum`, `sub`, `arange` or repeated `zero_` operations is visible. Calling the original `flag_gems.sort`
function directly produces the same nine kernel names/grids while exposing these ATen events. The direct entry is
only a diagnostic comparison, not a replacement binding or admission of those intermediates.

This agrees with the retained PyTorch 2.10 `_python_dispatch.py` documentation: calls made while handling a dispatch
event forward to the next mode, so the active mode does not recursively observe the implementation. The tool now
records ATen calls enclosing Triton launches and includes their half-open launch ranges in binding refusals and
producer diagnostics. Launch-sequence alignment must not be read as complete observation of internal producers.
This change supplies a more precise refusal; it does not solve the missing trace or certify integer prefix sums.
CPU dispatch regressions and the separate top-k/cumsum/indexing GPU batch exercise the diagnostic. Existing
provenance, trace non-interference and binding regression checks passed 102 tests and 16 subtests.
