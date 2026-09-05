# Results layout

`coverage/` is the authoritative all-operator and Flash-style funnel.  It
contains frozen input banks, runtime releases, full-coordinate T1 artifacts,
and causal/carrier/trajectory evidence.  `final/` contains compact historical
derivations and case measurements.

Do not interpret a screen-positive endpoint as a mathematically established
bias with loss consequences. Read [the evidence map](../docs/case_evidence_map.md)
for connections between derivations, measurements and training outcomes.
T1--T4 and [the old registry](../cases_flash_style.md) retain historical meanings.
There is no single historical audit JSON that covers every later experiment.

Raw tensors and compiler products are intentionally kept outside the
repository under `/data1/tzh`.

`property/single_point_collapse_v1/` and `property/single_point_collapse_v2/`
contain the frozen protocols and JSON records for the Liger single-boundary
collapse attempt.  Their `summary.json` files are the compact machine-readable
conclusions. For the first v2 stream continued to 10000 steps, also read
`full_10000_summary.json`; its loss difference reverses sign without undoing the
observed trajectory separation. The corresponding checkpoints remain outside Git under
`/data1/tzh/cache/kernel-analyzer/`.
