# Results layout

`coverage/` is the authoritative all-operator and Flash-style funnel.  It
contains frozen input banks, runtime releases, full-coordinate T1 artifacts,
and causal/carrier/trajectory evidence.  `final/` contains compact historical
derivations and case measurements.

Do not interpret a screen-positive endpoint as a case.  Read
[PROJECT.md](../PROJECT.md) for the current T1/T2/T3/T4 status and
[cases_flash_style.md](../cases_flash_style.md) for the verdict chain.

Raw tensors and compiler products are intentionally kept outside the
repository under `/data1/tzh`.

`property/single_point_collapse_v1/` and `property/single_point_collapse_v2/`
contain the frozen protocols and JSON records for the Liger single-boundary
collapse attempt.  Their `summary.json` files are the compact machine-readable
conclusions; the corresponding model checkpoints remain outside Git under
`/data1/tzh/cache/kernel-analyzer/`.
