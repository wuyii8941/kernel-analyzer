# Mamba seq256 transport-top3 checkpoint

The case plan contains three explicitly selected `LOCAL_CENTERED` short-screen
transport candidates (`backward:18750:output_0`, `backward:23604:in_out_ptr0`,
and `backward:24770:out_ptr0`).

- `mamba_seq256_r1` was rejected before measurement because its wrapper and
  reference graph were stale.
- `mamba_seq256_r2` was rebuilt with matching compiler inventory/campaign.
- The one-state engineering preflight passed all three exact endpoint and
  reference-cut gates.
- The formal 16+16 run completed only state 0 before being stopped: the
  installed Transformers Mamba path disables selective-scan kernels and falls
  back to a CPU-heavy sequential implementation (about 20 minutes per next
  state in this configuration).

No formation verdict is assigned. The partial state-0 vectors are not a
scientific sample and are excluded from the candidate map. The short-screen
signals remain candidates only; no new strict case was found.
