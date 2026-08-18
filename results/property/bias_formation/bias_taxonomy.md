# Bias Formation Taxonomy

This taxonomy is the frozen analysis vocabulary. Current case-level evidence is
reported below it; no cross-case mechanism claim is made.

| Stage | Formation pattern | Required evidence |
|---|---|---|
| A — Source bias | local → gradient → update directional | source-centering intervention reduces downstream bias |
| B — Transport bias | local centered → gradient biased | valid residual/transport permutation removes gradient bias |
| C — Numerical contract bias | representation contract repair removes bias | semantics and ABI preserved; sham unchanged |
| D — Optimizer bias | gradient centered → update biased | optimizer-state intervention changes update bias |
| E — No bias | difference remains centered | complete formation layers and negative control |

The first transition is assigned only from the v2.1 open-loop formation matrix.
SEUP and trajectory drift are consequence evidence and cannot fill formation
labels.  If the required decomposition or intervention is absent, the stage is
`UNRESOLVED`, not an inferred mechanism.

## Current map

* **Phi MM:** `CENTERED → BIASED → BIASED`, first confirmed at the
  parameter-gradient layer. This is a transport/contract candidate only until
  its declared intervention is run.
* **Qwen saved-P:** all three layers are `CENTERED` in both partitions. This is
  a measured case-level centered result, not evidence that all local error is
  harmless.
* **Liger:** calibration local status is `BIASED`, but confirmation is
  `UNRESOLVED_INSUFFICIENT_STATES`; no source stage is confirmed.
* **Qwen bmm:** `INELIGIBLE` because exact repair/sham provenance is absent;
  it is not a negative label.
