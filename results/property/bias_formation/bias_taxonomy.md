# Bias Formation Taxonomy

This taxonomy is the frozen analysis vocabulary, not a result claim.

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
