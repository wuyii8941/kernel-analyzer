# Bias-formation v2 protocol

This document freezes the corrected measurement semantics.  The v1 files are
an immutable scaffold and contain no scientific measurement result.

## Separate formation from consequence

`BiasFormationTrace` is open-loop and common-state only.  For each of 16
calibration and 16 independent confirmation states, candidate and repair use
the same weights, optimizer state, input, RNG and scheduler state.  It records

```text
local endpoint residual -> parameter-gradient residual -> effective update residual
```

It is the ground-truth measurement layer.  It may use candidate and repair
measurements, so the accurate statement is `verdict_blind`, not
`candidate_blind`.  Historical T1--T4/SEUP labels and trajectory drift are
forbidden inputs.

`BiasConsequenceTrace` is separate closed-loop evidence.  It records the
endpoint-mediated local increment, feedback increment, actual drift increment
and final drift.  It cannot set `first_confirmed_bias_stage`.

## Complete-vector and fail-closed rules

Complete-coordinate Gram/U-statistic/bootstrap summaries are primary.  A scalar
projection is auxiliary and must carry a `ProjectionCertificate` naming its
space, construction states, basis digest, orientation rule and freeze point.
Missing layers, missing common-state digests, nonfinite values, insufficient
states and unproven projection provenance never become centered by imputation.

`first_confirmed_bias_stage` is emitted only when every upstream layer is
explicitly `CENTERED` and the current layer is explicitly `BIASED`.  If an
upstream layer is unresolved, the result may report an observed downstream
bias, but its formation point remains `UNRESOLVED`.

## Exact roster status

The bound roster is in `results/property/bias_formation_v2/roster_bound.json`.
The feasibility report is intentionally conservative and was generated without
starting a GPU campaign.  SiLU has six unique natural states, not 32, and the
literature Flash Attention control is not an executable project artifact; both
are ineligible rather than padded or relabeled.

## Frozen property candidates

Only P1 conditional source asymmetry, P2 source--transport alignment and P3
forward--backward numerical consistency are operationally frozen at this stage.
Their required fields, matched interventions, shams and support/counterexample
rules are machine-readable in `property_specs.json`.  P4--P6 are deferred until
the observed transition layer makes them scientifically applicable.

No GPU measurement or property claim is made by this v2 scaffold.
