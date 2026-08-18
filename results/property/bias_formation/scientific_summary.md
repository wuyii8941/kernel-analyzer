# Bias Formation Map — current status

The project has frozen the scientific question and measurement protocol, but
has not yet measured natural formation stages.  Therefore this file makes no
claim that Liger, Phi, saved-P or bmm belongs to any stage.

The current deliverable is a map from implementation difference to the next
measurement boundary:

```text
implementation difference
  → local residual
  → parameter-gradient residual
  → effective-update residual
  → SEUP consequence
  → parameter drift
```

Synthetic detector controls are separate from natural-case evidence.  Existing
SEUP results remain consequence evidence only.  The next permitted scientific
measurement is the v2.1 16+16 open-loop formation campaign after all case
preflight and repair/sham provenance blockers pass.

The frozen population denominator is `bias_population.csv`: 1,562 endpoint
units from the existing endpoint population, plus 12 canonical known
strict/anchor records.  Its legacy endpoint roles are 57 coherent-carrier
observations, 588 normal references, and 917 unresolved rows.  These counts
are retained for coverage and sampling; none is a formation positive or
negative until v2.1 open-loop measurements produce all three formation layers.

The v2.1 reducer is `scripts/build_bias_formation_certificate.py`.  It rejects
the old single `common_state_digest` format, requires component-wise equality,
and does not emit a confirmed formation point from an incomplete layer.
