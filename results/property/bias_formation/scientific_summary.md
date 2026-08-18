# Bias Formation Map — current status

The protocol is frozen and the three eligible cases have now completed the
32-state open-loop formation campaign. The fourth case, Qwen bmm, remains
explicitly ineligible because its frozen roster has no exact repair/sham
wrapper source.

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

The measured transition matrix is:

| case | local | parameter gradient | effective update | interpretation |
|---|---|---|---|---|
| Liger fused CE | UNRESOLVED in confirmation | UNRESOLVED | UNRESOLVED | calibration-only local signal; no held-out formation confirmation |
| Phi MM | CENTERED | BIASED | BIASED | transport/contract candidate; SEUP consequence passes |
| Qwen saved-P | CENTERED | CENTERED | CENTERED | case-level variance-only candidate |
| Qwen bmm | INELIGIBLE | INELIGIBLE | INELIGIBLE | missing exact candidate/repair/sham provenance |

Synthetic detector controls remain separate from natural-case evidence. Existing
SEUP results remain consequence evidence only. No mechanism intervention is
promoted automatically: Phi requires a valid transport or numerical-contract
intervention, while Liger's confirmation gate did not trigger a source
intervention. Phi's separate consequence replay passes signed persistence and
recurrence gates, so the formation-to-persistence chain is measured for that
case even though its root formation mechanism remains unresolved.

The frozen population denominator is `bias_population.csv`: 1,562 endpoint
units from the existing endpoint population, plus 12 canonical known
strict/anchor records.  Its legacy endpoint roles are 57 coherent-carrier
observations, 588 normal references, and 917 unresolved rows.  These counts
are retained for coverage and sampling; they are not formation positives or
negatives. The case-level cells above come only from compact v2.1 certificates
and frozen roster feasibility.

The v2.1 reducer is `scripts/build_bias_formation_certificate.py`.  It rejects
the old single `common_state_digest` format, requires component-wise equality,
and does not emit a confirmed formation point from an incomplete layer.
