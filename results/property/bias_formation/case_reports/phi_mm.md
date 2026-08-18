# Phi MM

Role: arithmetic replication. Frozen endpoint:
`phi4_seq64:lm_head.input_gradient.mm`, sequence length 64.

## v2.1 open-loop formation result

The exact Phi release was measured on 16 calibration and 16 disjoint
confirmation common states. No weights or optimizer state were advanced. The
compact certificate is
`results/property/bias_formation/formation/phi4_lm_head_dx_seq64.json`.

| layer | calibration | confirmation |
|---|---|---|
| local endpoint | CENTERED | CENTERED |
| parameter gradient | BIASED | BIASED |
| effective update | BIASED | BIASED |

The confirmation cross-state ratios are 0.1067 (local), 0.6754 (gradient),
and 0.6754 (effective update); their 95% bootstrap intervals are
[0.0895, 0.1246], [0.5411, 0.8251], and [0.5411, 0.8251]. Under the frozen
formation map this is a **PARAMETER_GRADIENT** first observed/confirmed
transition: a transport-or-contract candidate, not a source-bias verdict.

This is formation evidence only. It does not by itself prove whether the
gradient direction is caused by source/transport pairing or by a forward/backward
numerical contract; that requires the declared intervention. It also does not
use T4 or SEUP as a label.

## Consequence

The separate Phi SEUP replay is recorded in
`results/property/bias_formation/consequence/phi4_lm_head_dx_seup.json`. It
passes the frozen consequence gates: signed persistence 1.0, local projected
accumulation fraction 0.9992, recurrence residual at most 7.04e-9, and final
selected-carrier drift L2 4.77e-5. This connects the measured gradient-stage
formation candidate to a persistent effective-update drift, but it does not
identify whether transport pairing or a numerical contract is the formation
cause.

## Exploratory pairing intervention

`interventions/phi4_mm_transport_pairing.json` keeps the local residual norm and
row multiset fixed while permuting its row pairing. The natural gradient
population is BIASED and the shuffled population is CENTERED, with local norm
error below 1e-6 on all 16 states. However, the simple analytic RMSNorm
transport reconstruction misses the actual gradient delta by 0.32--0.60
relative error. The strict result is therefore `MEASURED_NO_TRANSPORT_SUPPORT`:
the pairing effect is recorded as evidence, but it is not promoted to a P2
mechanism until the complete semantic transport region is closed.
