# Liger fused CE

Role: source-bias anchor. Frozen endpoint:
`qwen3_liger_fused_linear_ce:dW`, sequence length 128.

## v2.1 open-loop formation result

The exact fused dW endpoint and tied-parameter gradient were measured on 16
calibration and 16 disjoint confirmation common states. No weights or
optimizer state were advanced. The compact certificate is
`results/property/bias_formation/formation/liger_fused_ce_t128.json`.

| layer | calibration | confirmation |
|---|---|---|
| local endpoint | BIASED | UNRESOLVED_INSUFFICIENT_STATES |
| parameter gradient | UNRESOLVED_INSUFFICIENT_STATES | UNRESOLVED_INSUFFICIENT_STATES |
| effective update | UNRESOLVED_INSUFFICIENT_STATES | UNRESOLVED_INSUFFICIENT_STATES |

The calibration local lower bootstrap bound is 0.2501, just over the frozen
bias margin, but the independent confirmation bound is [0.1815, 0.3817] and
does not clear that margin. Therefore this run does **not** confirm a source
formation transition in the full 311,164,928-coordinate carrier. The earlier
SEUP/Liger carrier evidence remains consequence evidence; it is not copied
into this formation label. A source-centering intervention is consequently
not triggered by the frozen protocol.
