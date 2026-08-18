# Qwen saved-P softmax

Role: transport or numerical-contract candidate. Frozen semantic region:
`qwen_seq128_layer27_attention_softmax_saved_P`, sequence length 128.

## v2.1 open-loop formation result

The exact Qwen release was measured on 16 calibration and 16 disjoint
confirmation common states. No weights or optimizer state were advanced. The
certificate is
`results/property/bias_formation/formation/qwen_saved_p_seq128.json`.

| layer | calibration | confirmation |
|---|---|---|
| local endpoint | CENTERED | CENTERED |
| parameter gradient | CENTERED | CENTERED |
| effective update | CENTERED | CENTERED |

Confirmation cross-state ratios are 0.00035 (local), -0.00066 (gradient),
and -0.00066 (effective update); all bootstrap intervals remain inside the
frozen centered margin. This open-loop formation run therefore finds no
directional bias stage. Previously observed saved-P trajectory divergence is a
consequence/feedback question; it is not formation evidence and does not turn
this case into a transport or contract positive.
