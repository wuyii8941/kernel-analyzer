# Qwen3 named-operator barrier batch findings v0.1

## Verdict

VALID_BARRIER_CONDITIONED_BATCH_EVIDENCE

All 35 named-module representatives passed the fixed-boundary integrity gates. Effect patterns: `{'BOTH_NONZERO': 13, 'BOTH_ZERO': 22}`.

The barrier candidate did not reproduce the original compiled anchor. Every row below is therefore intervention-dependent `BARRIER_CONDITIONED` evidence and receives no original-candidate root-cause credit.

## Per-target effects

| Target | Pattern | Injection L2 | Repair L2 | Injection signed mean | Repair signed mean |
|---|---|---:|---:|---:|---:|
| `embedding.token` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.down_proj.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.down_proj.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.down_proj.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.gate_proj.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.gate_proj.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.gate_proj.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.k_proj.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.k_proj.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.k_proj.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.o_proj.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.o_proj.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.o_proj.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.q_proj.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.q_proj.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.q_proj.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.up_proj.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.up_proj.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.up_proj.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.v_proj.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.v_proj.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `linear.v_proj.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `norm.input.layer0` | BOTH_NONZERO | 0.0691206977 | 0.086813204 | 0.000145658851 | 0.000229261816 |
| `norm.input.layer1` | BOTH_NONZERO | 0.072741203 | 0.0841033608 | 0.000208396465 | 0.000119313598 |
| `norm.input.layer14` | BOTH_NONZERO | 0.058671616 | 0.059123788 | 0.000260937959 | 0.000147022307 |
| `norm.input.layer27` | BOTH_NONZERO | 0.0063046962 | 0.0154008642 | -1.9878149e-05 | 4.40180302e-05 |
| `norm.k_norm.layer0` | BOTH_NONZERO | 0.074546583 | 0.0727178305 | 0.000190481544 | 0.000199425966 |
| `norm.k_norm.layer14` | BOTH_NONZERO | 0.0687125698 | 0.0661225766 | 0.000251565129 | 0.000370264053 |
| `norm.k_norm.layer27` | BOTH_NONZERO | 0.0424565747 | 0.0484799184 | 0.000148657709 | 0.00020442903 |
| `norm.post_attention.layer0` | BOTH_NONZERO | 0.0670824945 | 0.0676041022 | 0.000424414873 | 0.000143062323 |
| `norm.post_attention.layer14` | BOTH_NONZERO | 0.0471987948 | 0.069615297 | 0.000130362809 | 9.00104642e-05 |
| `norm.post_attention.layer27` | BOTH_NONZERO | 2.48687829e-05 | 1.12840271e-05 | 4.47034836e-08 | -3.35276127e-08 |
| `norm.q_norm.layer0` | BOTH_NONZERO | 0.0857278481 | 0.0752161741 | 0.00026294589 | 6.5036118e-05 |
| `norm.q_norm.layer14` | BOTH_NONZERO | 0.0729226768 | 0.0772759467 | 0.000329274684 | 0.000153090805 |
| `norm.q_norm.layer27` | BOTH_NONZERO | 0.0420339108 | 0.045049075 | 0.000138457865 | 0.000103484839 |

## Interpretation

A zero row means only that isolated compilation of that invocation had no selected-token effect in this fixed-boundary program and state. A nonzero row establishes an effect in the same program, not in the original fused candidate. Early/middle/late rows are separate observations; they do not yet validate transport across layers or states.
