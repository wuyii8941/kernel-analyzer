# Qwen3 functional-operator barrier batch findings v0.1

## Verdict

VALID_BARRIER_CONDITIONED_BATCH_EVIDENCE

All 18 selected invocations passed. Effect patterns: `{'BOTH_NONZERO': 9, 'BOTH_ZERO': 9}`.

Fifteen rows directly cover their selected high-level functional invocation under a fixed boundary. The three SDPA rows are composite evidence only and do not separately cover qk-bmm, safe-softmax, or probability-value-bmm.

The barrier candidate did not reproduce the original compiled anchor, so no row receives original-candidate root-cause credit.

## Per-target effects

| Target | Pattern | Injection L2 | Repair L2 | Injection signed mean | Repair signed mean |
|---|---|---:|---:|---:|---:|
| `attention.rotary.layer0` | BOTH_NONZERO | 0.08123485 | 0.0664068162 | 0.000173900276 | 1.26883388e-05 |
| `attention.rotary.layer14` | BOTH_NONZERO | 0.0576780327 | 0.056395296 | 0.000327635556 | -0.000165693462 |
| `attention.rotary.layer27` | BOTH_NONZERO | 0.00210248493 | 0.0161742326 | 4.64171171e-06 | 9.84221697e-06 |
| `attention.sdpa.layer0` | BOTH_NONZERO | 0.0628973767 | 0.0706184655 | -3.53343785e-05 | -0.000218197703 |
| `attention.sdpa.layer14` | BOTH_NONZERO | 0.0607379228 | 0.0552600659 | 0.000180199742 | -0.00018395111 |
| `attention.sdpa.layer27` | BOTH_NONZERO | 0.0450740531 | 0.0334528759 | 0.000201169401 | 7.64802098e-06 |
| `mlp.gate_multiply.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `mlp.gate_multiply.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `mlp.gate_multiply.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `mlp.silu.layer0` | BOTH_NONZERO | 0.0797392875 | 0.0789319873 | 0.000314664096 | -0.000202890486 |
| `mlp.silu.layer14` | BOTH_NONZERO | 0.055626642 | 0.0677887052 | 6.73681498e-05 | 3.60608101e-05 |
| `mlp.silu.layer27` | BOTH_NONZERO | 0.0028061436 | 7.80849077e-05 | 4.94718552e-06 | -1.93715096e-07 |
| `residual.attention.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `residual.attention.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `residual.attention.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `residual.mlp.layer0` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `residual.mlp.layer14` | BOTH_ZERO | 0 | 0 | 0 | 0 |
| `residual.mlp.layer27` | BOTH_ZERO | 0 | 0 | 0 | 0 |
