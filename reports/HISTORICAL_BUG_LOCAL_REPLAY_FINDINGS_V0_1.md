# Historical Bug Local Replay Findings v0.1

Case: `pytorch_adaptive_avgpool_flatten_sum`  
Protocol: `HISTORICAL_BUG_BLIND_PROTOCOL_V0_1.md`

## Reproducible evidence

The two independent reports are:

- `results/operator_oracle/historical_case_local_replay_v0_1/result.json`
- `results/operator_oracle/historical_case_local_replay_v0_1/result_b.json`

The independent audit is:

- `results/operator_oracle/historical_case_local_replay_v0_1/audit.json`

The audit is valid and reports `LOCAL_PRODUCER_WITH_PROVENANCE` with allowed
claim level `LOCAL_INJECTION`.

## What the slice establishes

1. The complete eager/Inductor endpoint differs reproducibly. The compiled
   output repeats exactly in each process; the observed maximum absolute
   difference is `8.7196044921875` for the declared fp32 witness.
2. The `adaptive_avg_pool2d` boundary is exactly equal between the isolated
   eager and compiled pool executions on the same input.
3. The `flatten+sum` suffix, when compiled and given the exact eager pool
   tensor, differs from the eager suffix. This is same-input local production
   evidence for the reduction suffix.
4. Running that same compiled suffix on the compiled pool output does not
   change the result in this witness. Thus the preregistered boundary
   mediation test is negative; production and mediation are not conflated.
5. Inductor debug artifacts expose source nodes (`aten._adaptive_avg_pool2d`,
   `aten.view`, `aten.sum`) and generated kernel paths for the relevant stages
   in both independent processes.

## What it does not establish

- It does not identify a unique faulty operator, compiler pass, or source line.
- It does not prove that the first numerical divergence is the root cause.
- It does not execute repair/injection or prove non-target graph/kernel
  invariance under intervention.
- It does not reveal or use the historical patch; external scoring remains
  pending.

The current result therefore demonstrates that a real reduction-shaped wrong
result can pass through the observation → same-input production → provenance
chain without being converted into an unjustified root-cause claim.
