# Qwen3 backward multi-role cast repair contract v0.1

## Question

Can a generated family with one kernel body but several source-graph roles be
analysed without collapsing those roles into a misleading family average?

## Frozen role map and interventions

The subject is `triton_poi_fused__to_copy_26`, observed 84 times.  Runtime
signatures separate 56 calls of shape `[3072, 1024]` from 28 calls of shape
`[1024, 3072]`, but shape alone does not separate the two roles inside the
56-call cluster.  Static adjacency in each execution triple maps calls modulo
three as:

- 0: FP16-to-FP32 up-projection weight-gradient conversion;
- 1: FP16-to-FP32 gate-projection weight-gradient conversion;
- 2: FP16-to-FP32 down-projection weight-gradient conversion.

Before execution, this campaign selects calls 0, 1 and 2: all three roles in
the first-executed transformer-block triple.  Each arm replaces only its one
selected conversion with the eager `to(float32)` operation.  The other 83
family calls remain compiled.  Complete clipped-gradient and AdamW-update
vectors are retained.

## Validity and interpretation

Validity requires the frozen candidate and scorer identities, one backward
hook, one generated module, exactly 84 family calls, and exactly one hit and
repair of the declared call.  Failed gates invalidate an arm.

The estimand is role-specific selected-state repair impact.  This campaign can
resolve the role ambiguity for the selected triple; it does not establish that
the same role is interchangeable across all layer positions.  Exact-null
effects would be evidence about these selected casts only.  Eager is not a
correctness specification, and no root-cause, injection, necessity,
sufficiency, population, long-run or correctness claim follows.
