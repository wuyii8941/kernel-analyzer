# Qwen3 decoder layer 27 localization findings v0.1

## Verdict

`INVALID_FOR_ORIGINAL_CANDIDATE_LOCALIZATION`.

Reference preservation and repeatability passed, but the essential candidate
gate failed:

- frozen whole-compiled scorer SHA256:
  `1107b4ac9c2662b34572cee3b4b4e1bf454a4b6d0a6def0c427d84f9944a09f2`;
- layer-split all-compiled scorer SHA256:
  `0caf1a4c5e3f18e4fb918ea9e3d571a6874946d24d033d26db2f8d8338cdba57`.

All arms were internally repeat-exact. Therefore the failure is a stable change
of compiled realization caused by introducing prefix/layer/tail boundaries, not
runtime noise.

## Consequence

The observed mixed-arm layer-27 contrasts are deliberately not interpreted.
In particular, their nonzero values do not show that layer 27 contributed to the
original whole-compiled discrepancy. The intervention changed the treatment by
changing fusion, specialization or another compilation-context choice before it
changed the selected layer implementation.

This is a concrete counterexample to the claim that “repair makes the discrepancy
smaller” is sufficient for root-cause attribution.

## Method change required

Naive layer/module splitting cannot be the default operator-analysis method.
The next stage must:

1. inventory ATen nodes and fused kernels in the unchanged, anchor-valid original
   compiled graph;
2. use exact endpoint-preserving repair/injection only where a standalone
   operator boundary exists;
3. for an operator inside a fused kernel, introduce a matched barrier-control arm
   and report only a barrier-conditioned intervention effect unless the original
   endpoint is preserved.

Layer 27 remains unlocalized. No operator inside it is implicated or excluded by
this invalid experiment.
