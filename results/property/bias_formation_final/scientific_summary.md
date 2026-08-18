# Bias Formation Map — final current result

## Scientific question

When does an implementation-induced numerical difference stop being harmless variance and become directional training bias?

The measured chain is:

```text
implementation difference -> local residual -> parameter-gradient residual
                         -> effective-update residual -> SEUP consequence
```

## Current answer

Phi MM is the only completed natural case with a confirmed formation transition: local error is centered, while the parameter-gradient and effective-update populations are directionally biased in both 16-state partitions. Its separate SEUP replay closes the consequence link, but the residual/transport pairing intervention is not yet a complete causal transport proof because the current analytic reconstruction misses part of the gradient delta.

Qwen saved-P is centered at all three measured layers. This is a valuable case-level variance-only observation, not a universal negative. Liger has a directional calibration signal but its independent confirmation interval crosses the frozen bias margin, so source bias remains unresolved. Qwen bmm is not eligible for formation labeling because exact repair/sham provenance is missing.

## What can be claimed

1. The formation pipeline distinguishes local, gradient, and update stages with open-loop common states.
2. A real example exists where bias first appears at the parameter-gradient stage (Phi MM).
3. A local difference can remain centered through all measured stages (Qwen saved-P).
4. Persistence and formation are separate: Phi's SEUP consequence does not serve as its formation label.
5. No universal source, transport, contract, or optimizer property is established yet.

## What remains open

The endpoint denominator is retained in `population_screening.csv`, but only the exact v2.1 capture cases receive formation labels. The remaining endpoint population is explicitly `NOT_CAPTURED_EXISTING_ARTIFACT_ONLY`; legacy T1--T4 and SEUP roles are provenance, never formation ground truth.

The next scientific step is to close Phi's complete semantic transport decomposition and add an independent eligible case before promoting transport bias beyond a case-specific candidate.
