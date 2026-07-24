# Sum Legal-Reassociation Confirmation Findings — 2026-07-16

Contract: `SUM_LEGAL_REASSOCIATION_CONFIRMATION_MANIFEST_V0_1_2026-07-16.md`.

## Result

The frozen PyTorch nightly CUDA input `[2^25, 2, -2^25]` produced:

```text
eager                           2.0
compiled                        0.0
exact real/float64 sum          2.0
gamma_2 * sum(abs(x)) bound     8.000001192093038
eager truth error               0.0
compiled truth error            2.0
eager/compiled raw delta        2.0
default torch.allclose          False
Dynamo calls captured           1
Dynamo unique graphs            1
```

Both outputs lie inside the input-conditioned analytical envelope frozen before execution. The candidate graph was captured, so this is not a fallback-equality control.

```text
eager membership bit:    fail=0
compiled membership bit: fail=0
covered verdict:         ACCEPT
```

## Meaning

This is a real eager/compiled large-but-conforming pair. A fixed equality/default-allclose Oracle rejects it, while the semantic-envelope Oracle accepts both admitted floating evaluation results.

The result does not say the compiled value is more accurate—it is less accurate on this input. It says numerical correctness under the declared evaluation-order contract is a set-membership question, not “whichever is closest to eager” or “whichever is closest to exact real arithmetic.”

## Scope

This confirms one transformed point in the discovered CUDA float32 reduction family. It does not estimate how often such cases occur, validate the conservativeness/usefulness of the envelope for long reductions, or generalize to matmul, normalization and transcendental operators.
