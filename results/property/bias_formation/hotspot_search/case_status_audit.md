# Bias-case status audit

This file separates the existing complete Flash-style evidence from the
current BiasFormation v2.1 population screen.  The two ledgers answer
different questions and must not be merged.

## Current counts

| ledger | count | interpretation |
| --- | ---: | --- |
| compiler-bound F+B semantic cells | 791 | frozen four-model denominator |
| direct/semantic formation screens | 727 | complete open-loop local/gradient/update screen |
| exact downstream-closure screens | 40 | internal semantic region screened at its exact reachable carrier |
| semantic-region rows still pending | 24 | no verdict; not centered and not negative |
| short-screen signals | 69 | candidate selection only |
| new strict v2.1 confirmations in this rescreen | 0 | no new 16+16 formation case |

## Evidence tiers

### Strict v2.1 formation case

The known Phi-4 seq64 `lm_head dX` case is the only currently confirmed
v2.1 formation transition:

```text
LOCAL_CENTERED -> PARAMETER_GRADIENT_BIASED -> EFFECTIVE_UPDATE_BIASED
```

Its transport intervention is useful evidence, but the analytic transport
factorization remains unresolved.

### Complete Flash-style semantic cases already in the repository

The repository also contains complete forward/backward, repair, carrier and
paired-trajectory cases whose boundary is a closed semantic region rather than
the v2.1 population layer:

1. Liger fused CE — source/accumulation case; formation confirmation remains
   unresolved in the v2.1 screen.
2. Phi-4 seq64 `lm_head dX` — strict v2.1 formation case above.
3. Qwen layer-23 q-projection attention-state region — valid closed semantic
   Flash-style case.  `S_bwd`-only repair closes the directional component,
   `K`-only repair does not, the sham is exact, and the 32-step live-weight
   trajectory is complete.  It is not a single-kernel attribution and its
   strict v2.1 formation label is `NOT_CAPTURED`, not a centered or biased
   imputation.

The older saved-P and other semantic-region artifacts remain valid boundary
or negative evidence where their own reports say so; they do not become new
formation positives merely because a complete F+B trajectory exists.

Qwen CE/lm-head internal backward regions at seq64, seq128, and seq256 were
completed with independent 16+16 open-loop populations. All three are CENTERED
at the local endpoint, parameter-gradient, and effective-update layers in both
partitions. They close three pending loss-path rows without adding a positive
case.

## Execution boundary

The Mamba seq256 transport-top-three attempt passed one-state engineering
reach, then was stopped before scientific sampling because the installed
backend used a CPU sequential fallback.  It has no formation verdict and is
not counted above.  No active GPU campaign is running at this checkpoint.

## Reporting rule

Use the following wording:

> The systematic F+B rescreen found no new strict v2.1 formation case.  The
> repository nevertheless contains a second independent closed semantic-region
> Flash-style case (Qwen layer-23 attention state), in addition to the Liger and
> Phi complete cases.  Semantic-region completeness and v2.1 formation labels
> are reported separately.
