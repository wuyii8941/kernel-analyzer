# Held-out property validation (v1)

This audit is the close-out of the currently frozen development protocol. It
does not change the case registry or promote feedback drift into a source
persistence case.

## What was frozen

The predictor was frozen before the held-out trajectory was opened. Its scope
is **source persistence and source/transport formation**. It is not a general
predictor for feedback-sustained drift, routing/graph divergence, unresolved
VJP boundaries, or optimizer-only effects.

## Held-out results

| Population | Implementation relation | Frozen prediction | Consequence | Interpretation |
|---|---|---|---|---|
| Llama-3.2 3B lm-head dX | seen implementation, new operands | `SOURCE_PERSISTENCE_RISK` | 32-step actual amplification 1.221, above sign-flip null | source predictor supported on a new operand distribution |
| Ministral-3 3B lm-head dX | seen implementation, new operands | `SOURCE_PERSISTENCE_RISK` | 32-step actual amplification 1.229, above sign-flip null | same conditional mechanism reproduced |
| Gemma-4 E2B PLE/RMSNorm | new implementation pattern | `NO_SOURCE_PERSISTENCE_UNDER_PROTOCOL` | existing 32-step record has local 1.001 but feedback 3.235 / actual 3.225 | correct scope separation: feedback-sustained, out of source-predictor domain |

The Llama and Ministral records are a cross-model/new-operand validation of
the already observed lm-head implementation family; they are not two new
implementation classes. Gemma is a new implementation class and supplies no
new positive source case. Its trajectory drift must not be counted as a false
negative for the source predictor because the feedback predictor was
explicitly abstained before trajectory execution.

The attempted confirmation-population rerun is currently blocked by the
runtime's CUDA/driver visibility after model loading. That is an execution
block, not a scientific verdict; the existing frozen Gemma consequence record
is retained with its original provenance.

## Current claim boundary

The property has cross-model evidence within one implementation family and
has been tested against a new implementation-class out-of-domain case. There
is still no positive `NEW_IMPL` validation, so no universal cross-implementation
claim is made. The correct next stage is a pre-enumerated, non-overlapping
implementation pool and a low-cost source-persistence screen; controls and
out-of-domain feedback cases remain in the denominator and are not renamed as
new positives.

