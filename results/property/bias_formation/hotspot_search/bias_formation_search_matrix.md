# Bias-formation search matrix

This search is not a hunt for a larger number of positives.  The unit remains
one complete forward plus its actual backward, with the candidate and exact
repair compared from the same state.  The purpose of this matrix is to find
the transitions needed to explain when implementation variance becomes a
directional training signal.

## Formation transitions

| transition | scientific role | preferred semantic regions |
| --- | --- | --- |
| `LOCAL_BIASED -> GRADIENT_BIASED` | source/arithmetic bias | accumulation, cast, reduction, fused loss |
| `LOCAL_CENTERED -> GRADIENT_BIASED` | backward transport or F+B contract bias | attention backward, CE/lm-head backward, saved state |
| `GRADIENT_CENTERED -> UPDATE_BIASED` | optimizer rectification | Adam/momentum/clipping boundary (later) |
| difference with all layers centered | variance-only control | bmm, normalization writeback, add/loss controls |

The Phi seq64 `lm_head dX` result is the existing transport anchor.  It is not
an oracle by itself: a second transport/contract case and explicit centered
controls are needed before a formation rule can be claimed.

## Search order

1. **Loss path:** CE/lm-head backward and tied embedding reach, across Qwen,
   Phi, DeepSeek and Mamba.  This is the shortest path from an F+B difference
   to a parameter gradient.
2. **Attention backward:** saved state, softmax VJP, `dQ/dK/dV`, and q/k
   normalization semantic regions.  Internal differences that are erased by a
   BF16 writeback remain negative controls, not positive cases.
3. **Normalization backward:** RMSNorm/LayerNorm/qk-norm complete regions,
   always including the following reduction and parameter-gradient reach.
4. **State-space backward:** Mamba recurrent ports, selected by four-state
   screening and then frozen 16+16 confirmation.  Source-directional and
   transport-amplified candidates are kept as separate mechanisms.
5. **Optimizer boundary:** only if a complete F+B case is centered through the
   parameter gradient but biased after the actual optimizer map.

## Current selection rule

The four-model denominator is frozen at 791 compiler-bound F+B cells.  Short
screens are used only to select candidates; they never count as cases.  A
candidate becomes a case only after disjoint confirmation shows a stable first
formation stage.  Missing semantic ports remain `SEMANTIC_REGION_PENDING` and
are not imputed as centered.

Current evidence is:

- 727 exact representatives have a direct local/gradient screen (including
  three completed Qwen CE/lm-head semantic closures);
- 40 semantic regions have an exact downstream-closure screen;
- 24 semantic-region rows remain pending;
- 69 short-screen signals have been selected, but no new strict confirmation
  has been obtained;
- the only strict formation case remains Phi seq64 `lm_head dX`,
  `LOCAL_CENTERED -> PARAMETER_GRADIENT_BIASED`.

The attempted Mamba seq256 transport confirmation is an execution-blocked
extension of this matrix: its sequential CPU fallback was stopped after the
engineering state.  It has no scientific verdict and must not alter the
denominator or lower the confirmation criterion.

## Interpretation boundary

The final output may be a multi-mechanism map rather than one universal
property.  A valid result can therefore be:

* source bias in one family;
* transport/contract bias in another;
* optimizer bias if it is directly observed;
* variance-only controls where the local difference is erased or remains
  centered.

This preserves the distinction between **bias formation** and SEUP, which only
describes whether an already formed effective-update bias persists into a
parameter trajectory.
