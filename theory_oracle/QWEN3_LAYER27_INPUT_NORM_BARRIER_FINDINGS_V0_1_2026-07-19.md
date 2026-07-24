# Qwen3 layer-27 input-RMSNorm barrier findings v0.1

## Result

All fixed-boundary treatment-integrity gates passed.  Repeats were exact, the
eager fixed-boundary arm reproduced the frozen eager anchor, the requested
compiled/eager target modes were each invoked twice, and switching target mode
did not recompile the outer body.

Both injection and repair found a nonzero selected-token log-probability effect.
Their L2 magnitudes were approximately `0.0063047`, with opposite signed means
under the opposite contrast orientation.  Thus compiling this RMSNorm invocation
has an identifiable effect **inside the fixed-boundary program** at this state.

## Critical limitation

The barrier candidate hash did not equal the frozen original candidate hash.
Therefore this experiment does not establish that the layer-27 input RMSNorm is
a cause of the discrepancy in the original whole-compiled implementation.  Its
coverage state is `BARRIER_CONDITIONED`, not `VALID_EFFECT` for the original
candidate.

This separates two quantities that naive repair confounds:

- target effect with the intervention boundary held fixed;
- disturbance caused by introducing that boundary and changing fusion context.

The result supports the feasibility of intervention-dependent operator
attribution, while showing that candidate-equivalent treatment remains the main
unsolved requirement for original-program root-cause attribution.
