# Qwen3 original-candidate masked safe-softmax repair contract v0.1

## Subject

The frozen original compiled Qwen3 forward contains 28 calls to
`triton_red_fused__safe_softmax_add_prepare_softmax_online_view_8`, one per
decoder attention block.  It fuses addition of the FP16 attention mask with the
FP32 score tensor and the safe-softmax reduction.

Test calls 0, 14 and 27 as predeclared early, middle and late positions.  Replace
only the selected live generated-kernel `.run` with eager tensor mask addition
followed by `aten._safe_softmax`, writing into the same score buffer.  Preserve
the rest of the original candidate graph and generated execution.

## Endpoints and gates

Report candidate-to-repair effect and direction/distance relative to eager.
Zero and either directional sign are valid observations.  Require exact eager
and candidate anchors, exact graph family, exact live kernel resolution, 28
observed calls with one selected repair, exact repeats, no repair-time backend
compilation and exact candidate restoration.

## Claim limits

A valid result belongs to the fused mask-add plus safe-softmax generated
invocation.  It does not separate mask addition, max/sum reduction, exponent,
division or safe all-masked-row handling.  It is one-state, repair-only,
implementation-relative evidence and provides no correctness authority.
