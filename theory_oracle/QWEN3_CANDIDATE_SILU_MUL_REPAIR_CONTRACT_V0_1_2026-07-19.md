# Qwen3 original-candidate fused SiLU-multiply repair contract v0.1

## Subject

The frozen original compiled Qwen3 forward contains 28 calls to
`triton_poi_fused__unsafe_view_mul_silu_14`, one per decoder MLP.  The generated
kernel mutates the gate-projection buffer with fused `SiLU(gate) * up`.

Test calls 0, 14 and 27 as the predeclared early, middle and late positions.  At
the selected live generated-kernel invocation, replace only `.run` with two
eager PyTorch tensor operations: FP16 SiLU materialization followed by FP16
multiplication into the existing output buffer.  All other candidate graph,
fusion, layout, specialization and generated calls remain unchanged.

## Endpoints

Report candidate-to-repair effect, eager-to-repair distance, L2 distance change,
fractional L2 reduction and repair-vector alignment with candidate-to-eager.
Zero, toward-eager, away-from-eager and orthogonal results are all admissible.

## Fail-closed gates

- Eager and candidate reproduce their frozen anchors twice.
- The observed Dynamo graph family is exact.
- The live module contains the exact generated kernel.
- Every repaired run sees 28 family calls and repairs exactly one selected call.
- All arms repeat exactly, no repair triggers backend compilation, and restoring
  the kernel reproduces the candidate anchor.

## Claim limits

A passing result attributes an implementation-relative effect to the selected
fused generated-kernel invocation at one state.  It cannot separate SiLU from
multiplication, prove injection sufficiency, estimate a population effect, or
declare eager correct.
