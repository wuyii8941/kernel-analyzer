# Direct Triton MXFP8 E5M2 FLOOR binding review — 2026-10-19

This review covers `corpus/calls/triton_mxfp8_e5m2_floor.py` and
`corpus/call_bindings/triton_mxfp8_e5m2_floor.json`. The task is deliberately narrow: contiguous finite float32
input of shape `[4,64]`, final-dimension blocks of 32, E5M2 elements, ordinary E8M0 scale storage and the FLOOR
scale rule exposed by torchao 0.16.0. The binding maps the complete input, all task parameters, eight scale bytes,
all 256 FP8 element values and the unchanged input state. It does not select a task from the kernel name.

The task reference is the reviewed `F2_MX.torchao016_mxfp8_floor` profile. Its scale calculation was reproduced
independently from float32 exponent bits, and its E5M2 values were independently enumerated with exact rationals.
Fourteen f32/bf16 E4M3/E5M2 cases then matched the installed torchao 0.16.0 `to_mx` output byte-for-byte for scale
codes and value-for-value for elements. Those comparisons include normal values, two blocks, zero, fixed random
inputs and the E8M0 code-0 boundary. The installed torchao source is retained under `sources/TORCHAO016_*`; it is
not imported by the independent expected-value tests.

The formal safe-range run used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable.
All 6,144 element coordinates and 192 scale-code coordinates had complete exact task references; repeated launches
were bitwise identical and every device-to-task residual was zero. The statistical rules correctly withhold a
population-level no-bias claim because no equivalence bound was declared and every observed residual is zero. The
report is
`.cache/spec_trial_20261019_mxfp8/triton_mxfp8_e5m2_floor/safe_class_verdict_report.json`.

The full declared range gives a different result. All 192 scale codes still match, and all 5,781 finite element
comparisons have zero residual. In 363 of 6,144 element coordinates, however, the task result is finite while the
kernel writes positive or negative infinity. The analyzer reports this as a pointwise semantic contradiction between
finite/special-value classes and does not compute a misleading arithmetic mean containing infinities. The formal run
also reports differing element bytes between its paired launches and therefore withholds ordinary element
statistics; that unexplained execution finding remains visible. The report is
`.cache/spec_trial_20261019_mxfp8/triton_mxfp8_e5m2_floor/full_range_class_verdict_report.json`.

Separate fixed-input device evidence in `evidence/mx_torchao016/triton_a6000_20261019.json` starts each output from
two different byte patterns. Its safe case and zero case match completely, its chosen finite-saturation case has two
finite-to-infinity errors, and its code-0 nonzero case has 32 finite errors: torchao 0.16.0 returns element value 0.5
while this kernel writes 0. Both replays of every fixed case are identical. The retained Triton torch reference says
FP8 conversion must use finite saturation and clamps before conversion, so the infinity result is contrary to the
retained implementation's own stated conversion rule as well as to the selected torchao task.

The RTX A6000 cannot compile this retained kernel's E4M3 path because the device/compiler combination does not
support the requested FP8 type. E4M3 remains covered by the independent reference and installed torchao comparison,
but it has no device binding here. These results do not cover RCEIL, CEIL, EVEN, FP4, FP6, other axes or block sizes,
swizzled scales, special-value inputs, other devices or every OCP MX implementation.
