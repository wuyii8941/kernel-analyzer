# Direct Triton row index-select binding review — 2026-10-18

This review covers `corpus/calls/triton_index_select.py` and
`corpus/call_bindings/triton_index_select.json`. It applies only to one contiguous physical finite float32 input of
shape `[128, 32]`, one contiguous int64 index of shape `[64]` with values in `[0, 128)`, and the complete contiguous
output of shape `[64, 32]`.

## Task correspondence

The wrapper accepts exactly the two declared inputs, checks dtype, shape, layout, source finiteness and index range,
allocates one complete output, and returns no other value. One launch uses eight Triton programs. For each of the
2,048 output coordinates, the kernel loads its row index, reads the corresponding source row and column, and writes
that coordinate once. The plan binds both complete input pre-states to the reviewed INDEX `aten.index_select` task on
dimension zero, binds the complete returned tensor, and requires both inputs to remain unchanged.

The INDEX contract derives the call schema, coordinate rule and result shape from retained PyTorch 2.10 schemas,
documentation and source. The binding chooses that task explicitly; no task is inferred from the kernel name. A GPU 3
sanity run was bit identical to `torch.index_select` and left both inputs unchanged. PyTorch was used only as a
separate implementation check; the formal task value came from the reviewed INDEX evaluator.

## Formal measurement

The reviewed run used 8 development inputs and 16 confirmation inputs, giving 49,152 output coordinates. Every
binding was admitted, the program reference was complete at every coordinate, and repeated launches were bitwise
identical. The baseline had exact zero `K-G_pi`, `G_pi-f` and complete `K-f` residuals throughout. Statistical rules
were degenerate on the all-zero sample and therefore did not claim population equivalence. The report is
`.cache/spec_trial_20261018_diverse/triton_index_select_controlled_20261018/baseline_report.json`.

A controlled candidate added `0.25` to every selected value while the bound task remained read-only row selection.
All 49,152 `G_pi-f` coordinates excluded zero with mean and maximum absolute task residual `0.25`. `K-G_pi` stayed at
floating rounding scale: mean absolute residual about `5.43e-9` and maximum about `2.38e-7`. Complete `K-f` had mean
absolute residual about `0.25`, and the fixed complete-output rule family confirmed a nonzero average effect. The
temporary candidate and binding were removed after the run. Its report is
`.cache/spec_trial_20261018_diverse/triton_index_select_controlled_20261018/controlled_plus_quarter_report.json`.

## Scope

This binding does not cover other shapes, dtypes, layouts, nonfinite source values, invalid or non-int64 indices,
gather with a full index tensor, embedding options, scatter, updates, gradients, aliasing, fused surrounding
operations, or another Triton kernel with a similar name.
