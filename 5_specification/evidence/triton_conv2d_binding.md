# Direct Triton Conv2d call binding review — 2026-10-18

This review covers `corpus/calls/triton_conv2d.py` and
`corpus/call_bindings/triton_conv2d.json`. It applies only to the declared
batched grouped float32 Conv2d call.

## Task correspondence

The wrapper accepts three contiguous CUDA float32 tensors and six explicit
task parameters. It fixes input shape 2×4×8×9, weight shape 6×2×3×3, one bias
per output channel, stride 1×2, symmetric padding 1×1, dilation 1×1, two
groups, a non-transposed operation, and zero output padding.

The kernel decodes each program id into batch, output channel, output row and
output column. It selects the input-channel group from the output channel,
starts with the corresponding bias, enumerates every local input channel and
3×3 kernel coordinate, loads zero outside the padded input, and accumulates
`x[n, ci, ih, iw] * weight[co, local_ci, kh, kw]`. This is the retained
PyTorch Conv2d cross-correlation formula with exactly the parameters mapped by
the binding plan. The correspondence follows from the code and retained task
definition, rather than from agreement with its output.

The admitted CONV `real` profile interprets each finite received float32 value
as an exact real number. Device multiplication, accumulation order and final
float32 rounding remain in the program-numerics comparison.

## Complete boundary and measured evidence

The plan maps all three tensor pre-states, all six scalar or list parameters,
the complete public output, and requires every tensor input to remain
unchanged. The wrapper has no tensor parameters, buffers or closed-over tensor
state. Its 480 program ids each write one distinct coordinate of the complete
2×6×8×5 output.

Before retention, a temporary reviewed copy was measured on eight development
and sixteen confirmation inputs. All 24 calls used the admitted CONV reference
and had one completely covered TTIR launch. The three loads were traced to the
declared input, weight and bias; no hidden source, missing write, post-write
modification, changed input or failure class was reported. Repeated executions
were bitwise identical.

The strict reference covered all 11,520 sampled coordinates and met the
declared resolution. The task residual excluded zero at no coordinate, with
projection means around `1e-17`. Neither the program-numerics comparison nor
complete `K-f` confirmed a nonzero mean under R1, R2, R3 or R5 for this input
population. The temporary report is
`.cache/spec_trial_20261018_diverse/triton_conv2d_controlled_20261018/baseline_report.json`.

A controlled wrapper then changed only the store to `acc + 0.01`. Its admitted
task and all binding fields remained the same. All 11,520 task-residual
coordinates excluded zero, the mean absolute task residual was about
`0.0099999998`, and R1 and R5 confirmed nonzero means. The program-numerics
comparison still confirmed no mean under any declared rule. This shows that
the added task term is separated from ordinary float32 convolution effects.
The report is
`.cache/spec_trial_20261018_diverse/triton_conv2d_controlled_20261018/injected_report.json`.

After retention, the repository declaration independently passed the same
formal entry without a temporary registry override. All 24 production calls,
11,520 coordinates, three declared tensor sources and complete TTIR writes
passed with no failure class. The task residual excluded zero at no coordinate;
R1, R2, R3 and R5 confirmed no nonzero mean for `K-G_pi`, `G_pi-f` or complete
`K-f` on this population. The run took about 257 seconds. Its report is
`.cache/spec_trial_20261018_diverse/triton_conv2d_controlled_20261018/production_reviewed_report_20261018.json`.

## Scope and limits

This binding does not cover other shapes, strides, padding, dilation, groups,
dtypes, layouts, 1-D/3-D or transposed convolution, backward convolution, or a
library convolution lowered through cuDNN. Each requires its own complete call
correspondence. Non-detection on this normal-input population is not proof of
absence of bias for every valid input.
