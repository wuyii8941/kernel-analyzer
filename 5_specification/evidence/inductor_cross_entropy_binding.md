# Inductor CrossEntropy call binding review — 2026-10-18

This review covers `corpus/calls/inductor_cross_entropy.py` and
`corpus/call_bindings/inductor_cross_entropy.json`. It applies only to the
declared rank-2 mean-loss call and its complete logits VJP.

## Task correspondence

The wrapper receives exactly seven named inputs. It requires rank-2 floating
`logits`, rank-1 int64 `target` with the same row count, no class weights,
zero label smoothing, `ignore_index=-100`, reduction `mean`, and scalar
upstream one. It returns the scalar loss and every coordinate of the logits
gradient from one compiled region.

An independent `make_fx` trace on the installed PyTorch 2.10.0+cu128 package
shows this public expression as:

1. `_log_softmax(logits, 1, False)`;
2. `nll_loss_forward(..., target, None, mean, -100)`;
3. a scalar `ones_like(loss)` cotangent;
4. `nll_loss_backward` followed by `_log_softmax_backward_data`;
5. return of the loss and complete logits gradient.

This is the CE contract's `index_unsmoothed` task with the exact parameters
listed in the plan. The correspondence comes from the public call, its fixed
arguments, and the traced ATen graph; it is not inferred from agreement with
the Triton output.

## Complete boundary

The plan maps all seven received inputs. Both tensor inputs are mapped from
their complete pre-state; all five scalar or null task parameters are mapped
explicitly. It maps both public outputs in full and requires both input tensors
to remain unchanged. The wrapper has no tensor parameters, buffers, or
closed-over tensor state.

The earlier isolated measurement used the identical current wrapper and the
same task mapping, with only the test binding identity and relative file paths
changed. Across eight development and sixteen confirmation inputs it captured
the complete compiled region and established an admitted CE reference for the
loss and all 184 gradient coordinates. A controlled `loss+0.25` version changed
only the loss task residual by about `-0.25000004`; the gradient result remained
unchanged. This supports output separation but does not itself approve the
current production files.

After retention, the production declaration must still pass the formal entry
without a registry override. Each sampled call must independently establish
the installed package version, all Triton launches, both input pre/post-states,
all output coordinates, and absence of undeclared tensor sources. The retained
measurement report records the results of that check.

The production run passed all of these checks for eight development and sixteen
confirmation inputs. Every call used the reviewed binding and an admitted CE
reference. All three captured TTIR launches had complete IR coverage. The loss
had 24/24 complete call-level reference values; the gradient had 4,416/4,416
complete coordinates. Neither output had an unwritten coordinate, a changed
post-write value, an undeclared tensor source, an unresolved enclosure, or a
failure class. The task residual excluded zero at no coordinate. Under the
declared rules, neither `K-G_pi` nor complete `K-f` confirmed a nonzero mean for
the loss or gradient on the sixteen confirmation units.

## Scope and limits

The measurement declaration uses finite f32 logits of shape `[8, 23]` and
targets drawn from classes `[0, 23)`. It therefore does not exercise ignored
rows, a zero mean denominator, class weights, label smoothing, probability
targets, other reductions, other upstream values, or other ranks and dtypes.
Those are not certified by this call binding. Other CrossEntropy wrappers and
generated kernels require their own explicit correspondence.

One repeated-input population experiment can detect a bias under its declared
distribution; it cannot prove absence of bias for every valid input.
