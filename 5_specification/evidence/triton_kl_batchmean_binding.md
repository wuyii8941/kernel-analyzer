# Direct Triton KL batchmean binding review — 2026-10-19

This review covers `corpus/calls/triton_kl_batchmean.py` and
`corpus/call_bindings/triton_kl_batchmean.json`. It applies only to the declared contiguous 8x64 float32 call and
its sampled input population.

## Task correspondence

The wrapper accepts exactly two CUDA float32 matrices. `input` is finite and lies in `[-16,16]`; `target` is finite
and lies in `(0,4]`. Both have shape 8x64 and contiguous storage. One Triton launch reads all 512 pairs, computes
`target * (log(target) - input)`, sums all pointwise terms and divides by the batch size 8. The same launch writes
every input-VJP coordinate as `-target/8`. The wrapper returns the scalar loss and the complete 8x64 gradient.

The binding fixes `reduction="batchmean"`, `log_target=false`, float32 input and target, and scalar upstream one. It
maps both full input pre-states, both full public outputs, and the unchanged post-state of both inputs. There are no
other tensor arguments, hidden buffers or public outputs. This is therefore the selected PyTorch 2.10 `DIVERG`
profile, rather than a name-based match to an arbitrary KL kernel.

## Device and formal checks

A direct GPU 3 check compared the Triton call with public PyTorch 2.10 `torch.nn.functional.kl_div` and autograd on
the same physical inputs. The scalar loss and every gradient coordinate were equal in that run, output shapes and
dtypes were complete, and both inputs remained unchanged.

The formal contract entry evaluated 24 independently sampled calls. Every call retained the reviewed binding, used
a complete call-level program reference, captured the only launch, checked both unchanged inputs, and resolved at
precision level 1. All 24 scalar loss values and 12,288 gradient coordinates had complete finite references.
Repeated launches were bitwise identical.

For the normal candidate, no task-formula interval excluded zero. The mean absolute `G_pi-f` residual was about
`5.92e-16` for the scalar loss and exactly zero for the gradient; the corresponding maxima were about `1.78e-15`
and zero. The loss device-to-program mean absolute residual was about `6.19e-7`, while the gradient result was exact.
The report correctly leaves the statistical bias question unconfirmed for the scalar loss and reports a degenerate
zero-difference sample for the exact gradient, rather than claiming a universal absence of bias. The report is
`.cache/spec_trial_20261019_kl/production_report_20261019.json`.

A controlled copy added `0.25` to both public formulas. The program-to-task comparison then excluded zero for all
24 loss values and all 12,288 gradient coordinates. Its mean and maximum absolute task residual were `0.25` for both
outputs. All four loss projections and the complete candidate-to-task gradient projections confirmed a nonzero
effect; the ordinary device-to-program residuals remained around `6.35e-7` for loss and `1.84e-9` for gradients.
The report is `.cache/spec_trial_20261019_kl/injected_report_20261019.json`. Temporary admission copies used only for
this controlled run were removed after measurement.

## Scope

This binding does not cover other shapes, strides, dtypes, reductions, log-space targets, target gradients,
nonpositive targets, special values, different upstreams or fused consumers. The result supports this complete call
under the declared population; it does not certify every kernel whose name contains `kl` or `divergence`.
