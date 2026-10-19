# Direct Triton float32 SGD binding review — 2026-10-19

This review covers `corpus/calls/triton_sgd.py` and `corpus/call_bindings/triton_sgd.json`. It applies to one
contiguous float32 parameter, gradient and existing momentum buffer, each with 4,096 finite coordinates. The fixed
public SGD choices are lr 0.125, momentum 0.75, dampening 0, weight decay 0.25, Nesterov enabled, maximize enabled,
and non-differentiable non-foreach/non-fused execution.

One Triton launch uses sixteen programs. Every program reads the old parameter, gradient and old buffer for 256
consecutive coordinates, then writes the new parameter and new buffer in place. The wrapper validates every tensor's
shape, dtype, device, layout and finite range. The binding maps all three complete input tensors, maps both complete
post-states, declares both mutations, and declares the gradient unchanged. A GPU 3 sanity run confirmed both returned
views alias their mutated inputs and the gradient remains bitwise unchanged.

The formal baseline used 8 development and 16 confirmation inputs. All 24 bindings were reviewed and usable, and both
outputs had complete call-level references for 98,304 coordinates. For the parameter and buffer, every program-to-task
coordinate was exactly zero: the Triton program formula therefore agrees with the admitted SGD task. Device execution
had ordinary float32 effects: mean absolute K−f was about `1.91e-8` for the parameter and `2.92e-8` for the buffer,
with maxima about `2.07e-7` and `3.56e-7`. The buffer's aligned rules detected a small numerical direction even though
Gπ−f was exactly zero. The report is `.cache/spec_trial_20261019_sgd/triton_sgd/baseline_report.json`.

A temporary controlled candidate preserved the correct parameter update and added 0.25 only when writing the momentum
buffer. All 98,304 buffer Gπ−f coordinates excluded zero with absolute difference exactly 0.25, while every parameter
Gπ−f coordinate remained zero. This demonstrates that checking only the public parameter output would miss a wrong
optimizer state. The temporary candidate and plan were removed after the run; its report is
`.cache/spec_trial_20261019_sgd/triton_sgd/controlled_buffer_plus_quarter_report.json`.

This binding does not cover an absent momentum buffer, another option combination, multiple parameters, sparse or
complex gradients, AMP overflow skipping, differentiable execution, another shape or dtype, or another kernel with a
similar name.
