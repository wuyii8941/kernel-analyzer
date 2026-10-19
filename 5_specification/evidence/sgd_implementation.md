# Public SGD implementation review — 2026-10-19

The active SGD contract covers one public PyTorch 2.10 `torch.optim.SGD` step on one dense real parameter tensor. Its
input includes the complete parameter and momentum-buffer pre-state, a complete gradient or an explicit absent
gradient, every algorithmic hyperparameter, and the public `foreach`, `fused` and `differentiable` choices. Its output
contains both the complete parameter and complete momentum-buffer post-state.

The retained PyTorch 2.10 page fixes the update order: optional gradient negation for maximize, coupled L2 weight
decay, momentum-buffer update, optional Nesterov direction, and the final parameter update. It also documents that the
first momentum buffer is the processed gradient, without dampening, and that learning rate is outside the buffer. The
retained 2.10 source independently shows that a null gradient skips the parameter, that an existing buffer is retained
when momentum is zero, and that the constructor rejects Nesterov without positive momentum and zero dampening.

The former draft also named internal `aten._fused_sgd_`, silently ignored undeclared fields and had no complete-output
checker. It remains under `contracts/draft`. The reviewed profile only names the public optimizer task. It refuses
internal AMP `grad_scale`/`found_inf`, closures, sparse and complex gradients, differentiable execution and
multi-parameter orchestration rather than guessing their semantics.

The reviewed adapter implements the formula directly with exact rational arithmetic. Twenty-one contract examples
exercise first and warm buffers, dampening, Nesterov, maximize, weight decay, null versus zero gradients, public
strategy options and malformed or unsupported calls. Eleven deliberately incorrect formulas are rejected. An
independent coordinate implementation and PyTorch 2.10 agree in 80 combinations covering f16, bf16, f32 and f64;
plain, decay/maximize, first-buffer, warm dampening and Nesterov cases; and loop, foreach, default and fused strategies.
