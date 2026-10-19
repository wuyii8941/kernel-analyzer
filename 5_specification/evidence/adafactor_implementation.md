# Public Adafactor implementation review — 2026-10-19

The active ADAFACTOR contract covers one public PyTorch 2.10 `torch.optim.Adafactor` step for one dense real
parameter. Separate profiles cover rank-1 vector state and rank-at-least-2 factored state. Both include the complete
parameter and optimizer state before and after the call. Empty row, column and variance fields with `t=null` represent
the actual uninitialized per-parameter state. A null gradient creates no state and leaves initialized state unchanged.

The retained PyTorch 2.10 page fixes the schedule, parameter scaling, clipping, maximize and decoupled weight-decay
task. The retained source fixes constructor validation, lazy state keys, the two execution strategies and the exact
factored implementation. In particular, `row_var` keeps the last dimension at size one, `col_var` keeps the
penultimate dimension at size one, leading dimensions are independent batches, and `eps1` clamps both the row
normalizer and the final variance estimate. `eps1=None` resolves to `torch.finfo(parameter.dtype).eps`. The previous
draft squeezed state axes, required initialized state, combined a different Transformers algorithm into the same
contract, and treated an active row-normalizer clamp as undefined. That version remains under `contracts/draft`.

Twenty-one examples cover cold and warm vector and factored state, rank-three batching, both epsilon clamps,
dtype-resolved epsilon, clipping, maximize, decay, null and zero gradients, execution choices, malformed state and the
reference budget. Twelve deliberately wrong readings are rejected. A separate NumPy formula and public PyTorch 2.10
agree in 48 combinations: four floating dtypes, four shape/state/option cases and loop, foreach and default execution.
Additional checks verify exact public state keys and shapes, null-gradient behavior and complete-output rejection of a
missing field or squeezed state axis.

The strict reference uses exact rationals where possible and outward real enclosures for noninteger step powers and
square roots. It refuses step powers above 10,000 rather than silently rounding. Sparse or complex values, tensor
learning rates, closures, internal functional calls, multiple parameters and device rounding are outside the reviewed
profiles.

The official page links the Adafactor paper, but the required `localhost:27891` proxy refused the connection during
this review. No active formula claim rests on the unavailable paper. The selected task and all enabled decisions rest
on the retained PyTorch 2.10 documentation and source.
