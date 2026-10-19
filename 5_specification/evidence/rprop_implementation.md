# Public Rprop implementation review — 2026-10-19

The active RPROP contract covers one public PyTorch 2.10 `torch.optim.Rprop` step for one nonempty dense real
parameter. It includes the complete parameter and optimizer state before and after the call. Empty `prev` and `eta`
with `t=null` represent the actual uninitialized per-parameter state. A null gradient creates no state and leaves
initialized state unchanged.

The retained PyTorch 2.10 page fixes the sign branches, step-size factors, bounds, parameter update and maximize
task. The retained source fixes constructor validation, lazy state keys and the loop and foreach execution paths. It
also resolves a boundary detail that the documentation's algorithm box does not fully describe: the public source
multiplies and clamps every coordinate, including the zero-product branch whose factor is one. This matters when a
loaded step size, or the initial learning rate, lies outside the selected bounds. The contract records this as a
versioned PyTorch execution choice rather than a claim about every Rprop algorithm.

PyTorch 2.10 accepts some negative or unordered `step_sizes` tuples and then applies its underlying clamp operation.
Those configurations can produce unusual results, such as a negative step size when the declared minimum exceeds the
maximum. The reviewed profile does not invent a mathematical interpretation for them. It requires
`0 <= step_min <= step_max` and returns unsupported outside that range. Capturable or differentiable execution,
sparse or complex values, tensor learning rates, internal functional calls and multiple parameters are also outside
the reviewed profile.

Twenty examples cover cold and warm state, all three sign products, both clamps, maximize, null and zero gradients,
arbitrary positive tensor ranks, execution choices, malformed state and refused configurations. Ten deliberately
wrong readings are rejected. A separate NumPy formula and public PyTorch 2.10 agree in 60 combinations: four floating
dtypes, five state and option cases and loop, foreach and default execution. Additional checks verify exact public
state keys and shapes, null-gradient behavior, zero-product clamping and rejection of missing, wrong-shape or
wrong-dtype outputs.

The strict reference uses exact rational products, comparisons, clamps and updates. Device rounding is kept out of the
task answer and measured separately. The PyTorch page links the original RPROP paper, but the required
`localhost:27891` proxy could not resolve during this review. No active claim rests on the unavailable paper; the
selected public task and enabled decisions rest on the retained PyTorch 2.10 documentation and source.
