# mamba-ssm 2.3.2.post1 selective state update review — 2026-10-19

The active SSU contract covers the complete value and state transition of one installed mamba-ssm 2.3.2.post1
`selective_state_update` call. Two installed distributions contained the same source file as the retained
`MAMBA232_SSU.py.gz`. The source fixes the library task as
`state_next = exp(dt*A)*state + dt*B*x`, followed by the grouped C projection, D*x, and an optional SiLU(z) gate.
The paper's full zero-order-hold B-bar formula is a different task and remains in the archived draft rather than a
second active reading.

The review found two details that materially change the reference. PyTorch softplus uses the logarithmic branch at
`dt <= 20` and the exact linear branch only above 20; the former adapter incorrectly described these as one exact
function. Also, although the installed function signature gives D and dt_bias default values of `None`, its starred
stride arguments select integer zero when either is absent and raise `TypeError` before a Triton launch. The actual
Mamba and Mamba2 inference call sites supply both tensors. The admitted profile therefore requires D and dt_bias and
refuses the failed optional calls. Optional z, headless input, contiguous B/C head groups, distinct state-slot
mapping and negative padding rows remain covered.

Eighteen contract examples cover the recurrence, updated-state output, threshold boundary, D, z, groups, complete
state pools, headless input, state mapping, padding and refused inputs. Eleven deliberately wrong formulas are
rejected. Seven independent 768-bit Arb element loops enclose every returned y and state coordinate, and the same
seven cases agree with a public PyTorch formula. The complete-output checker requires both y and the entire post-state
with float32 dtype and the declared shapes. Duplicate active state slots return `not_established` because the Triton
programs have no declared serial order.

The installed official Triton function and its installed reference helper were also compared on GPU for six callable
configurations covering headless and multi-head forms, one and several groups, softplus on and off, optional z, and
state sizes 3 through 16. Maximum absolute differences were `1.19e-7` for y and `1.19e-7` for state; every other
input stayed unchanged and all results were finite. Calls with absent D or dt_bias were separately reproduced as the
wrapper error above and are not counted as successful evidence.

This profile does not cover the paper's full ZOH discretisation, absent D/dt_bias calls in another fixed release,
duplicate live state slots, other dtypes, mixed-precision storage, sequence scans, gradients, layout or alias effects.
