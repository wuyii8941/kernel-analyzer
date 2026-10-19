# Public RAdam implementation review — 2026-10-19

The active RADAM contract covers one public PyTorch 2.10 `torch.optim.RAdam` step for one dense real parameter tensor.
It includes the complete parameter, first moment, second moment and step before and after the call. `m=[]`, `v=[]`,
`t=null` represents the actual uninitialized per-parameter state. A null gradient creates no state and leaves an
initialized state unchanged.

The retained PyTorch 2.10 page fixes the documented update and the strict `rho_t > 5` branch choice. The retained
2.10 source separately fixes constructor validation, lazy state keys, maximize and decay order, step increment and
loop/foreach execution choices. `decoupled_weight_decay` is an explicit public option: coupled mode accumulates decay
in both moments, while decoupled mode changes the parameter base and leaves the moments free of decay. Capturable or
differentiable execution, tensor learning rates, sparse or complex gradients, closures, internal functional calls and
multi-parameter orchestration are outside the reviewed profile.

The previous draft required an already initialized state and treated the broad name RAdam as sufficient to choose a
threshold and decay convention. It remains under `contracts/draft`. The reviewed contract is explicitly the
versioned PyTorch task and does not generalize its threshold to every algorithm named RAdam. The adapter represents
moments and the unrectified branch exactly and uses outward real enclosures for the square roots on the rectified
branch. Its complete-output checker requires all four fields and verifies the selected update relation without a
numerical tolerance. Exact step exponents above 10,000 are refused as `not_established` rather than silently rounded.

Twenty-two examples cover absent and warm state, both sides of the branch boundary, coupled and decoupled decay,
maximize, null versus zero gradients, execution options and malformed calls. Eleven deliberately wrong formulas are
rejected. An independent coordinate implementation and PyTorch 2.10 agree in 72 combinations: four floating dtypes,
six state/option cases and loop, foreach and default strategies. Separate checks establish `rho_5 < 5`, `rho_6 > 5`,
the exact public state keys and null-gradient behavior.

The official page links the RAdam paper, but the required `localhost:27891` proxy did not return it during this
review. No active claim is inferred from an unavailable paper: the selected task and every enabled decision rest on
the retained PyTorch 2.10 documentation and source.
