# Public RMSprop implementation review — 2026-10-19

The active RMSPROP contract covers one public PyTorch 2.10 `torch.optim.RMSprop` step for one dense real parameter
tensor. It includes the complete parameter, square average, optional centered gradient average, optional momentum
buffer and step before and after the call. `v=[]`, `gave=[]`, `b=[]`, `t=null` represents uninitialized state.
`gave` exists only with centered mode and `b` only with positive momentum; a null gradient creates or changes no state.

The retained PyTorch 2.10 page fixes the square average, centered estimate, eps position and momentum update. The
retained 2.10 source separately fixes maximize before coupled decay, lazy state keys, step increment, constructor
checks and loop/foreach execution choices. The public constructor only rejects negative alpha, so the contract does
not invent an upper bound; inputs whose resulting square estimate is negative remain outside the finite-real domain.
Capturable or differentiable execution, sparse or complex gradients, tensor learning rates, closures and
multi-parameter orchestration are outside the reviewed profile.

The previous draft always returned `gave` and `b`, even when PyTorch would not allocate those keys. It also delegated
the reference to the older shared optimizer model and had no complete-output checker. The reviewed adapter implements
the selected formula directly. Square and centered averages and step are exact; parameter and momentum-buffer values
are outward real enclosures. The checker requires every state field and verifies the square-root relations without a
tolerance.

Twenty-two contract examples cover cold and warm state, centered mode, momentum, maximize, decay, null versus zero
gradients, alpha outside the conventional interval, public strategy options and malformed calls. Twelve wrong
formulas are rejected. An independent coordinate implementation and PyTorch 2.10 agree in 60 combinations covering
four floating dtypes, five state/option cases and loop, foreach and default strategies. Separate checks confirm that
a null gradient creates no state and preserves initialized state.
