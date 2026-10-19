# Public Adam implementation review — 2026-10-19

The active ADAM contract covers one public PyTorch 2.10 `torch.optim.Adam` step for one dense real parameter tensor
with coupled L2 weight decay. It includes the complete parameter, first moment, second moment, optional AMSGrad
maximum and step before and after the call. `m=[]`, `v=[]`, `vmax=[]`, `t=null` represents the actual uninitialized
per-parameter state; `vmax` is absent when AMSGrad is disabled. A null gradient leaves absent or initialized state
unchanged.

The retained PyTorch 2.10 page fixes the documented formula. The retained 2.10 source separately fixes public state
creation and keys, option validation and execution order: maximize first, step increment, coupled decay, moment
updates, correction at the new step, AMSGrad maximum over the uncorrected second moment, and eps outside the square
root. `decoupled_weight_decay=true` is refused here because it selects AdamW. Internal fused functions, AMP inputs,
capturable or differentiable execution, sparse or complex gradients, closures and multi-parameter orchestration are
outside this contract.

The previous draft combined Adam, AdamW and internal fused calls and always returned a `vmax` vector even when the
public optimizer would have no such state. That draft remains under `contracts/draft`. The reviewed adapter now
implements the selected formula directly. Moments and step are exact rationals; parameter values use outward real
enclosures. Its complete-output checker requires every state field and verifies the square-root update relation
without a numerical tolerance.

Twenty-two contract examples cover cold and warm state, AMSGrad, maximize, coupled decay, null versus zero gradients,
strategy options and malformed or unsupported calls. Twelve deliberately wrong formulas are rejected. An independent
coordinate implementation and PyTorch 2.10 agree in 80 combinations: four floating dtypes, five state/option cases,
and loop, foreach, default and fused strategies. Separate checks confirm that a null gradient creates no state and
that disabling AMSGrad leaves `max_exp_avg_sq` absent.

The public documentation links the Adam and AMSGrad papers. They were not newly fetched during this review because
the required localhost proxy was unavailable; no claim in the active contract depends on an unretained paper text.
