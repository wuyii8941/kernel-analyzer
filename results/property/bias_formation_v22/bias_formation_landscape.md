# Bias Formation Landscape v2.2

## Corrected interpretation

The old v2.1 global cross-state gate is only a `GLOBAL` observation.  It is not
a necessary condition for training bias.  Existing paired trajectories were
therefore reaudited with a basis-free parameter-separation criterion:

```text
complete F+B candidate/repair + exact sham
→ live parameter separation grows
→ trajectory-level causal separation
```

No fixed carrier, monotone projection, or common sign across unrelated states
is required at this observation layer.  This reclassification does not itself
identify P1--P6 or establish an oracle property.

## Current trajectory-level population

The strict artifact audit finds eight complete trajectory artifacts, eight
semantic cases, and seven mechanism-family clusters. One additional layer-23
key live-weight artifact is explicitly excluded because its same-weight sham
is not exact:

| semantic mechanism family | cases | count |
|---|---|---:|
| fused accumulation/loss | Liger fused CE | 1 |
| loss-head transport | Phi seq64 lm_head dX | 1 |
| GEMM accumulation | Qwen seq64/128 v_proj | 2 |
| saved-state softmax transport | Qwen saved-P seq128 | 1 |
| nonlinear backward | Qwen3-VL SiLU invocation | 1 |
| recurrent input projection | Mamba seq64 input projection | 1 |
| attention-state transport | Qwen layer-23 q-projection region | 1 |

The two Qwen v_proj rows are two shape/trajectory instances of one candidate
mechanism family, not two independent physical mechanisms.  The excluded
layer-23 key repair is not a complete trajectory case because its same-weight
sham fails exactness at every step.  The same caution applies to any later
grouping by operator name or model.

## What this fixes

Qwen seq128 v_proj and Qwen3-VL SiLU previously failed a step-1 fixed-direction
gate. Symmetric four-counterfactual recurrence now resolves them differently:
Qwen128's aligned rounding-only local effect is diffusive/canceling, whereas
SiLU's persistent separation is sustained by closed-loop feedback after a
local trigger. They therefore supply a persistence-negative boundary and a
second persistence regime, respectively; neither is silently promoted into a
Flash-style persistent-local-source mechanism.

## What remains to be measured

The old artifacts do not contain preregistered repeated condition strata, so
the `CONDITIONAL` level is not retroactively claimed.  Future formation runs
must group repeated atoms/states by a condition fixed before inspecting the
candidate residual.  They must then test the original mechanism candidates:

* P1 conditional source asymmetry;
* P2 source--transport alignment;
* P3 forward/backward numerical-contract consistency;
* P4 nonlinear rectification;
* P5 optimizer rectification;
* P6 semantic-orbit centering.

SEUP remains the persistence/consequence layer after formation.  The present
landscape supplies a positive trajectory population and formation-level
negative controls needed to ask which of P1--P6 predicts entry into trajectory
bias; it does not answer that property question by itself.

The current negative evidence is formation-level rather than trajectory-level:
Qwen CE/lm-head seq64/128/256 are centered at local, gradient, and update
layers; normalization writeback controls erase their internal variation; and
bmm remains a centered/hard-negative boundary.  These controls show that an
implementation difference is not automatically a trajectory case.  A future
trajectory negative must use the same basis-free separation certificate, not
inherit the old fixed-carrier failure label.
