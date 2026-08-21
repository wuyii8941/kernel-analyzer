# Persistence property v1: development result

## Result

The surviving candidate is **transported conditional-mean persistence**:

> An implementation difference produces source-persistent parameter drift when
> its conditional mean over a semantics-preserving arithmetic orbit has a
> temporally non-canceling component, and the real F+B/optimizer transport
> preserves or amplifies that component. Orbit variance without this transported
> temporal mean is diffusive under the tested protocol.

For an endpoint residual $\epsilon_{t,\pi}$ under equivalent arithmetic
variants $\pi$, define

$$m_t=\mathbb{E}_\pi[\epsilon_{t,\pi}],\qquad
q_{t,\pi}=\epsilon_{t,\pi}-m_t.$$

The property is not $\|m_t\|$ alone. It is the ordered persistence of the
transported mean $M_t m_t$, where $M_t$ is the actual reference F+B and optimizer
map. This separates bias generation from the later feedback term.

## Decisive development pair

| Case | orbit-mean energy | source $A$ | effective-update $A$ |
|---|---:|---:|---:|
| Phi lm_head dX | 0.923817 | 1.442566 | 3.289092 |
| Qwen128 v_proj | 0.999863 | 1.018049 | 0.999092 |

Qwen has the larger orbit-mean energy fraction but no effective-update
persistence. Therefore orbit-mean magnitude is falsified as the property.
Phi has a small but significant temporally shared source component, which its
backward transport amplifies; Qwen's source directions largely re-draw across
states.

## Matched interventions on the real Phi backward MM

Randomizing the exact GEMM K-axis ordering preserves real $GW$ semantics. It
changed amplification from
`3.325296` to
`3.112275`
(ratio `0.935939`). Fixed schedule is therefore not
the main anchor: most bias is shared by the reduction orbit.

Replacing deterministic low-precision materialization by unbiased stochastic
rounding produced four amplifications `0.982547, 1.018562, 0.917381, 0.903571`;
their mean/natural ratio is `0.287347`.
The SR endpoint residual norm is larger than the deterministic endpoint
difference, so the loss of persistence cannot be explained by smaller error.

## Feedback boundary

`A_B>1` alone is not a feedback mechanism certificate. The bmm hard control has
feedback amplification `2.572634`. Saved-P
does not exceed that floor, Qwen128 exceeds it by only about 7%, and SiLU by
about 54%; all remain unresolved until an RMS-matched perturbation null is run.

## Current claim and boundary

Supported now:

* semantic-orbit mean magnitude alone is not predictive;
* fixed reduction order is not necessary for Phi persistence;
* removing conditional mean with SR suppresses Phi persistence while increasing
  local error magnitude;
* source temporal structure and F+B transport are both required in the observed
  Phi/Qwen pair.

Not yet supported:

* universality across operator families;
* a source-free predictor for arbitrary kernels;
* feedback-sustained bias as a distinct mechanism;
* long-horizon loss failure (M7).

## New-architecture stress control: OLMoE MoE routing

The OLMoE router/expert path was not used to select or tune the predictor. It
was measured afterwards as a genuinely different semantic family. A strict
expert-order orbit (default plus seven non-default orders) preserves the
real-valued expert sum and the exact routing decisions. Across 16 states, the
transported orbit mean was not directional in the router-gate slice
(`A=1.0036`, 4+4 cross-fit `A=1.0037`) or in the layer-0 attention slice
(`A=0.9832`, cross-fit `0.9635`). A small final-RMSNorm excess did not survive
the frozen 4,000-draw sign-flip null (`p=0.154`).

This is a new-architecture formation control, not a new positive case. It is
consistent with the property's abstention boundary: an implementation can
produce a measurable backward difference while its transported conditional
mean is not persistent. A live-weight consequence was intentionally not run
after the source/transport formation screen failed; feedback-sustained drift
remains a separate out-of-domain mechanism.

The official fused Mamba seq64/seq256/seq1024 confirmations were also
reconciled against this boundary: all have `natural_bias_case_added=false`
and fail the frozen direction gate. The separate seq128 generated
convolution-backward binding remains `UNRESOLVED_COMPILE`, so it contributes
neither a positive nor a negative verdict. These outcomes are retained as
coverage/provenance, not silently counted as missing cases.

## Frozen confirmation

The confirmation did not reuse Phi or Qwen128 for threshold selection. It bound
new Qwen seq256 and DeepSeek seq128 lm-head backward invocations and their state
orders before values were measured. The frozen predictions were:

1. a temporally persistent orbit mean transported into the declared parameter
   predicts effective-update persistence;
2. an orbit mean whose direction re-draws predicts diffusion even when its norm
   is large;
3. stochastic centering must reduce persistence without requiring lower RMS.

The confirmation below upgrades the result beyond development evidence, while
still not making it a universal all-operator oracle.

## Prospective confirmation result

The frozen Qwen seq256 invocation passed all three preregistered gates. Its
local orbit mean was only weakly above its sign-flip null
(`A=1.001706`),
while the effective update reached
`A=1.434768`.

The cross-model DeepSeek test preserved an informative failed prediction:
the local orbit mean was not persistent
(`A=0.999775`,
`p=0.603599`),
although the natural update was persistent. Directly averaging eight real
backward orbit variants measured the required transported mean and gave
`A=1.487177`
(`p=0.000250`).

Thus the evidence rejects the stronger but unnecessary rule that $m_t$ must
share a fixed direction in endpoint coordinates. The supported rule is about
$M_t m_t$ in declared parameter coordinates. This is precisely a
source--transport interaction, not source magnitude or endpoint direction alone.

## Oracle cost and generality

The direct confirmation used eight semantic-orbit members plus one FP32 repair
per state. The research runner conservatively reruns full F+B, but an automated
implementation can capture the endpoint once and replay its already-proven VJP;
the marginal cost is therefore $K$ endpoint kernels and $K$ local VJPs, not
$K$ complete training steps. Sixteen states and $K=8$ are the confirmation
configuration; a two-state engineering pass is already supported.

The oracle is general for endpoints with a valid semantics-preserving orbit and
an executable F+B boundary. It must abstain for operators without such an orbit
or without a closed transport. This is broad across reduction implementations,
but it is not yet an all-operator universal oracle.
