# Three complete forward/backward bias cases

## Counting rule

A complete case must bind one real forward to its actual backward, show a
directional local error, trace that error to a coherent parameter-gradient or
weight carrier, and pass a causal intervention. A nonzero forward residual is
not required: an exact forward followed by a biased, exactly bound backward
edge is still a complete forward/backward case. Here, "natural" means that the
case occurs on unmodified model states and inputs rather than an injected
sentinel or a controlled synthetic domain.

There are three cases in this comparison. FlashAttention is the literature
anchor; the other two are natural cases found by this project. Thus our own
natural-case count is **2**, not 0 or 1.

| Case | Origin | Biased endpoint | Isolated cause | Accumulation evidence |
|---|---|---|---|---|
| FlashAttention | Qiu and Yao | attention backward / query-weight gradient | low-precision online-softmax output reused in backward | coherent weight error across tokens and steps |
| seq128 `lm_head` MM | this project | actual input VJP `dX` | BF16 GEMM reduction and arithmetic path | downstream gradient carrier and 32-step weight divergence |
| Liger fused linear CE | this project | actual `dW` | 64 chunk contributions stored and added in BF16 | direct tied-weight gradient carrier across held-out states |

## 1. FlashAttention reference case

For

\[
S=\alpha QK^T,\qquad P=\operatorname{softmax}(S),\qquad O=PV,
\]

and upstream cotangent \(G=\partial L/\partial O\), the backward is

\[
dV=P^TG,\qquad dP=GV^T,
\]

\[
\delta=\operatorname{rowsum}(G\odot O),\qquad
D=P\odot(dP-\delta),
\]

\[
dQ=\alpha DK,\qquad dK=\alpha D^TQ,
\qquad dW_Q=dQ^TX\quad\text{when }Q=XW_Q^T.
\]

The paper identifies a directional low-precision error in the tiled
online-softmax output \(O\). Because backward reuses \(O\) through \(\delta\),
the error becomes a structured query-gradient error. Its projection onto the
query-weight update is coherent across tokens and training steps, so it does
not cancel. Recomputing the relevant forward quantity at higher precision, or
changing the mathematically equivalent online-softmax scaling, removes the
cause and restores training stability.

This is a natural, complete forward/backward and long-trajectory case reported
by the paper, not a discovery of this repository.

## 2. seq128 `lm_head` input-gradient MM

The concrete Qwen3-1.7B invocation is

\[
Y=XW^T,
\]

with actual backward

\[
dX=GW,\qquad dW=G^TX.
\]

The proof binds the real forward to the real input-VJP edge using the same
saved \(W\) and upstream cotangent \(G\). The forward result is unchanged by
the backward-only repair; the directional error is in `dX`. This still meets
the complete F+B definition because the forward, saved values, cotangent, and
actual backward program are closed as one unit.

Disabling BF16 reduced-precision reduction removes about **91.05%** of the
local residual RMS. The remaining error is consistent with the GEMM
FMA/reduction tree and accumulation order. In an independent 32-step
live-weight experiment:

- the parameter-gradient carrier is nonzero in 32/32 steps;
- FP32 master weights and materialized BF16 weights diverge in 32/32 steps;
- final pairwise L2 distances are 0.00487622 (FP32 master) and 0.00536579
  (materialized BF16).

Therefore `lm_head` **is a natural, paper-level complete F+B case**. It is not
the FlashAttention kernel or the same source bug; it reproduces the same
causal form with a different local arithmetic mechanism.

## 3. Liger fused-linear cross-entropy `dW`

The closed terminal region is

\[
Z=HW^T,\qquad
L=-N^{-1}\sum_t\log\operatorname{softmax}(Z)_{t,a_t},
\]

\[
G_{t,v}=N^{-1}
\left(\operatorname{softmax}(Z)_{t,v}-\mathbf1[v=a_t]\right),
\qquad dH=GW,\qquad dW=G^TH.
\]

The eager region reproduces the full-model loss, and its logits VJP equals the
cotangent captured from the real full backward. The candidate is Liger's actual
custom autograd program. At \(T=128,D=2048,V=151936\), it processes two tokens
per chunk and makes 64 sequential additions into a BF16 `dW` accumulator.

Changing only that accumulator to FP32 leaves loss and `dH` bitwise identical
in all 24 held-out states, while the default-minus-FP32 `dW` carrier is positive
in 24/24. Mean `dW` RMS errors against the same FP32 region reference are

\[
6.6562\times10^{-6}\;\text{(default)},\quad
5.5979\times10^{-6}\;\text{(FP32 accumulator)},\quad
5.5504\times10^{-6}\;\text{(eager)}.
\]

The intervention removes **95.7%** of the mean candidate-added `dW` error. In
a disjoint full-step confirmation, only the tied
`model.embed_tokens.weight` gradient changes; loss, terminal `dH`, and the
other 309 parameter gradients are bitwise exact. Its frozen carrier is positive
in 24/24 states with bootstrap 95% CI \([0.168,0.220]\).

Appending mathematically ignored zero rows changes the chunk schedule from
64 two-token chunks to 32 four-token chunks. The `dW` effect remains directional
with BF16 accumulation (24/24), but is incoherent with FP32 accumulation
(13/11). This establishes a **chunk geometry x accumulation precision**
interaction. It refines this case and is not counted as a fourth case.

## Common conclusion and boundary

All three support

\[
\text{directional finite-arithmetic error}
\longrightarrow
\text{coherent gradient carrier}
\longrightarrow
\text{parameter/weight accumulation}.
\]

They do not support one shared kernel bug. The two project cases are both
precision-mediated, although shape, layout, support, and chunk geometry decide
whether the fixed-precision arithmetic produces a coherent direction. Two
independent project cases are still too few for a defensible generalized
cross-operator property.

## Evidence

- FlashAttention paper: <https://arxiv.org/pdf/2510.04212>
- `lm_head`: `results/final/precision.json.gz` (`mm_case`, `mm_arithmetic`,
  `mm_carrier`, and `mm_steps` entries in `results/final/manifest.json`)
- Liger fused CE: `archive/nonprecision_v1/runs/liger.fused_ce.mechanism.json`,
  `liger.fused_ce.certificate.json`, and `liger.fused_ce.chunk.certificate.json`
