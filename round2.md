# Qwen3-VL all-op F+B result

> Historical Qwen3-VL evidence note. Current SiLU classification and project
> terminology are defined in `docs/current_mainline.md` and the long-run
> machine audit; this note preserves the original F+B derivation only.

## Scope

This round covers one natural Qwen3-VL-Reranker-2B multimodal loss step. The
proof unit is one concrete forward origin together with the backward program
that AOTAutograd actually executes. Operator names and shapes are never used to
pair forward and backward.

Property induction has not started.

## Mathematical coverage

| Graph | Forward nodes | Backward nodes | F+B units | Proved units | Auxiliary nodes |
|---|---:|---:|---:|---:|---:|
| BF16 | 5,194 | 8,669 | 3,589 | 3,589 | 27 / 27 |
| FP32 | 4,764 | 8,242 | 3,211 | 3,211 | 27 / 27 |

The BF16 eager census contains 14,660 actual ATen invocations. Instrumented
and uninstrumented eager runs have identical loss and all 625 parameter
gradients.

Every AOT call-function node is accounted for. The proof includes:

- exact arguments, saved tensors, dimensions, reductions and tuple ports;
- matrix, normalization, nonlinear, indexing, convolution and loss VJPs;
- 519 unit-alpha gradient fan-in additions;
- zero-arithmetic SSA routes for add, clone and alias;
- forward origins with no requested trainable input edge;
- three functionalized deepstack overwrites and all 27 auxiliary scatter VJP
  nodes.

This proves real-arithmetic correspondence between the concrete AOT forward
and actual AOT backward. It does not prove the finite-precision arithmetic of a
generated Inductor, Triton or cuBLAS kernel.

## Natural AOT SiLU difference

The eager and AOT losses are bitwise equal. Their gradient difference is
caused entirely by the backward of the 28 text SiLU invocations.

For

\[
y=x\sigma(x),
\]

the AOT graph executes the decomposed BF16 VJP

\[
d x=q\,\sigma(x)\left[1+x(1-\sigma(x))\right],
\]

where each elementary operation is materialized at the graph dtype. Eager uses
`aten.silu_backward`.

A backward-only intervention retained the exact `aten.silu` forward and
replaced eager's VJP with the AOT decomposition. The intervention reproduced
the AOT gradient bitwise for all 625 parameters:

| Precision | AOT-eager gradient L2 | Intervention residual | Exact parameters |
|---|---:|---:|---:|
| BF16 | 21.36792274 | 0 | 625 / 625 |
| FP32 | 0.0009945415 | 0 | 625 / 625 |

Thus decomposition choice is the implementation trigger and low precision is
the dominant amplifier. The BF16 error is about 21,485 times the FP32 error in
global L2.

## Directional gate

Six frozen natural image/text states used the same shape and visual-token
positions. In every state:

- eager and AOT loss were bitwise equal;
- the decomposed-SiLU intervention reproduced all 625 AOT gradients exactly;
- relative global gradient L2 error was 1.74% to 5.55%.

However, the all-model cross-state error statistic was

\[
\widehat{U}=-4.40445,
\qquad
\frac{\widehat{U}}{\mathbb{E}_s\lVert e_s\rVert^2}=-0.01136.
\]

The tested layer-0 gate, layer-2 gate and visual patch-embedding weight
endpoints also had negative cross-state inner products. Although 23 of 28
local SiLU VJPs had positive mean pairwise inner products, their signed means
were not stable and the effect did not survive as a coherent full-gradient
carrier.

## Verdict

This is a natural, independently derived, complete F+B **causal numerical
difference**, but it is not currently a FlashAttention-style directional-bias
case. It therefore does not increase the project's accepted natural bias-case
count. The negative directional result is retained rather than relabeled as a
property.

Compact evidence is generated from:

- `results/round2/vl_math_ledger.json.gz`;
- `results/round2/vl_math_ledger_fp32.json.gz`;
- `results/round2/vl_silu_cause.json`;
- `results/round2/vl_silu_cause_fp32.json`;
- `results/round2/vl_bias.json`.
