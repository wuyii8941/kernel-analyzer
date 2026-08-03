# Results

## Coverage

| Item | Result |
|---|---:|
| BF16 eager invocations | 9,269 / 9,269 |
| FP32 eager invocations | 8,701 / 8,701 |
| FP16 eager invocations | 27,639 / 27,639 |
| Atomic forward + actual-VJP units | 3,491 / 3,491 |
| Unresolved mathematical units | 0 |

## Bias census

| Precision | Endpoint strata | Directional | Nonfinite | Complete coherent cases |
|---|---:|---:|---:|---:|
| BF16 | 21,861 | 122 | 0 | 1 |
| FP16 | 22,641 | 510 | 21 | 0 |

The primary Qwen all-op census's single complete case is the seq128 `lm_head`
input-gradient matrix multiplication. It is a natural, complete F+B case: its
forward is exactly bound to the biased actual input-VJP edge. It is a different
kernel from FlashAttention but has the same abstract mechanism: directional
local arithmetic error, coherent parameter-gradient carrier, and repeated
weight divergence. Its 32-step experiment is paired baseline versus analytic
VJP repair at the exact input-gradient edge.

The 21 FP16 nonfinite cases are attention-mask additions. Their `-inf` values
are causally absorbed by the closed softmax forward/backward region.

## Compiled extension

The structural inventory contains 17,472 generated-region occurrences across
12 configuration-shape cells. In BF16 Inductor, 16 locally directional concrete
region-shape targets were tested over 512 target-states with exact identity
shams.

- Regions with a positive global carrier lower bound: 0 / 16.
- Parameters with a positive independent carrier lower bound: 0.
- Additional complete paper-like cases: 0.
- Seq256 fused NLL changes loss in 22 / 32 states but parameter gradients in
  0 / 32 states; it is a loss-only bias.

This does not certify every generated region in every compiled configuration.

## Full BF16 Inductor Triton screen

The primary BF16 Inductor configuration was subsequently screened without
selecting operator names or candidate values: all 686 seq64, 686 seq128, and
740 seq256 Triton invocations were observed in each of 32 frozen states. One
pilot state was excluded, seven states froze the carrier directions, and 24
states were used once for joint confirmation.

| Shape | Endpoint/closure units | Exact controls | Joint tests | Passed units | Complete closures |
|---|---:|---:|---:|---:|---:|
| seq64 | 1,113 | 291 | 1,636 | 43 | 2 |
| seq128 | 1,113 | 286 | 1,645 | 69 | 1 |
| seq256 | 1,225 | 342 | 1,761 | 125 | 0 |

The seq64 forward/backward and seq128 backward closures are the already known
terminal NLL implementation case. Seq256 produced no complete closure; in
particular, none of 57 newly closed RMSNorm weight-gradient split reductions
passed the joint gate. No additional natural forward+backward case was found.

At the external GEMM boundary, seq128 and seq256 one-state identity pilots each
replayed 760/760 calls exactly in both repeats. These validate the generated
call boundary, but are not presented as independent mathematical proofs because
candidate and control dispatch the same PyTorch/cuBLAS operation.

## Non-dtype factors

Strict-FP32 controlled experiments isolate three factors that determine the
direction of finite-arithmetic bias while keeping dtype fixed: causal-softmax
support geometry, RMSNorm hidden-energy-to-epsilon ratio, and terminal NLL
implementation/materialization. Controlled SiLU input domain/scale and BF16
GEMM operand layout also produce complete local forward/VJP effects, but their
independent full-parameter carriers fail. Liger fused cross entropy adds the
project's second natural complete F+B case: chunk geometry interacts with a
BF16 `dW` accumulator, and changing only that accumulator to FP32 removes 95.7%
of the mean candidate-added `dW` error. The final tied-weight carrier confirms
on 24/24 held-out states; the other 309 parameter gradients are bitwise-exact
controls. A subsequent frozen 32-step paired trajectory evaluates both
accumulators at each arm's evolving weights. All 64 same-weight carrier
projections are positive and all controls remain exact. The FP32-master
default-minus-repair distance grows from 8.5868e-6 after the first update to
2.2394e-3 after the final update; the materialized-BF16 distance grows from
6.3461e-5 to 3.2074e-3. This closes the stateless-SGD live-weight consequence,
without claiming AdamW behavior or loss instability.

These are mechanism cases, not yet a generalizable cross-operator property.
The three-case comparison with the FlashAttention reference is in `case.md`.

## Qwen3-VL round 2

The second-model census records 14,660 BF16 eager invocations. Its independent
AOT proof closes 3,589/3,589 BF16 and 3,211/3,211 FP32 forward/backward units,
including all functionalized auxiliary nodes.

AOT's graph-dtype decomposition of text SiLU backward exactly explains the
candidate gradient difference in all 625 parameters under both BF16 and FP32.
The BF16 global delta L2 is 21.3679, versus 0.0009945 in FP32. Across six
natural states, however, the global coherence ratio is -0.01136 and the tested
weight carriers are not directional. This is retained as a complete causal
numerical difference and a negative bias result, not promoted to a third
project bias case. See `round2.md`.

## Files

- `results/final/summary.json`: concise result.
- `results/final/math.json.gz`: complete invocation proof ledger.
- `results/final/precision.json.gz`: BF16/FP16 census and causal evidence.
- `results/final/compiled.json.gz`: compiled-region aggregates.
- `results/final/generated.json`: full BF16 Inductor Triton screen.
- `results/final/vl.json.gz`: compact Qwen3-VL proof and bias evidence.
- `results/final/trajectory.json.gz`: complete Liger 32-step repair trajectory.
- `results/final/manifest.json`: checksums and archive contents.

Run `python3 scripts/check.py` to verify the package.
