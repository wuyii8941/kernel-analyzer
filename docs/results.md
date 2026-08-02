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
weight divergence.

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
controls.

These are mechanism cases, not yet a generalizable cross-operator property.
The three-case comparison with the FlashAttention reference is in `case.md`.

## Files

- `results/final/summary.json`: concise result.
- `results/final/math.json.gz`: complete invocation proof ledger.
- `results/final/precision.json.gz`: BF16/FP16 census and causal evidence.
- `results/final/compiled.json.gz`: compiled-region aggregates.
- `results/final/generated.json`: full BF16 Inductor Triton screen.
- `results/final/manifest.json`: checksums and archive contents.

Run `python3 scripts/check.py` to verify the package.
