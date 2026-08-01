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

The single complete case is the seq128 `lm_head` input-gradient matrix
multiplication. It is a different kernel from FlashAttention but has the same
abstract mechanism: directional local arithmetic error, coherent parameter
gradient carrier, and repeated weight divergence.

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

## Files

- `results/final/summary.json`: concise result.
- `results/final/math.json.gz`: complete invocation proof ledger.
- `results/final/precision.json.gz`: BF16/FP16 census and causal evidence.
- `results/final/compiled.json.gz`: compiled-region aggregates.
- `results/final/manifest.json`: checksums and archive contents.

Run `python3 scripts/check.py` to verify the package.
