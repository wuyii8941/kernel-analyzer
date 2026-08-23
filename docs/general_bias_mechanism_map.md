# Where implementation bias is formed

This analysis asks one narrow training question:

> At what point does a candidate kernel's numerical difference become a directionally accumulating parameter-update difference?

It separates three places in the real training step:

1. the target operator's output difference;
2. the resulting parameter-gradient difference after backward;
3. the effective parameter-update difference after the optimizer mapping.

All three values below use the same ordered 32-state reference trajectory for
each case.  They are measurements on the declared parameter carrier, not
full-model training results.

| case | operator-output `A` | parameter-gradient `A` | effective-update `A` |
|---|---:|---:|---:|
| Liger fused CE | 2.984 | 2.931 | 2.931 |
| Phi-4 `lm_head dX` | 2.074 | 4.701 | 4.701 |
| Qwen `lm_head dX` | 1.008 | 1.698 | 1.698 |

The three cases do not form bias in exactly the same place.  Liger is already
directional at its fused operator boundary.  Phi and Qwen become more
directional when the operator difference is transported through backward to
the measured parameter gradient.  Stateless SGD then preserves that
directionality almost exactly.

The optimizer is not always passive.  On the same Phi gradient sequence,
stateless SGD preserves `A=4.701`, while the frozen AdamW response maps it to
`A=1.031`.  In this probe, AdamW suppresses the directional gradient error.

Exact replay of both the measured error and its negative is available for
saved-P and SiLU. Both contain one response component that stays the same when
the error sign is reversed and another component that changes sign; neither can
be described as an operator-output bias alone. Phi is deliberately left
unresolved for this decomposition: the exact negative of its natural BF16 endpoint error is
not representable in any of the 16 tested states, with relative representation
error `0.0386--0.0761`.  Approximate negative arms are not used as evidence.

The state-propagation probe for Phi gives a final drift/direct-sum ratio of
`1.0104`.  The accumulated local update sequence therefore explains nearly all
of that probe's drift; it is not an example of large extra closed-loop
amplification.  In contrast, 11 of 12 sampled controls have locally diffusive
operator increments but persistent feedback.  That feedback is treated as a
training-dynamics background, not promoted to operator-source bias.

## Current scientific conclusion

The measured cases support a three-part explanation:

- the implementation and its operands create an operator-output residual;
- backward and the optimizer decide whether that residual becomes a
  directional parameter update;
- later training state decides whether the update sequence is preserved,
  suppressed, or amplified.

This is a measured mechanism map, but it is not yet a universal low-cost
property. Under the frozen generic predictor specification, none of the five
evaluated cases has every required source measurement, exact response, and propagation
input.  The predictor therefore emits five explicit abstentions and no score.
That fail-closed result is recorded in
`results/property/joint_bias_formation_v1/joint_predictor_evaluation_v1.json`.

The machine-readable measurements are in
`results/property/joint_bias_formation_v1/general_mechanism_map_v1.json`.
