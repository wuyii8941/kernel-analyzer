# What the current data says about the optimizer

The current evidence does **not** show that AdamW is the original source of
the numerical bias. It shows that the optimizer is part of the path that can
preserve, suppress, or amplify an error.

The cleanest comparison is Phi `lm_head dX`:

| same gradient difference | directionality |
|---|---:|
| parameter-gradient sequence | `A=4.665` |
| stateless SGD mapping | `A=4.701` |
| cold-start AdamW mapping | `A=1.031` |

The directional difference is already present before the optimizer. AdamW
reduces it under the declared cold-start protocol. That is evidence for an
optimizer-conditioned screen, not proof that AdamW created the error.

The Gemma feedback experiment gives a second, different result: replacing the
normal optimizer behavior with stateless SGD or resetting moments reduces the
later feedback separation. This shows that optimizer state can maintain a
trajectory split even when the direct local effect is close to diffusive. It
does not prove that every observed split is an optimizer bug.

The two Phi numbers must not be combined as one experiment:

- `A=3.325 -> 0.956` is a 16-state stateless-SGD stochastic-rounding
  intervention;
- `A=1.029` is a 32-step cold-start AdamW direct-persistence result.

The first does not explain or repair the second. A matched AdamW source
intervention is still open.

To attribute a formation effect to the optimizer itself, v4 requires a
same-state comparison using the same weights, inputs, gradient difference and
captured moments, with moment reset and stateless SGD as controls. Natural
early/middle/late training measurements must also capture their own weights,
gradients and moments; late moments must not be installed on an early state.
Until those measurements exist, the correct wording is:

> The optimizer changes whether a numerical difference reaches the effective
> update and can maintain feedback; the current data does not identify the
> optimizer as the universal root cause.

