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

The new exact 32-step same-state replay, with raw gradient/update/moment
capture, gives the same pattern numerically:

| replay arm | `A32` |
|---|---:|
| gradient difference | `4.6827` |
| stateless SGD | `4.6827` |
| captured AdamW moments | `1.0296` |
| moment reset at every step | `1.0139` |

The artifact is
`results/property/direct_persistence_v4/optimizer_state/phi_seq64_same_state_ablation.json`.
The 118-MB raw vectors stay outside the repository under
`/data1/tzh/cache/kernel_analyzer/direct_persistence_v4/` and are identified
by a SHA-256 digest in that artifact.

## Third same-state replay: Liger

The Liger fused cross-entropy case was replayed with the same four response
arms. Its scores were:

| replay arm | `A32` |
|---|---:|
| gradient difference | `2.8382` |
| stateless SGD | `2.8382` |
| captured cold-start AdamW moments | `1.6809` |
| moments reset at every step | `1.8284` |

AdamW reduces the source directionality here but does not remove it. This is
different from Qwen, where the captured AdamW response is near the diffusive
range, and from Phi, where the gradient directionality is much stronger than
the AdamW update directionality.

The artifact is
`results/property/direct_persistence_v4/optimizer_state/liger_t128_same_state_ablation.json`.

The directional difference is already present before the optimizer. AdamW
reduces it under the declared cold-start protocol. That is evidence for an
optimizer-conditioned screen, not proof that AdamW created the error.

## Second same-state replay: Qwen

The same 32-step raw replay was run for the Qwen `lm_head dX` family at the
declared seq128 identity. The scores were:

| replay arm | `A32` |
|---|---:|
| gradient difference | `1.3430` |
| stateless SGD | `1.3430` |
| captured cold-start AdamW moments | `0.9611` |
| moments reset at every step | `1.0000` |

Here AdamW suppresses the direct directionality rather than creating it. This
is the opposite response from the Phi replay, where the gradient and SGD
scores were about `4.683` while captured AdamW was about `1.030`. Together the
two replays establish a narrower, testable statement: **the optimizer changes
the mapping from a numerical gradient difference to an effective update, and
the direction can be suppressed or retained depending on the case.** They do
not identify the optimizer as the source of the numerical error.

The Qwen artifact is
`results/property/direct_persistence_v4/optimizer_state/qwen_seq128_same_state_ablation.json`.
Its 121-MB raw vectors remain outside the repository and are recorded by SHA-256
in the artifact.

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
