# CAST implementation review — 2026-10-18

## Admitted task

The active `finite_rne` profile covers only finite conversions among f16, bf16, f32 and f64. Every input value must
be exactly representable in its declared source dtype, the target result must remain finite, and the caller must
explicitly select IEEE round-to-nearest, ties-to-even. The result is one complete tensor with the target dtype and
the exact input shape. Integer, boolean, float8, complex and nonfinite conversions remain outside this review.

This explicit task choice matters because the retained public PyTorch `Tensor.to` page names the target dtype but
does not by itself state one universal floating rounding direction. The retained IEEE 754 rule defines the selected
rounding profile. The retained PyTorch 2.10 prim and ATen schemas fix the two accepted functional call interfaces.
The former broad draft remains under `contracts/draft` and is not loaded at runtime.

## Reference and independent checks

`adapters/CAST.py` validates both source and target dtype, finiteness, exact source representability, finite target
range and the selected `_to_copy` options before conversion. It uses the exact rational rounding oracle for the
answer and the common complete-output checker for dtype, explicit shape and every coordinate.

The independent test constructs expected answers with separate integer bit-length and `Fraction` arithmetic. It
checks normal values on both sides of midpoints, even and odd midpoint ties, positive and negative values, zero and
subnormal boundaries, the normal/subnormal transition, maximum finite values, widening and identity conversion for
all four admitted formats. Domain tests cover every source/target pair and reject unrepresentable source records,
overflow, nonfinite values, unreviewed dtypes and copy options. Output tests reject wrong dtype, shape, coordinates
and ambiguous empty shapes. The contract checker establishes 12/12 examples and rejects all four wrong rounding
implementations.

As a same-source interface cross-check, 2,048 coordinates from all 16 source/target dtype pairs agreed exactly with
the installed PyTorch 2.10 conversion behavior. This supports the call correspondence; it is not the independent
basis for the rounding formula.

## Limits

This review does not establish signed-zero handling, NaN payloads, infinity, output overflow, integer or float8
conversion, device transfer, memory format, asynchronous copies, aliasing, or a concrete Triton kernel. A kernel may
use another rounding rule legally; it is comparable to this profile only after its task binding explicitly selects
roundTiesToEven and accounts for all inputs, outputs and state.
