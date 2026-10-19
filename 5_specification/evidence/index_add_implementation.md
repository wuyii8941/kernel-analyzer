# Index-add implementation review — 2026-10-19

The active INDEXREDUCE contract covers complete out-of-place index_add forward values on positive rank-1 through
rank-4 physical finite f16, bf16, f32 or f64 tensors. Self and source have the same dtype, index is a nonempty int64
vector, and alpha is finite and exactly representable in the common dtype.

The retained PyTorch 2.10 index_add page defines `self[index[i], ...] += alpha * source[i, ...]` along the selected
dimension and requires source length on that dimension to equal the index length. The retained native schema file
confirms the one-output functional call and separates its out overload. These sources fix the selected value task
without relying on a tested Triton result.

The former active draft combined index_add, index_reduce and scatter_reduce values, VJPs, in-place behavior, reduction
identities and tied-extremum allowed sets. It remains in
`contracts/draft/INDEXREDUCE_extended_20261019.yaml`; none of those additional tasks is admitted by this review.

The reviewed adapter validates physical dtype, common rank and dtype, positive dimensions, finite exactly
representable values, a literal int64 index vector, normalized dimension, source shape, index range and finite
representable alpha before constructing every output coordinate. It refuses mutation, out tensors, other reductions,
backward, special values, signed-zero distinctions and unselected dtypes or ranks.

Validation comprises 20 contract examples and six mutated formulas. Sixteen generated cases cover all four floating
dtypes and ranks one through four, including duplicate indices, different dimensions and non-unit alpha. An
independent coordinate implementation and PyTorch 2.10 agree on every checked value, dtype and shape.
