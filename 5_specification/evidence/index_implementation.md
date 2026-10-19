# INDEX read-only lookup implementation review — 2026-10-18

## Reviewed task

The active INDEX contract now covers only the default functional forward forms of `gather`, `index_select` and
`embedding`. The former broad draft also mixed scatter, mutation, backward accumulation and nondeterministic duplicate
writes under one allowed-set output. That version remains in `contracts/draft/INDEX_extended_20261018.yaml` and is not
used at runtime.

The reviewed profile accepts physical finite f16, bf16, f32 and f64 source values that are exactly representable in
their declared dtype, plus literal int64 indices. `gather` requires equal input/index rank and no broadcasting;
`index_select` requires a one-dimensional index; `embedding` requires a two-dimensional weight. Only nonnegative
in-range indices and default non-sparse options receive answers. Scatter, updates, gradients, integer data, special
floating values, other index dtypes and nondefault embedding options remain unsupported.

## Source correspondence

Retained PyTorch 2.10 pages state the coordinate rule and no-broadcast shape conditions for `gather`, the slice and
shape rule for `index_select`, and the row-lookup and output-shape rule for functional embedding. The retained ATen
schema file separates these default calls from out, backward, scatter and in-place operations. The retained PyTorch
2.10 `Embedding.cpp` implements forward embedding as `weight.index_select(0, indices)` followed by restoration of the
index shape. These sources support the selected read-only task without relying on a tested Triton result.

## Independent implementation checks

The adapter validates the schema and physical records before evaluation, constructs every output coordinate, and
checks complete observed dtype, shape and values. Twenty-six contract examples are consistent and six distinct wrong
lookup implementations are rejected.

Independent NumPy coordinate loops recomputed gather, index-select and embedding results for 40 generated cases.
Separate PyTorch 2.10 calls matched the profile for f16, bf16, f32 and f64. Domain tests reject unrepresentable values,
negative zero, special values, non-int64 indices, bad ranks and ranges, sparse gather, nondefault embedding options and
scatter. These checks support the declared profile only; they do not certify the archived broader draft.

2026-10-19 keyword correspondence review: a separate FlagGems batch supplied `index=` and was refused because the
reference's internal parameter was named `idx`. Retained `TS_NATIVE_FUNCTIONS.yaml.gz` declares `self, dim, index`;
`T_GATHER.txt.gz` and `T_INDEX_SELECT.txt.gz` declare public `input, dim, index`. The explicit per-operation mappings
now resolve these names to `a, dim, idx` before the existing signature binding and physical-index validation.
Review of the shared adapter confirms that it preserves every supplied value, rejects duplicate aliases and
positional/keyword collisions, and applies the same mapping to reference evaluation, complete output checking and
wrong-formula witnesses. It does not infer an argument name from a kernel. Existing DSL names remain supported.
The coordinate formulas, accepted index dtype/range and output guarantee are unchanged. Targeted checks passed
14 tests and 60 subtests before updating the retained contract/adapter; the contract checker passed 21 examples and
rejected all six wrong formulas. This is a local source/implementation review, not an external review.

Public embedding follow-up: a real FlagGems 5.0.2 `F.embedding(indices, weight)` matched native PyTorch but the INDEX
reference treated its first argument as weight. The public operation had incorrectly reused the native signature.
The retained functional source and API definition specify indices first, public padding `None` translated to native
`-1`, and an in-place renormalisation when `max_norm` is supplied. The public wrapper now validates that signature
and its default read-only scope before reusing the existing lookup; the native signature remains unchanged. The
complete EMB forward/VJP task continues to handle nondefault padding and frequency choices. Local source review and
17 targeted tests with 87 subtests check positional/keyword calls, duplicate arguments, strict index decoding,
scalar/empty index shapes, and refusal of mutation. The INDEX checker now has 26 examples and six rejected wrong
formulas; EMB has 18 examples and seven rejected wrong formulas. No new mathematical contract was introduced.
