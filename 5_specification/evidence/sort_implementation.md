# SORT implementation review — 2026-10-19

The active SORT contract now covers only PyTorch 2.10 functional `sort`, `argsort`, and sorted `topk` on rank-one-or-higher physical finite f16, bf16, f32, or f64 tensors. Every row along the selected dimension must have pairwise distinct values. This restriction makes both values and original indices unique, so the reference does not choose a behavior that PyTorch leaves unspecified for unstable ties.

The retained PyTorch 2.10 pages define direction, dimension, returned indices, stability, top-k selection, and sorted output. The retained native schema file distinguishes functional, stable, out, and named-dimension forms. The broader former contract, which represented tied indices with per-position allowed sets, remains in `contracts/draft/SORT_extended_20261019.yaml`; it is not an admitted ground truth.

The executable adapter first checks the exact schema, option types, physical dtype, finite values, exact representability, rank, dimension, k, and row uniqueness. It then returns complete values and int64 indices. Ties, NaN, infinities, signed-zero distinctions, integer or complex data, rank-zero input, unrepresentable literals, unsorted top-k, named dimensions, and output-buffer forms are refused.

Validation comprises 17 contract examples, six mutated readings, complete dtype/shape/output checks, 240 generated sort/argsort/topk comparisons against independently constructed NumPy coordinate orderings, and PyTorch 2.10 checks in all four admitted dtypes. The old lower-level allowed-set helper remains tested for the drafted GRAPH composition; it does not widen the active SORT profile.
