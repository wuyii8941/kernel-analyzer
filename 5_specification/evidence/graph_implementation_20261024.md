# GRAPH implementation evidence — 2026-10-24

Status: implementation candidate completed; independent implementation re-review still required. The contract is
`reviewed_for_implementation`; `activation.implementation_status` remains `pending`.

## Common semantic gate

`kernel_spec/graph_semantics.py` is now the shared admission point for complete and lazy GRAPH execution. It maps an
operation to one reviewed base contract and selected profile (or an explicit GRAPH-local rule), rejects a base
contract whose implementation is not validated, binds reviewed keyword aliases, and separates physical graph inputs
from internal exact/enclosure values. MATHFN point evaluation uses its reviewed Arb-backed core; interval inputs use a
separately named enclosure extension with the same domain and magnitude limits.

The complete evaluator no longer calls `aten.OPS` merely because an operation name exists. The lazy evaluator defaults
to complete-graph checking, forces node obligations, stores only mathematical `Undefined` as local poison, and lets
`Invalid`, `Unsupported` and `NotEstablished` escape with their original meanings. Helper-only evaluation remains
available only through the explicit `development_helper` admission mode.

Large captured workloads use an explicit pure output-dependency slice. Their records say `scope=output_slice` and
`status=not_admitted`; this preserves useful development comparisons without claiming that the original whole graph
was certified. `inductor_default` is labelled counterfactual in those records.

## Independent checks

`tests/test_graph_independent_structure.py` contains a small Fraction/NumPy control interpreter that does not import
the GRAPH evaluator or semantic dispatcher. It checks chains, shared values, branches and merges, tuple result
identity, repeated outputs, conversions, empty shape propagation and the fresh local-update case. Rejected elementary
domains and legal tied-sort permutations are fixed independently.

`tests/test_graph_semantic_gate.py` covers the review's counterexamples:

- complete and lazy `log(0)` are mathematical undefined;
- `log([-1,4]) -> select(1)` uses one whole-node undefined rule in both paths;
- an unused `log(-1)` may coexist with an independent value;
- an unused `exp(101)` or invalid `view` blocks complete-graph admission;
- a helper outside the contracts needs explicit development mode;
- logical `1/3` is not accepted as a physical f32 input;
- an admitted operation with no lazy backend is `not_established`, not unsupported;
- contract profile restrictions and reviewed public argument aliases are enforced.

`tests/test_graph_admission_accounting.py` verifies that a historical complete reference, a partial slice, a pending
implementation or missing per-call checks cannot enter the strict count. `sufficiency/analyze_compiled.py` now derives
strict and bias-query counts from those records rather than hard-coded zeros or the global GRAPH status.

## Executed checks

- Final full specification suite: 1852 passed, 1 skipped, 6365 subtests in 440.43 seconds; the two warnings were
  existing PyTorch convolution/std warnings.
- Contract checker after the changes: GRAPH 13/13 examples, 1/1 diagnostic reading and 4/4 mutations; all 64 contracts
  consistent.

No GPU test was run for this review because GRAPH's admitted claim here is the mathematical composition evaluator,
not a concrete Triton binding. GPU and compiler evidence remains a per-call obligation.
