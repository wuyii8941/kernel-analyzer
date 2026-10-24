# GRAPH independent review — 2026-10-24

Reviewed target: `d36cdee`, including the GRAPH contract and adapter, complete and lazy evaluators, related base
adapters and local-update tests. This review accepts the limited composition direction after the changes below. It
does not certify graph capture, real-call binding, compiler materialisation, kernel coverage or statistical claims.

## Decision

- **GR.IDENTITY — accepted after narrowing.** The task is the supplied graph interpreted with explicitly selected
  base-contract semantics. What an error can be detected depends on the graph's capture stage and call binding; no bug
  class is permanently included or excluded by its name. A complete role contract may be used directly, while GRAPH
  is for tasks that actually require composition.
- **GR.POISON — conditionally accepted.** Only contract-defined mathematical undefinedness in a pure value graph may
  propagate locally. Invalid input, unsupported semantics, an unestablished result, budget exhaustion and execution
  or memory failures are not poison values. A result for an explicit output slice is a result for that smaller task,
  not whole-graph certification.
- **GR.JOINT_CHOICES — accepted as a relation.** Every consumer of one node observes the same complete legal choice.
  Separate nodes may choose separately unless another contract couples them; no probability distribution or
  probabilistic independence is implied.
- **GR.EFFECTS — accepted only in the existing narrow case.** A fresh, nonescaping, unaliased allocation whose only
  use is one local update may be replaced by the returned post-value. Its initial value, all contributions, result and
  errors remain observable. General mutation and aliases remain unsupported.
- **`inductor_default` — diagnostic only.** It is a counterfactual conversion-elision reading. It cannot be selected
  from result agreement or treated as an actual compiled execution without compiler and materialisation evidence.
- **GRAPH-local `full`, `index_put`, `scatter` — accepted.** Directory placement is immaterial, but their domains and
  complete-result rules are part of GRAPH rather than silently inherited from helper availability.

## Required implementation changes

1. Complete and lazy execution must share one node admission rule containing the operation/overload, selected
   contract/profile, applicable domain, guarantee and dependency identity. A name in `covers` or a helper table is not
   enough.
2. The known mismatches must close: `log(0)` is undefined under MATHFN; `exp(101)` is outside its reviewed scope;
   helper-only operations cannot enter the contract path. Internal exact/interval values need an explicit composition
   extension and may not be disguised as new physical base-contract inputs.
3. The two paths must use the same state distinction. Only mathematical `Undefined` may be stored as local poison;
   `Invalid`, `Unsupported` and `NotEstablished` must stop complete-graph admission. Whole-node and coordinate-slice
   semantics cannot be mixed without reporting the smaller scope.
4. Output consumers must preserve exact points, numerical enclosures, complete joint-result sets and
   partial/unestablished results as different cases. They must not take set midpoints or form independent coordinate
   products.
5. Historical complete development references must not become strict admissions when a global contract flag changes.
   Every real call needs its own complete-scope admission record and binding evidence.

## Evidence required before implementation validation

- an independent rational control interpreter for topology, arguments, tuple identity, shared use, output order and
  the admitted local update;
- independent elementary-function or interval oracles supplied by the reviewed base contracts, rather than comparing
  the complete and lazy implementations only to each other;
- structure families covering chains, shared subgraphs, branches, multiple outputs, conversions, empty tensors,
  rejected domains, joint choices and private updates;
- regressions for the known domain, status and diagnostic-reading mismatches;
- a new implementation review before `activation.implementation_status` changes from `pending` to `validated`.

The semantic direction is no longer awaiting approval. The implementation remains pending until the above evidence
and the current code are reviewed. Existing 141/400, 104/357 and 2110-style development/matching counts are not
automatically upgraded.
