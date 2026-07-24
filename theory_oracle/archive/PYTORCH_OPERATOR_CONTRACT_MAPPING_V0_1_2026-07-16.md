# PyTorch Operator Contract Mapping v0.1 — 2026-07-16

> Scope: map the abstract Operator Oracle contracts to PyTorch's documented numerical and reproducibility guarantees. This document does not infer undocumented semantics.

## 1. Authoritative documentation checked

PyTorch's current numerical-accuracy note states that floating addition/multiplication are non-associative and that PyTorch does not guarantee bitwise-identical results for mathematically identical floating computations, across releases/commits/platforms, or between CPU and GPU even after randomness is controlled. It also documents that batched versus sliced computations may differ, extreme intermediates may overflow, TF32 may read reduced mantissa precision, and some FP16/BF16 GEMM reductions may truncate intermediate accumulation.

Source: [PyTorch Numerical accuracy, version 2.13](https://docs.pytorch.org/docs/2.13/notes/numerical_accuracy.html), retrieved 2026-07-16.

PyTorch's reproducibility note states that complete reproducibility is not guaranteed across releases/platforms or CPU/GPU even with identical seeds, while deterministic-algorithm configuration can constrain some operations on a fixed platform/release.

Source: [PyTorch Reproducibility, version 2.13](https://docs.pytorch.org/docs/2.13/notes/randomness.html), last updated 2026-05-14 and retrieved 2026-07-16.

## 2. Consequences for the Oracle

### 2.1 Bitwise eager equivalence is not the default floating correctness relation

Because the framework explicitly permits numerical differences in several settings, compiled versus eager bitwise mismatch cannot by itself establish a PyTorch correctness violation. It can establish a compatibility/reproducibility difference if a project contract requires bitwise replay in a narrower fixed configuration.

### 2.2 Configuration is part of contract identity

The semantic/numerical envelope depends on at least:

- dtype and accumulation precision;
- device/backend/platform;
- TF32 and reduced-precision-reduction settings;
- deterministic-algorithm policy;
- release/commit;
- batched/sliced/operator formulation when relevant.

A change in these fields creates a new contract. Pooling them as repeat noise would be a category error.

### 2.3 The documented allowance is not yet a quantitative error budget

Documentation that results “may differ” or reduced precision “may lead to unexpected results” does not provide a numerical `rtol`, ULP limit or population failure rate. Therefore it does not by itself instantiate an acceptance envelope for arbitrary floating outputs.

The valid options are:

1. derive a contract from an operator-specific documented guarantee, if one exists;
2. construct an `S2` high-precision/numerical-analysis envelope under explicit arithmetic assumptions;
3. define an `S3` application tolerance;
4. define an `S4` baseline compatibility margin;
5. otherwise return `UNINSTANTIATED` for floating acceptance.

### 2.4 Determinism is protocol-relative

Enabling deterministic algorithms on a fixed stack may reduce exact-input runtime variability to zero. That does not imply zero implementation bias. Conversely, identical seed without a fixed algorithm/platform/release is insufficient evidence that two repeats instantiate the same randomness contract.

## 3. Mapping to the initial contract suite

| Contract | PyTorch interpretation |
|---|---|
| C1 specified metadata/discrete | correctness only for behavior fixed by the relevant operator/API semantics |
| C2 ranking on identical operand | exact if tie/NaN/order semantics are fixed; upstream operand differences belong to impact, not ranking correctness |
| C3 reduction bound | `S2` analytical accuracy contract under declared arithmetic model; not automatically a documented PyTorch-wide promise |
| C4 stochastic/RNG | replay correctness only when algorithm/state consumption are promised; otherwise compare conditional law |
| C5 stateful transition | exact fields plus separately bounded floating state under the relevant optimizer/control semantics |
| C6 compatibility | default fallback for underspecified floating equivalence; never relabeled correctness |

## 4. Framework-specific core verdict

For a PyTorch operator instance:

```text
if documented exact/finite semantic obligation is violated:
    REJECT CORRECTNESS
else if an S2 numerical envelope exists and candidate lies outside it:
    REJECT NUMERICAL_CONFORMANCE
else if a predeclared S3/S4 compatibility contract is resolved:
    ACCEPT / REJECT / INDETERMINATE within that contract
else:
    UNINSTANTIATED for floating acceptance,
    while still reporting exact-core conformance and discrepancy decomposition
```

Execution-path, matched-input and configuration failures still take precedence as `INVALID`.

## 5. What this mapping establishes

It establishes that a hybrid Oracle is necessary for PyTorch:

- a strong exact/specification core;
- operator-specific numerical truth/envelopes where available;
- explicitly policy-based compatibility elsewhere;
- separate impact analysis.

It also rules out two tempting but unsupported universal policies: “any eager/compiled floating mismatch is a bug” and “all outputs within one global `rtol/atol` are correct.”

## 6. Remaining mapping work

The numerical-accuracy note is framework-wide and intentionally broad. Before claiming contract coverage for a specific operator family, inspect its operator/API semantics and backend/precision documentation. Lack of a quantitative promise must remain visible as `UNINSTANTIATED`; it cannot be filled by convention or by the candidate data.
