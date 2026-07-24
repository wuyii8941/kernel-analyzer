# First Empirical Contract Witnesses — 2026-07-16

> Scope: two confirmed PyTorch wrong-code reproducers executed on a known-broken PyTorch 2.11 environment and the current 2.13 nightly. These are initial contract witnesses, not held-out Oracle validation or operator-family coverage.

## 1. Environments

| Label | PyTorch | CUDA package | GPU |
|---|---|---|---|
| broken/control | `2.11.0+cu126` | 12.6 | Tesla T4 |
| current/fixed candidate | `2.13.0.dev20260609+cu126` | 12.6 | Tesla T4 |

Both studies used existing confirmed-bug reproducers without modifying their computation.

## 2. Witness W1 — adaptive pool + flatten + sum wrong stride

Artifact: `paper/confirmed_bugs/pytorch/bug_006_adaptive_avgpool_flatten_sum/minimal_repro.py`.

### Contract

```text
subject: composite compiled region/program
exact obligations: flatten preserves real batch indexing/stride
numerical obligation: reduction of the indexed values follows an admitted floating envelope
claim level: region/program conformance, not constituent-operator attribution
```

### Broken result, PyTorch 2.11

For shape `[4, 2049, 8, 8]`:

| Batch | Absolute eager/compiled difference |
|---:|---:|
| 0 | `1.5259e-05` |
| 1 | `5.7245e-01` |
| 2 | `2.1724e+00` |
| 3 | `8.7196e+00` |

Across batch sizes 2, 3, 4 and 8, every batch except batch 0 was wrong. Breaking the problematic fusion via an equivalent reduction form or no-op boundary reduced max difference to `3.0518e-05`. FP64 still showed max difference `1.1381e+01`.

### Current-nightly result

For the identical primary case, absolute differences were:

```text
[1.5259e-05, 0, 1.5259e-05, 3.0518e-05]
```

No bad batch was found for batch sizes 2, 3, 4 or 8. FP64 max difference was `5.6843e-14`.

### Oracle interpretation

- broken version: `REJECT` composite correctness because the exact batch-index/stride relation is violated;
- current version: the known wrong-stride witness is absent and it is a matched negative control;
- current nonzero fp32 differences are not a correctness rejection by exact equality;
- a universal/current-version numerical `ACCEPT` would still require a certified envelope and broader inputs.

### New finding

The very conservative generic fp32 reduction bound can be too wide to detect a wrong-stride output. The exact indexing obligation must be evaluated before the numerical reduction envelope. FP64 and generated-code/root-cause evidence make the structural violation unambiguous in the broken version.

## 3. Witness W2 — `_foreach_sub` ignores `alpha`

Artifact: `paper/confirmed_bugs/pytorch/bug_055_foreach_sub_alpha_ignored/reproduce.py`.

### Contract

```text
subject: torch._foreach_sub API computation
exact option obligation: alpha=2.0 participates in every result
mathematical relation: output_i = a_i - 2*b_i under declared floating expression
negative contrast: alpha omitted means a_i - b_i and is not an allowed substitute
```

### Broken result, PyTorch 2.11

Compiled output matched the `alpha=1`/no-alpha output bitwise for all three tensors. Maximum candidate-versus-correct differences were:

```text
1.8688, 1.4400, 1.5892
```

The `torch._foreach_add(..., alpha=2)` contrast matched eager exactly.

### Current-nightly result

Compiled `_foreach_sub(..., alpha=2)` matched eager exactly for all tensors. Its differences from the no-alpha output were `1.8688`, `1.4400`, and `1.5892`.

The legacy reproducer terminated with an assertion because it asserts that the bug must still exist. That assertion failure is expected for a fixed negative control; the printed semantic relations establish the result.

### Oracle interpretation

- broken version: `REJECT` exact option/expression contract;
- current covered cases: `ACCEPT` the alpha-preservation witness;
- the verdict does not depend on selecting a raw-delta tolerance;
- broad family acceptance still requires held-out tensors/dtypes/signatures and execution-identity evidence.

## 4. What has now been empirically validated

- a confirmed exact/composite violation produces `REJECT`;
- a fixed version on the same input removes the violation;
- an exact option relation can distinguish the intended computation from a plausible wrong computation;
- nonzero fixed-version floating delta need not be rejected;
- contract precedence matters: structural/option semantics come before floating tolerance;
- region/program subject labels can be preserved without claiming a constituent operator cause.

## 5. What remains unvalidated

- held-out bug/control evaluation after freezing the scoring protocol;
- positive controls with small raw delta and conforming controls with large raw delta;
- formal `INDETERMINATE` and `INVALID` witness rates in one unified run;
- stochastic law conformance;
- `R4` in-program operator correspondence;
- quantitative improvement over raw delta/allclose baselines;
- external validity across the full operator-family matrix.

## 6. Decision

These runs move Operator Oracle v0.1 from purely logical examples to two real broken/fixed contract pairs. They do not complete the validation standard. The next evidence should be predeclared held-out confirmed bugs and matched allowed/fixed controls, explicitly including hard raw-delta quadrants.
