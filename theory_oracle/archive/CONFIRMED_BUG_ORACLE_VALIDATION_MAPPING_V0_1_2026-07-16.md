# Confirmed-bug Mapping for Operator Oracle Validation v0.1 — 2026-07-16

> Purpose: reuse independently confirmed DeepOPFuzz bugs as Oracle validation controls. Confirmation/fix status is taken from the local `paper/confirmed_bugs/BUG_MAPPING.md` and `paper/bug_status_audit_2026-06-10.md`; this document does not claim a fresh issue-status audit.

## 1. Why these cases are stronger than arbitrary mutants

An arbitrary implementation mutation is not automatically wrong. The confirmed-bug corpus provides stronger labels through one or more of:

- an exact API/metamorphic relation;
- high-precision or cross-implementation mathematical evidence;
- generated-code root cause;
- maintainer triage/bug labeling;
- accepted fix or fix PR.

The Oracle must still reconstruct each case's contract independently. “Maintainer confirmed” is benchmark-label evidence, not a substitute for writing the violated relation.

## 2. High-value PyTorch controls

| Case | Independent violated relation | Contract family | Expected Oracle behavior |
|---|---|---|---|
| #179561 bf16/fp16 cast round-trip elided | explicit precision conversion is observable and cannot be erased | cast/exact representation + composite numerical | reject; raw magnitude may vary but relation violation remains |
| #179931 adaptive-pool + flatten + sum stride | flattened batch indexing must preserve actual stride/values | composite structural/index + reduction | reject wrong batches; batch 0 is a useful within-case negative control |
| #180164/#180771 slice-scatter backward | documented scatter/backward relation and shape/stride semantics | backward + exact indexing | reject wrong/all-zero gradient relation |
| #182012 diagonal-scatter backward | expanded/diagonal update relation must preserve full gradient structure | backward/state relation | reject even if a coarse gradient norm were misleading |
| #183986 expanded index family wrong result | exact indexing/gather/scatter relation | exact/discrete | reject without floating tolerance |
| #184415 foreach-sub alpha ignored | result must equal `a - alpha*b`, not `a-b` | exact option semantics + numerical expression | reject; tests configuration/keyword preservation |
| #185472 hardtanh bf16 boundary grad | documented/decomposition boundary convention must match at admitted bf16 values | backward boundary/state-conditioned | reject boundary witnesses; nearby non-trigger values are negative controls |
| #181581 create-graph metadata dropped | `requires_grad`/higher-order differentiation state is observable | exact metadata/transition | reject even if forward numerical outputs match |
| crash-class cases such as cumsum split-scan | admitted input should produce specified output rather than compiler crash | availability/exception semantics | reject exact-core behavior when input is valid |

These cases cover deterministic bias, signature-conditioned effects, exact state, backward semantics and composite fusion/codegen errors. They should not all be collapsed into one “numerical delta” class.

## 3. Cross-framework controls that challenge eager-as-truth

| Case | Key structure | Oracle lesson |
|---|---|---|
| TensorFlow #116943 SELU/ELU cancellation | eager uses `exp(x)-1` and loses small values; XLA `expm1` is closer/correct | compiled-reference disagreement can favor compiled; truth-relative geometry is necessary |
| TensorFlow complex sign underflow | eager magnitude computation underflows while another implementation/high precision is correct | common baseline can be wrong in an input-conditioned tail |
| TensorFlow #116436 monotonic ArgMin rewrite | non-strict monotonic transform changes tie structure | set/tie semantics beat raw numerical closeness |
| TensorFlow fake-quant rounding | exact rounding/nudge semantics at boundaries | representation contract and boundary witnesses |
| TensorFlow/XLA NaN/special-value cases | position/domain-specific NaN semantics differ | special values require explicit finite relations, not generic tolerance |
| TVM cumsum axis / gather negative index / scatter semantics | exact axis/index relation | cross-framework exact-core positive controls |
| TVM NaN propagation | mathematical/special-value relation | tests whether nonfinite cases are counted rather than dropped |

These cases help prevent the Oracle from becoming PyTorch-eager-specific.

## 4. Mapping to validation hard quadrants

### Large discrepancy + violating

- wrong-stride adaptive-pool/reduction;
- ignored alpha;
- all-zero backward from stride/meta failures.

### Small discrepancy + violating

- cast-elision inputs chosen near representable boundaries;
- hardtanh bf16 boundary cases;
- exact metadata/index/control mismatch even when floating output change is tiny.

The positive label comes from the exact relation, not from choosing a magnitude threshold.

### Large discrepancy + conforming

Use independently allowed controls rather than confirmed bugs:

- top-k tied-index alternatives;
- valid eigenvector sign/subspace alternatives;
- different legal reduction/matmul outputs inside a certified envelope;
- same stochastic law with different unspecified coupling.

### Small discrepancy + conforming

Ordinary outputs safely inside numerical envelopes and exact relations.

### Zero relative discrepancy + wrong

Construct or select shared-wrong cases checked against independent truth. TensorFlow eager-wrong/XLA-right cases already demonstrate why the corpus must not define truth by majority or eager status; shared-wrong controls require a third truth source.

## 5. Positive-control contract requirements

For every selected bug, record before evaluation:

```text
governing API/mathematical relation
admitted input/signature and excluded undefined behavior
truth or exact metamorphic construction
candidate/reference versions and expected fixed/broken status
subject level: operator, composite region or full program
expected conformance verdict
expected realization/impact claim level
raw-delta baselines
```

A root cause in fusion/codegen may make the proper subject a composite region rather than one source operator. The benchmark remains useful even when the correct Oracle output is `REGION REJECT` plus `operator NOT_IDENTIFIABLE`.

## 6. Negative controls from the same corpus

- fixed release/commit on the identical reproducer;
- patched kernel/lowering where one known root-cause change restores the relation;
- non-trigger shapes/strides/values specified by the bug mechanism;
- alternative correct backend validated against the same independent relation;
- allowed tie/precision variants that intentionally differ from eager.

These matched negatives are important: otherwise the Oracle might simply learn bug-directory or extreme-input artifacts.

## 7. Selection policy

Do not choose only cases that the current Oracle trivially catches. Predeclare a stratified sample across:

- exact/structural/index/control;
- cast and floating boundary;
- reduction/contraction/composite fusion;
- backward/gradient/state;
- special values;
- stochastic behavior where confirmed labels exist;
- silent wrong result and crash;
- operator, region and full-program realization levels.

Keep fixed, open and cross-framework cases separate in reporting. Bug status is not an input feature to the Oracle.

## 8. What this corpus can establish

It can provide independently labeled violation/negative controls and test raw-delta incremental validity. It cannot by itself establish natural-workload prevalence, training harm or broad operator-family coverage: confirmed bugs are deliberately stress-selected.

## 9. Revised empirical-validation order

1. use a small stratified subset to validate contract reconstruction and verdict mechanics;
2. freeze the extraction/scoring protocol;
3. evaluate a held-out confirmed-bug subset plus matched fixed/non-trigger negatives;
4. compare semantic-envelope verdicts with raw delta/allclose/fork baselines;
5. only then expand to real-model nominal populations for prevalence/impact.

This order uses strong bug ground truth for detector validity and real workloads for operational relevance, without confusing the two distributions.
