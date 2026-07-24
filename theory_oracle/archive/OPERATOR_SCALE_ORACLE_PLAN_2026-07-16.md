# Operator-scale Discrepancy Oracle Plan — 2026-07-16

> Status: operator is the selected analysis scale. The parameterized decision relation is now defined in `OPERATOR_ORACLE_V0_1_DEFINITION_2026-07-16.md`; O-1 remains open for concrete operator-class geometry and independently justified acceptable-set instantiation. Bias/variance profiles alone remain measurement outputs.

## 1. Core decision

The primary analysis scale is **operator**.

Three roles must remain separate:

- **operator**: the semantic computation whose discrepancy is the Oracle object;
- **operator input case**: one real operand/context sample used to estimate bias and variance;
- **region**: an optional multi-operation intervention unit when an experiment cannot alter one operator in context.

A region result never becomes an operator result merely because the region contains that operator. A physical kernel is an implementation artifact and may implement one, several or part of an operator.

## 2. Operator identity

### 2.1 Primary unit: operator instance

An operator instance is defined by:

```text
semantic operator
+ program call site / role
+ forward, backward or optimizer phase
+ attributes
+ shape, dtype, stride/layout and device signature
+ compiler configuration
```

The same operator name with different signatures is not automatically one statistical subject.

### 2.2 Secondary unit: operator family

Operator-family claims aggregate instances only after declaring which signature/context dimensions are conditioned, stratified or sampled. “Reduction,” “matmul” or “cast” is a family label, not a bias/variance conclusion.

### 2.3 Input case, not “state scale”

For operator `o`, an input case `x` contains the actual operand tensors, attributes and relevant RNG/control context observed during training or inference. The earlier full training state is only the provenance needed to reproduce `x`; it is not the Oracle scale.

## 3. Operator-level estimand

Let the two implementations of operator `o` produce:

```text
Z_R,o(x, r)
Z_C,o(x, r)
D_o(x, r) = Z_C,o(x, r) - Z_R,o(x, r)
```

under a declared operator-input population `Q_o` and repeat protocol.

The primary objects are:

```text
B_o = E_{X~Q_o,r}[D_o(X,r)]                  relative operator bias
H_o = Cov_X(E_r[D_o | X])                    input-conditioned heterogeneity
N_o = E_X[Cov_r(D_o | X)]                    within-input runtime variability
```

`B_o` may be a tensor/vector or a predeclared invariant representation. `H_o` and `N_o` are covariance structures or declared functionals. No output-coordinate average is required.

Without an independent mathematical reference, `B_o` is implementation-relative operator bias, not correctness bias.

## 4. Two operator questions that must not be merged

### A. Operator behavior discrepancy

Compare eager/reference and compiled implementations of the same declared operator on real operator inputs.

This estimates the operator's input-output discrepancy law under `Q_o`. Isolated replay is admissible for this question if execution identity and compilation contract are explicit.

### B. In-program operator contribution

Ask whether changing operator `o` inside the full compiled program changes a downstream numerical, semantic or transition endpoint.

This is a causal intervention question. Fusion or scheduling can make a single-operator intervention unidentified. Isolated operator discrepancy does not prove in-program contribution; region repair does not prove a particular operator caused the effect.

The Oracle reports these as separate fields.

## 5. Revised phases

### O-1. Define the operator Oracle decision relation

Before collecting formal operator data, freeze:

- exact, behavioral-equivalence, specification-backed or impact relation;
- estimand map `theta_o = T(Law(D_o))`;
- independently justified acceptable set `A_o`;
- confidence/power protocol;
- `invalid / equivalent / non-equivalent / indeterminate` verdict semantics;
- aggregation from operator instance/signature to operator family.

**Gate O-1:** without `A_o` and a verdict map, subsequent outputs are explicitly labeled measurement-only and cannot be called an instantiated Oracle.

**Current O-1 status:** the common relation registry and verdict map are frozen in Operator Oracle v0.1. No concrete operator contract has yet passed O-1 because operator-specific `geometry` and `A_o` have not been justified.

### O0. Freeze the operator comparison contract

For every admitted operator instance, declare:

- semantic operator identity and phase;
- signature and configuration;
- reference and candidate implementations;
- real input population `Q_o` and its checkpoint/model coverage;
- output correspondence and geometry;
- algorithmic/runtime randomness and coupling;
- specification strength;
- whether only behavior discrepancy or also in-program contribution is claimed.

**Gate O0:** if the expected operator relation or output correspondence is undefined, no operator Oracle is emitted.

### O1. Calibrate operator execution and input sampling

- reference/reference and candidate/candidate self-pairs;
- proof that the intended candidate operator implementation executed;
- no fallback or configuration-label mismatch;
- real inputs sampled without selecting known forks;
- discovery and held-out input/checkpoint clusters;
- separate operator instance and family identifiers.

**Gate O1:** failed execution identity or biased input collection invalidates the observation.

### O2. Estimate operator bias and variance structure

For each instance/signature:

- relative bias tensor/vector or invariant projections;
- input-conditioned heterogeneity;
- exact-input runtime variability;
- covariance, tails and rare extremes;
- sensitivity to checkpoint/model/configuration;
- finite-input/checkpoint uncertainty.

**Gate O2:** every “variance” names the varying axis; output coordinates are not pseudo-replicates.

### O3. Validate across operator families

Cover numerical structures rather than a few convenient fork examples:

- elementwise and cast/precision transformations;
- reductions and normalization;
- tensor contractions such as matmul/conv;
- transcendental/softmax-like computation;
- indexing, ranking or routing operations;
- stochastic operations where in scope;
- backward and optimizer/control operations for training claims.

This taxonomy is a coverage plan, not a prior assignment of bias or variance.

**Gate O3:** family claims require held-out instances/signatures and cannot be inferred from one call site.

### O4. Connect operator discrepancy to downstream impact

Track whether `D_o` is associated with or projects into:

- later operator outputs;
- model output/loss;
- semantic decision law;
- gradient/update/next state.

Distinguish:

- discrepancy-generating operator;
- discrepancy-propagating operator;
- amplifying/suppressing operator;
- boundary-converting operator.

Association is descriptive. A causal contribution requires O5.

### O5. Test in-program operator contribution

Use single-operator repair/injection only when non-target computation and the original compiled endpoint remain under the declared integrity relation.

Apply the `R0--R4` subject-correspondence and `I0--I3` intervention levels from `OPERATOR_REALIZATION_IDENTITY_CONTRACT_V0_1_2026-07-16.md`. Isolated behavior, region conformance, boundary-value effect and operator-realization causal effect are separate claim levels.

If fusion prevents this:

- report operator contribution as `not identifiable under current intervention`;
- a region intervention may be reported separately as region sensitivity;
- do not divide a region effect among contained operators without an additional causal design.

**Gate O5:** removal of an endpoint difference is not sufficient for unique operator necessity or root cause.

### O6. Freeze the operator Oracle product

For each operator instance/family, emit:

```text
OperatorOracle = {
  contract_id_and_version,
  operator_identity_and_signature,
  input_population,
  expected_relation_and_reference_status,
  geometry_and_acceptable_set,
  execution_validity: VALID | INVALID,
  behavioral_verdict: ACCEPT | REJECT | INDETERMINATE | UNINSTANTIATED,
  claim_level: CORRECTNESS | BEHAVIORAL_EQUIVALENCE,
  conformance_estimands_and_joint_uncertainty,
  discrepancy_decomposition: {relative_bias, input_heterogeneity, runtime_variability},
  violated_or_unresolved_constraints,
  impact_verdict: ACCEPT | REJECT | INDETERMINATE | NOT_TESTED | NOT_IDENTIFIABLE,
  coverage_scope
}
```

No single scalar or fork flag replaces this result. The profile fields explain the explicit contract verdict; they are not a substitute for it.

## 6. Immediate next study

The next study is **operator-contract instantiation**, not undirected discrepancy collection or full-system instrumentation.

1. choose a coverage-balanced semantic set containing exact, numerical, distributional and transition contracts;
2. for each selected operator instance, define its expected relation and reference status;
3. justify a semantic output geometry and acceptable set without using the candidate observations being judged;
4. construct counterexamples that should clearly pass, reject, remain indeterminate and become invalid;
5. only after the contract survives those cases, freeze its real input population and collect confirmation evidence;
6. evaluate whether instance-level verdicts generalize to held-out signatures before making a family claim;
7. defer in-program repair until a true single-operator integrity design exists.

The immediate purpose is to demonstrate that v0.1 can be instantiated into understandable verdicts for materially different operator semantics. Operator-scale feasibility and bias/heterogeneity measurement follow that definition gate.

## 7. Kill criteria

- semantic operator identity cannot be matched across the two declared implementations;
- compiled behavior cannot be separated from fusion context even for the behavior question;
- real operator input population cannot be defined without fork/event selection;
- apparent operator bias fails held-out input/checkpoint reproduction;
- operator signature/context determines the result so completely that the proposed family claim is meaningless;
- operator profiles add no information beyond end-to-end discrepancy baselines;
- in-program contribution remains unidentifiable and is relabeled as region or kernel root cause;
- no truth exists but implementation-relative bias is presented as correctness error.

## 8. What this plan deliberately avoids

- treating region as a synonym for operator;
- treating physical kernel names as semantic operator identity;
- making the training state the analysis scale;
- tracing every operator before feasibility is established;
- selecting only clipping/sampling/fork cases;
- assuming reduction is bias or floating point is variance;
- equating local discrepancy with harmful training impact.
