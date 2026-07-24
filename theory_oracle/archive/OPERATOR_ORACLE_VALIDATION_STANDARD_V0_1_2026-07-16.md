# Operator Oracle Validation Standard v0.1 — 2026-07-16

> Purpose: define evidence required before calling Operator Oracle v0.1 a useful general testing Oracle rather than a coherent contract language.

## 1. Validation target

The Oracle claims incremental value over raw eager/compiled numerical delta by using semantic envelopes, truth references, structured geometry, population contracts and validity/identity gates.

Validation must test that claim. It is insufficient to show that the Oracle can detect examples selected because they already forked.

## 2. Ground-truth case classes

Build cases from independently labeled semantic relations, not from the Oracle score being evaluated.

### Positive controls — must reject

- exact structural/index/control violations;
- outputs outside certified numerical envelopes;
- probability-law/support violations;
- incomplete or wrong state transitions;
- independently confirmed compiler wrong-code cases;
- contract-violating mutations whose label follows from the specification.

### Negative controls — must not reject correctness

- different but explicitly allowed tie outcomes;
- floating results inside a non-singleton/certified accuracy envelope;
- equivalent non-unique decompositions under proper residual/subspace geometry;
- same target stochastic law with different unspecified RNG coupling;
- reference/eager worse while candidate satisfies truth-relative contract;
- specification-preserving transformations.

### Invalid controls — must refuse evidence

- fallback mislabeled as candidate;
- unmatched operands/state;
- observation-induced recompilation changing the realization;
- missing/nonfinite cases handled contrary to protocol;
- region evidence mislabeled as operator evidence.

### Indeterminate controls — must abstain

- confidence set overlaps the acceptable boundary;
- rare-event sample size cannot resolve the declared rate;
- floating API has no independent envelope/margin;
- candidate realization correspondence is not identified.

## 3. Required hard quadrants

Any validation suite must contain all four discrepancy/conformance combinations:

| Raw discrepancy | Semantic conformance | Why necessary |
|---|---|---|
| small | conforming | ordinary true negative |
| large | conforming | tests raw-delta false positives and invariant/set-valued geometry |
| small | violating | tests exact/control/common-mode and direction-sensitive failures |
| large | violating | ordinary true positive |

Also include **zero relative discrepancy with shared wrong behavior**. Only independent truth/specification can catch it; differential metrics cannot.

## 4. Discovery/confirmation separation

1. use discovery cases to refine contracts and find missing semantic fields;
2. freeze contract version, geometry, envelope, thresholds and verdict logic;
3. evaluate on held-out inputs, signatures, configurations and mutation/bug cases;
4. record every exclusion, invalid observation and abstention;
5. do not redefine a failing control after seeing confirmation outcomes unless a new contract version is created and retested.

## 5. Baselines

At minimum compare against:

- max absolute delta;
- relative/scale-normalized delta;
- tensor norm delta;
- eager/compiled allclose under conventional fixed tolerances;
- global signed mean shift;
- paired event/fork indicator where applicable;
- truth-relative error alone when truth exists.

The Oracle's advantage cannot be claimed merely because a weak or improperly tuned baseline was selected.

## 6. Evaluation metrics

Because the Oracle can abstain and has scoped claim levels, one accuracy number is inadequate. Report:

```text
false rejection rate on independently labeled conforming cases
detection rate on independently labeled violations
invalid-control refusal rate
indeterminate/abstention rate
selective error among determinate verdicts
coverage by operator family/signature/configuration
calibration of population confidence statements
ranking/triage value versus raw delta
```

For universal correctness witnesses, report detection directly rather than diluting them into workload frequency. For workload contracts, respect predeclared weights.

## 7. Mutation validity

A mutation is useful only if its semantic label is independently known.

- **positive mutation:** provably moves behavior outside the contract envelope;
- **negative mutation:** changes implementation/output representation while remaining inside the envelope;
- **ambiguous mutation:** excluded from accuracy scoring or labeled unknown.

Mutation magnitude must span small and large raw deltas. Otherwise the experiment cannot determine whether the Oracle adds anything beyond delta magnitude.

Compiler configuration changes are not automatically positive mutations; many are legal implementation choices.

## 8. External-validity matrix

Held-out confirmation should cross:

- exact/discrete, numerical, stochastic and transition relations;
- structural, reduction, contraction, normalization, selection/routing, backward and stateful families;
- deterministic and nondeterministic protocols;
- nominal and stress input populations;
- multiple shapes/dtypes/layouts/devices/precision modes;
- isolated, region and identified operator realization levels.

Coverage gaps remain explicit. A successful `argmax`/`sum` suite cannot support a universal DL-compiler Oracle claim.

## 9. Causal-attribution validation

For repair/injection claims include controls where:

- intervention changes only boundary value and should recover a known effect;
- non-target fusion/layout also changes and must be downgraded;
- two operators interact nonadditively;
- substitute causes make one repair ineffective despite causal relevance;
- injected discrepancy is harmless in reference context but harmful in candidate context.

Score whether the framework assigns the correct `I0--I3` claim level, not only whether an endpoint changed.

## 10. Success criteria

The Oracle earns a validated-scope claim only if, on held-out cases:

1. exact/specification violations are caught with no averaging-away;
2. conforming large-delta cases materially reduce raw-delta false positives;
3. small/zero-relative-delta violations demonstrate information not present in differential magnitude;
4. invalid and unidentified cases are refused rather than silently scored;
5. abstention is reported and not counted as correctness;
6. results reproduce across predeclared input/signature strata within the claimed scope;
7. operator/region/impact claim levels remain correct under intervention controls.

Numerical target values for “materially” must be preregistered for a concrete validation study; they cannot be chosen from confirmation results.

## 11. Kill criteria

Reject or narrow the general Oracle claim if:

- verdict ranking is effectively identical to a simple raw delta baseline;
- semantic geometry reduces no false positives on allowed large differences;
- truth/specification catches no violations beyond differential comparison;
- most floating contracts remain uninstantiated, making claimed coverage nominal;
- abstention/invalid rates are hidden or treated as passes;
- contract performance fails on held-out signatures/configurations;
- compiled realization identity cannot be established at the claimed operator scale;
- mutation labels depend on the same Oracle being validated;
- repair/injection claim levels collapse under fusion/interactions.

## 12. Current status

The project has partial confirmation evidence: real exact/backward/metadata violations, matched fixed cases, an eager-wrong/compiled-right truth case, a shared-wrong zero-discrepancy case and validity/scope refusals. Separately frozen supplements now add a real CUDA sum pair that fails default allclose while both outputs remain inside a precomputed analytical envelope, and a stochastic 100-draw control that correctly returns `INDETERMINATE` under a predeclared law margin/confidence rule. The complete unified success gate has not been run; the preregistration audit still applies to the original schema and the supplements do not retroactively repair its missing fields.

The correct claim remains:

> a logically audited, framework-mapped Operator Oracle v0.1 with a usable exact core and partial incremental-value evidence—not yet a validated general-purpose compiler testing Oracle.
