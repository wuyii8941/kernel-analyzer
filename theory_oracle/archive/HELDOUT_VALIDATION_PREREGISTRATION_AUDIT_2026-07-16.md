# Held-out Validation Preregistration Audit — 2026-07-16

## 1. Audit conclusion

`HELDOUT_OPERATOR_ORACLE_VALIDATION_CONTRACT_V0_1_2026-07-16.md` successfully froze the validation **questions, case families, expected verdict classes and anti-post-hoc rule**. It did not freeze every concrete input, margin, sample budget, confidence method or candidate artifact required for aggregate scoring.

It should therefore be described as a **precommitted validation schema**, not a complete executable preregistration manifest.

This distinction matters: the Oracle forbids selecting an acceptable set from the outputs it judges. A validation contract must obey the same rule.

## 2. Field audit

| Cell | Independent relation | Concrete artifact/input | Acceptable boundary | Uncertainty rule | v0.1 status |
|---|---|---|---|---|---|
| H1 | yes, explicit cast semantics | issue artifacts/minimal input recoverable | exact | not needed for witness | executable positive; fixed negative missing |
| H2 | yes, exact indexing | executable artifact | exact | not needed for witness | executable positive; fixed negative missing |
| H3 | yes, backward relation | executable artifact | exact | not needed for witness | broken/fixed executable |
| H4 | yes, boundary-gradient relation | minimal input recoverable | exact | not needed for witness | intended path inapplicable on available T4 |
| H5 | yes, metadata/state relation | executable artifact | exact | not needed for witness | broken/fixed executable within capability scope |
| H6 | yes, stable mathematical relation | executable artifact and tail inputs | truth-relative | covered rounding interpretation | executable scoped numerical witness |
| N3 | yes, set-valued top-k ties | exact operand not frozen | exact allowed set | not needed | mechanics instantiable; not fully preregistered compiler case |
| N4 | analytical family named | operand/realized algorithm absent | envelope not instantiated | absent | `UNINSTANTIATED` until versioned manifest |
| U1 | formula/no-allowance condition | operator class named; exact input absent | deliberately absent | not applicable | expected `UNINSTANTIATED`; empirical refusal now demonstrated |
| U2 | stochastic class named | target law and observations absent | equivalence margin absent | sample size/CI absent | validation cell itself `UNINSTANTIATED` |
| hard quadrants | constructions named | exact scale grid absent | baseline thresholds absent | scoring details partial | cannot support aggregate comparison yet |

## 3. What current evidence may support

Current evidence supports scoped statements:

- exact membership contracts catch real cast/index/backward/metadata violations;
- endpoint-aware H3 ordering adds information beyond forward raw delta;
- truth-relative H6 evidence prevents treating eager as truth;
- identity gates refuse fallback/inapplicable paths;
- set-valued semantics can accept a large raw representation delta;
- missing numerical allowances correctly produce `UNINSTANTIATED`.

It does not support a detection rate, false-rejection rate or general superiority over raw delta across operator families.

## 4. Requirements for an executable confirmation manifest

Before aggregate scoring, a new version must freeze, per row:

```text
case id and immutable artifact hash/path
framework/build/hardware applicability rule
exact input or deterministic input generator and seed
subject/signature and candidate identity evidence
contract source and semantic envelope
population acceptable set or exact relation
sample/repeat budget and simultaneous confidence construction
raw-delta baseline definitions and thresholds, if classification is scored
missing/crash/fallback/nonfinite disposition
expected verdict class without observing candidate output
```

This is not an implementation plan; it is the minimum information needed for the claimed 0–1 decision to have a fixed meaning.

## 5. Fail-closed rule

No value introduced after inspecting a case's candidate output may be counted as preregistered for that case. Such a value can define a new Oracle version, but the case then returns to discovery/calibration and must be evaluated on fresh held-out evidence.

## 6. Subsequent separately frozen supplements

This audit triggered two new manifests rather than retrospective edits:

- a fresh-scale CUDA sum confirmation froze `[2^25,2,-2^25]` and its `gamma_2` envelope before execution, then observed a real eager/compiled default-allclose failure with both results conforming;
- a multinomial confirmation froze target law, TV margin, `n=100` and a 95% Hoeffding rule before draws, then correctly returned `INDETERMINATE`.

They validate the missing mechanics but remain separate from original v0.1 aggregate scoring.
