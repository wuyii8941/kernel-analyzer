# PyTorch Contract Records v0.1 — 2026-07-16

> Purpose: validate that the Oracle contract process can distinguish directly instantiable, set-valued, distributional and partially specified real API semantics. These records are validation probes, not the project's operator-coverage claim.

## 1. Record P1 — `torch.argmax`

### Authority and domain

The PyTorch 2.13 API states that `torch.argmax` returns the index of the maximum and, when multiple maximal values exist, returns the first maximal index. Source: [torch.argmax](https://docs.pytorch.org/docs/2.13/generated/torch.argmax.html), retrieved 2026-07-16.

Initial domain is restricted to finite, non-NaN input tensors with declared dimension/flattening and identical input bits to reference and candidate. NaN behavior remains outside this record unless separately specified.

### Contract

```text
source level: S1
relation: exact discrete
S_P1(x): singleton index specified by maximum + first-tie rule
acceptable population policy:
    universal correctness: zero valid mismatching witnesses
```

### Verdict meaning

- wrong index on a matched operand: `REJECT CORRECTNESS`;
- correct index for the covered input: `ACCEPT` for that case;
- unknown candidate execution or changed operand: `INVALID` for operator conformance;
- compiled/eager argmax disagreement caused by different upstream operands: not a P1 failure; report upstream impact.

This is a genuine Oracle with no floating tolerance.

## 2. Record P2 — `torch.topk`

### Authority and domain

The PyTorch 2.13 API specifies return of the `k` largest or smallest values and indices, but explicitly states that indices of tied elements are not guaranteed stable and may vary across invocations. Source: [torch.topk](https://docs.pytorch.org/docs/2.13/generated/torch.topk.html), retrieved 2026-07-16.

### Contract

```text
source level: S1
relation: exact set-valued
S_P2(x): every value/index result satisfying the documented top-k relation,
         with all documented tie alternatives admitted
```

For `sorted=True`, enforce documented ordering of returned values; do not invent stable tie-index ordering.

### Verdict meaning

- different tied indices may both `ACCEPT`;
- selecting an element strictly outside the valid top-k boundary gives `REJECT CORRECTNESS`;
- a bitwise/reference-equality Oracle would generate false positives here.

This record demonstrates why the semantic envelope may be a set rather than a singleton.

## 3. Record P3 — `torch.multinomial`

### Authority and domain

The PyTorch 2.13 API specifies sampling from rows of nonnegative, finite weights with nonzero sum, with or without replacement, and permits an explicit generator. It does not document one mandatory sampling algorithm whose same-seed token must be reproduced by every implementation. Source: [torch.multinomial](https://docs.pytorch.org/docs/2.13/generated/torch.multinomial.html), retrieved 2026-07-16.

### Contract

```text
source level: S1 for input/output structure and target sampling law
relation: distributional law
S_P3(x): documented weighted categorical/without-replacement law
replay relation: UNSPECIFIED unless a narrower API/backend contract is supplied
```

### Verdict meaning

- invalid output index, shape, or without-replacement duplication contrary to the documented relation: exact-core `REJECT`;
- conditional output-law mismatch: distributional evidence toward `REJECT`;
- same-seed token mismatch alone: not a correctness rejection;
- exact-law `ACCEPT` from finite sampling alone: unavailable; use `INDETERMINATE` unless formal/exhaustive evidence or an independently justified nonzero law-distance margin exists.

This record separates semantic law, RNG coupling, runtime variability and sampling uncertainty.

## 4. Record P4 — `torch.optim.SGD`

### Authority and domain

The PyTorch 2.13 API documents the SGD/momentum/weight-decay/Nesterov update algorithm and exposes for-loop, foreach and fused implementation choices. Source: [torch.optim.SGD](https://docs.pytorch.org/docs/2.13/generated/torch.optim.SGD.html), retrieved 2026-07-16.

### Contract split

```text
S1 exact/finite obligations:
    parameter/state structure, option-controlled branches,
    mutation and declared optimizer-state evolution

S2 numerical obligations:
    floating next-state fields relative to the documented update equation,
    under a declared dtype/arithmetic envelope

configuration identity:
    foreach/fused/device/dtype and all optimizer options
```

### Verdict meaning

- wrong branch, missing update, wrong state field or option semantics: exact-core correctness rejection;
- floating next-state mismatch versus eager: implementation discrepancy only;
- floating next-state outside a certified update envelope: numerical-conformance rejection;
- without that envelope or an external compatibility margin: floating acceptance is `UNINSTANTIATED`.

This record shows why a transition Oracle cannot be reduced to parameter norm and why implementation variants are configuration strata rather than repeat variance.

## 5. Cross-record sanity result

| Record | Envelope shape | Can finite evidence directly reject? | Can one finite success directly accept? |
|---|---|---|---|
| P1 argmax | singleton index | yes | for covered case only |
| P2 top-k | set-valued due to ties | yes | for covered case only |
| P3 multinomial | probability law | yes, with sufficient law evidence or exact-core witness | not exact law from finite samples |
| P4 SGD | product of exact and numerical transition relations | exact fields yes; numerical with envelope | exact fields for case; numerical only with envelope |

The differing answers are not inconsistency. They follow from different semantic objects and evidence strength.

## 6. What these records prove and do not prove

They prove that the contract method can map real APIs to nontrivial, common-sense verdicts and can refuse unsupported claims. They do not establish coverage of numerical operator families or validate compiled execution identity. Reduction, matmul/conv, transcendental, normalization, backward and mixed-precision contracts still require their own records and evidence.
