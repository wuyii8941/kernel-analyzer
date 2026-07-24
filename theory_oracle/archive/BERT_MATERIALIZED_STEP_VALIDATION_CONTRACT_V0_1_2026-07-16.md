# BERT Materialized-Step Validation Contract v0.1 — 2026-07-16

> Frozen after the two-state execution/schema smoke and before full discovery/confirmation runs. Smoke states are excluded from confirmation because they came from the discovery partition.

## 1. Question

Does the materialized deterministic BERT/SGD executor preserve complete-state validity and the covered exact transition core across frozen state banks, while correctly refusing numerical transition acceptance in the absence of an independent update envelope?

This validation does not test a numerical correctness claim.

## 2. Subject

```text
model: data/external_models/bert-tiny-sst2
forward/loss: BERT SST-2, eval mode, fp32 parameters, CUDA fp16 autocast,
              float32 cross entropy
candidate: torch.compile/Inductor, fullgraph=True, dynamic=False, tracked backend
optimizer: actual torch.optim.SGD step, lr=1e-5, momentum=0, weight_decay=0
GradScaler: absent/outside subject
sequence length: 64
repeats: 2 exact-state arms per implementation
seed: 20260716
```

## 3. State banks

```text
discovery:    all 128 states in data/external_datasets/sst2_discovery_128
confirmation: all 128 states in data/external_datasets/sst2_confirmation_128
```

The discovery bank includes the two smoke rows. The confirmation bank is untouched at freeze time. Banks are reported separately and never pooled into a deployment prevalence claim.

## 4. Fixed verdict fields

### Validity

Require:

- one stable compiled graph for the fixed signature;
- exactly one tracked candidate runtime invocation per measured candidate step;
- exact baseline state restoration before every arm;
- identical inputs/labels and coupled initial RNG;
- finite loss/logits and successful backward/optimizer step.

Any failure returns `INVALID` for the affected observation and fails the bank gate.

### Exact transition core

Require equality of:

- state-dict names, shapes and dtypes;
- gradient presence, shape and dtype;
- optimizer structure/options and empty no-momentum optimizer state relation;
- nonfloating next-state fields;
- mutable buffers under eval-mode subject;
- coupled next CPU/CUDA RNG state.

Violation gives `REJECT` for the covered exact relation. Passing gives scoped exact-core `ACCEPT`.

### Numerical transition

No independent gradient/update forward-error envelope is supplied. Every valid observation therefore returns:

```text
UNINSTANTIATED
```

Nonzero or zero eager/compiled parameter differences cannot change this verdict.

### Impact

Loss delta and prediction disagreement are descriptive. No application margin is supplied, so impact remains `NOT_INSTANTIATED`.

## 5. Measurements

For each state/repeat record:

- exact/candidate validity fields;
- loss signed delta and prediction disagreement;
- floating next-state L2 and maximum coordinate discrepancy;
- changed state-field count;
- exact-repeat stability.

## 6. Bank success gates

The mechanics validation passes a bank only if:

1. all candidate identities are valid and graph count/hash is stable;
2. all observations satisfy the covered exact core;
3. every numerical verdict remains `UNINSTANTIATED`;
4. both repeats are identical for every reported discrepancy/verdict field;
5. crashes, missing rows and nonfinite values are not dropped;
6. confirmation is interpreted without changing any field above.

Passing validates execution and refusal mechanics for this subject. It does not prove BERT numerical transition correctness.

## 7. Kill/narrowing criteria

- state reset or coupled RNG cannot be verified;
- compiler fallback/graph proliferation occurs;
- repeat instability appears under the deterministic protocol;
- exact-core violations are averaged into a numerical norm;
- a numerical pass is inferred from small delta;
- discovery and confirmation differ in an unreported configuration field.
