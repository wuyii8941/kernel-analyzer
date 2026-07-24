# BERT Training-Step Operator Coverage Ledger v0.1 — 2026-07-16

## 1. Subject

This ledger attaches the existing Operator Oracle contracts to the materialized BERT/SGD step without promoting source-module presence into in-program operator realization claims.

Subject artifacts:

- BERT-tiny SST-2 model in `data/external_models/bert-tiny-sst2`;
- materialized full-graph forward/loss/backward followed by shared host `torch.optim.SGD`;
- validation results in `results/training_step_oracle/bert_{discovery,confirmation}_v0_1`.

## 2. Source-module inventory

The loaded model contains:

| Module family | Count | Training-step role |
|---|---:|---|
| `Linear` | 14 | attention projections, FFN, pooler, classifier |
| `LayerNorm` | 5 | embedding and encoder normalization |
| `Embedding` | 3 | word, position and token-type lookup |
| `GELUActivation` | 2 | encoder nonlinearities |
| `Dropout` | 8 | inactive under the declared eval-mode subject |
| `Tanh` | 1 | pooler activation |

Functional attention softmax, matrix products, elementwise/residual operations and cross entropy are also present in the semantic program but are not separately enumerated by module count here.

This is a source-level inventory, not a physical kernel inventory.

## 3. Realization identity

The tracked candidate compiled exactly one stable 134-node graph with the same graph hash in discovery and confirmation. This establishes an executed composite region with stable input/loss/logit boundary:

```text
whole compiled core: R3 region realization
constituent source modules/operators: R1 source/graph presence at most
R4 in-program operator correspondence: not established
```

No intermediate observation was introduced because doing so could change fusion/decomposition. Consequently, the ledger deliberately issues no constituent compiled-operator behavior verdict.

## 4. Contract coverage by family

| Family | Available semantic contract | Current numerical envelope | Identity/verdict |
|---|---|---|---|
| embedding/index lookup | exact indices, shape/dtype and lookup relation | not needed for exact isolated relation | source contract known; in-program candidate operator `NOT_IDENTIFIABLE` (R1) |
| linear/matmul | documented contraction relation and conditional precision-aware geometry | realized accumulation/precision schedule not certified | `UNINSTANTIATED`; R1 |
| LayerNorm | documented axes, epsilon, affine and biased-variance convention | no certified composite propagation bound | exact source fields known; numerical `UNINSTANTIATED`; R1 |
| attention softmax | normalized-exponential formula and exact dim/domain fields | API has no universal quantitative accuracy allowance | numerical `UNINSTANTIATED`; R1 |
| GELU/tanh | documented mathematical mapping/domain | no operator/backend quantitative envelope | numerical `UNINSTANTIATED`; R1 |
| residual/elementwise | algebraic source relation | fused precision/cast schedule not certified | region only; constituent `NOT_IDENTIFIABLE` |
| cross entropy/log-softmax | target-index and reduction relation | no frozen quantitative envelope | numerical `UNINSTANTIATED`; region boundary only |
| backward | gradient structure and source derivative relations | no full-network gradient error envelope | exact gradient structure accepted at step boundary; numerical `UNINSTANTIATED` |
| SGD | documented no-momentum update/options and empty state | no gradient/update numerical envelope | shared host optimizer, not a distinct compiled operator; exact step fields accepted |
| dropout | identity under eval mode | none required | covered as whole-step mode/RNG obligation; no executed stochastic operator claim |

## 5. Coverage summary

```text
source module families inventoried:                 yes
whole compiled region identity:                     R3
constituent in-program operator correspondence R4:  0 established
operator-level numerical ACCEPT/REJECT verdicts:    0
step exact-core verdicts:                            256/256 ACCEPT across both banks
step numerical verdict:                              UNINSTANTIATED
```

The zero operator verdict count is not a failed experiment. It is the correct result of the realization-identity gate.

## 6. What can currently be claimed

- one stable compiled BERT forward/loss/backward region was executed;
- source operator families and their applicable contract forms are known;
- the materialized step exact core accepted across both state banks;
- numerical next-state discrepancy is real and state conditioned;
- the evidence does not identify which constituent operator generated it.

## 7. Next operator gate

A constituent operator may receive a behavior verdict only through one of:

1. an R2 isolated candidate realization on operands captured without claiming original fused behavior;
2. an R4 provenance/correspondence certificate for the in-program realization;
3. an R3 composite region contract with no constituent promotion.

Repair/injection remains separate. A configuration change is I0; a stable region replacement is I1; neither becomes a unique operator cause.
