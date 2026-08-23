# Current main result

This file is the short, authoritative description of the project. Older round
notes are retained as experiment history and must not be used to change the
current case count.

## Question

When does a numerical difference introduced by one LLM training operator turn
into a parameter-update difference that keeps adding up instead of canceling?

## What is measured in one training step

The original implementation and the repaired implementation start a step from
the same weights and optimizer state. The repair changes only the operator
under test.

For a live pair of training runs, the change in their parameter difference is
split into two parts:

\[
D_{t+1}-D_t=L_t+B_t.
\]

- `L_t` is the update difference caused by the operator when both
  implementations are evaluated from the same current state.
- `B_t` is the extra difference caused by the two runs having reached
  different weights or optimizer states in earlier steps.
- `D_{t+1}-D_t` is the actual change in distance between the two training
  runs.

These three quantities answer different questions. A large actual separation
does not by itself prove that the operator's direct effect keeps pointing in a
consistent direction.

We use

\[
A_X(T)=\frac{\left\|\sum_{t=1}^{T}X_t\right\|}
{\sqrt{\sum_{t=1}^{T}\|X_t\|^2}}
\]

to describe how much a sequence cancels. At 32 steps its maximum is
`sqrt(32) = 5.66`: values near 1 look like diffusion, while much larger values
mean that the step differences reinforce one another. `A` is a continuous
measurement, not a universal pass/fail constant.

## Current headline

Three bounded records support the narrow claim that the direct operator or
backward effect can keep adding up:

- Liger fused cross entropy;
- Phi-4 `lm_head dX`;
- the Qwen `lm_head dX` family.

They are not three independent implementation mechanisms. Liger is a fused
accumulation example. Phi and Qwen are two models using the same mathematical
`lm_head` input-gradient matrix-multiplication family.

The 32-state measurements at three points are:

| measured record | operator output | parameter gradient | parameter update |
|---|---:|---:|---:|
| Liger fused CE, seq128 | 2.984 | 2.931 | 2.931 |
| Phi-4 `lm_head dX`, seq64 | 2.074 | 4.701 | 4.701 |
| Qwen `lm_head dX`, seq256 companion measurement | 1.008 | 1.698 | 1.698 |

Liger is already directional at the operator boundary. For Phi and Qwen, the
direction becomes clearer after the real backward pass reaches a parameter
gradient. Stateless SGD preserves it. The optimizer is not always passive:
on Phi, a gradient sequence with `A=4.665` becomes an AdamW update sequence
with `A=1.031` when both arms use the same stored moments.

The Qwen row needs an explicit boundary. The historical strict live trajectory
is seq128, while the three-stage row above and the short-screen evaluation are
seq256. They use the same mathematical repair family but are not one exact
endpoint instance. Until a single sequence length is rerun end to end, slides
and papers must say `Qwen lm_head dX family`, not attach the seq256 three-stage
numbers to the seq128 strict certificate.

Phi currently has the cleanest direct-effect/feedback separation. In its
16-state live evaluation following 16 calibration states, the direct operator
effect explains `0.9992` of the accumulated projection and the recurrence
residual is below `7.1e-9`. Liger and Qwen do not yet have the same exported
direct/feedback/actual certificate, so that result is not claimed for all
three cases.

## Two Phi measurements that must stay separate

- `A=4.701` is the 32-state same-state parameter-update difference caused by
  the repaired `lm_head dX` operator.
- `A=4.488` is the actual 32-step separation of the candidate and repaired
  final-norm parameter runs.

They concern the same Phi operator family and parameter, but come from two
different measurement protocols. They are not a range and must not be merged
into `4.49--4.70`.

In the four-arm comparison, only `model.norm.weight` is allowed to evolve:

| comparison | final parameter distance | `A` |
|---|---:|---:|
| candidate operator vs repaired operator | `9.186e-5` | `4.488` |
| different data order, same batch multiset | `3.548e-5` | `0.0067` |
| BF16 F+B vs FP32 F+B | `3.223e-4` | `1.857` |
| different RNG seed | `0` | not informative because dropout is zero |

The precision arm runs the full model forward and backward in BF16 versus
FP32, but updates only the declared final-norm parameter. It is therefore a
controlled one-parameter comparison, not full-parameter BF16-versus-FP32
training.

## Coverage: what was exhaustive and what was not

The four-model, three-sequence-length core contains:

- 466,419 eager calls;
- 70,171 calls from the compiled implementation under test;
- 186,807 units in which one forward calculation is linked to its real
  backward calculation;
- 1,562 concrete tensor locations selected for the first numerical test.

All 1,562 locations have a first numerical result: 1,390 pass and 172 reject.
Their forward and actual backward mathematics are bound. This is an
exhaustive census and first numerical screen; it is not 1,562 complete
32-step training experiments.

The core locations are also grouped into 804 groups according to generated
code region and the backward calculation that consumes the result. A later
search uses a different denominator: 791 operator-and-backward combinations,
deduplicated into 493 distinct implementation patterns. The numbers 804 and
791 describe different scopes and must not be treated as interchangeable.

Only rows that also have a valid one-operator repair, a reachable parameter,
and the required trajectory data enter deeper tests. Six historical records
pass the older strict repair/carrier/trajectory checklist. Three records enter
the narrower current headline above.

## Other forms of parameter separation

Not every real separation is a persistent direct operator effect.

- Some numerical differences are visible in one step but cancel over time.
- Some become directional after backward or optimizer processing.
- Some direct operator effects remain close to diffusion, while earlier
  parameter or optimizer differences cause the two training runs to keep
  separating.
- Missing repair, parameter reach, or trajectory evidence remains unresolved;
  it is not converted into a negative result.

In the mechanically selected 12-row control sample, 11 rows have a
near-diffusive direct operator effect and a persistent later feedback effect;
one row contains both. This is why final parameter distance cannot replace a
direct operator-persistence measurement.

## Short screening result

The frozen retrospective evaluation has 14 rows: three known positives and
11 nonzero controls. Using only the first 16 parameter-update differences:

- all three positives are selected;
- two of 11 controls are also selected;
- five of 14 rows are sent to the full test;
- precision is `3/5` and recall on this small set is `3/3`.

On the same set, the short persistence score has AUROC `1.00`; local error RMS
has AUROC `0.242`, and BF16 dtype alone has AUROC `0.50`. Across a separate 32-row
formation population, local RMS has Pearson correlation `0.018` (`p=0.921`)
and Spearman correlation `0.243` (`p=0.178`) with directionality.

This supports a cheap, fail-closed prioritization step: selected rows receive
the full test, while unselected rows are not declared safe. It is not a
universal all-operator classifier.

## Safe conclusion

Output tolerance and error size do not tell us whether an implementation
difference will keep changing model parameters in one direction. The relevant
chain is:

```text
operator output difference
-> parameter-gradient difference after the real backward pass
-> parameter-update difference after the optimizer
-> direct accumulation or cancellation
-> additional separation caused by changed training state
```

The repository supports this measured workflow and three bounded headline
records. It does not yet support a universal rule for unseen implementations,
full-parameter training failure, or a claim that every observed parameter
separation is caused by a persistent direct operator effect.

Machine-readable sources:

- `results/property/joint_bias_formation_v1/headline_case_evidence_scope_v1.json`
- `results/property/joint_bias_formation_v1/three_stage_summary.json`
- `results/property/bias_formation/consequence/phi4_lm_head_dx_seup.json`
- `results/property/joint_bias_formation_v1/source_persistence_reclassification.json`
- `results/property/joint_bias_formation_v1/oracle_baselines/frozen_evaluation_v2/comparison_v2.json`
- `results/coverage/coverage_table_v1.json`
