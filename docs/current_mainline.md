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

## Historical stateless-SGD result

Three bounded records show that the direct operator or backward gradient
difference can keep adding up when parameter updates use stateless SGD:

- Liger fused cross entropy;
- Phi-4 `lm_head dX`;
- the Qwen `lm_head dX` family.

They are not three independent implementation mechanisms. Liger is a fused
accumulation example. Phi and Qwen are two models using the same mathematical
`lm_head` input-gradient matrix-multiplication family.

The 32-state measurements at three points are:

| measured record | operator output | parameter gradient | stateless SGD parameter update |
|---|---:|---:|---:|
| Liger fused CE, seq128 | 2.984 | 2.931 | 2.931 |
| Phi-4 `lm_head dX`, seq64 | 2.074 | 4.701 | 4.701 |
| Qwen `lm_head dX`, seq256 companion measurement | 1.008 | 1.698 | 1.698 |

Liger is already directional at the operator boundary. For Phi and Qwen, the
direction becomes clearer after the real backward pass reaches a parameter
gradient. Stateless SGD preserves it. The optimizer is not always passive:
on Phi, a gradient sequence with `A=4.665` becomes an AdamW update sequence
with `A=1.031` when both arms use the same stored moments.

The old Qwen row still remains a valid historical SGD-family result. A new
seq128 run now closes the exact endpoint, parameter and state order under
AdamW; its different result is reported below instead of being attached to
the seq256 SGD numbers.

## Same-optimizer correction

The earlier 14-row Oracle comparison was not fair: the three historical rows
above used stateless SGD, while the controls used AdamW. We therefore reran
all three historical rows with the same AdamW settings as the controls and
kept all twelve mechanically sampled controls. The corrected set has 15 rows.

| row | local update `A16` | local update `A32` | random-sign 95% bound | AdamW result |
|---|---:|---:|---:|---|
| Liger fused CE | 1.338 | 1.720 | 1.116 | persistent |
| Phi-4 `lm_head dX` | 1.013 | 1.029 | 1.004 | persistent, small margin |
| Qwen seq128 `lm_head dX` | 0.971 | 0.957 | 1.045 | canceling |
| sampled Phi seq256 row `0543` | 1.007 | 1.014 | 1.011 | persistent, small margin |

This changes the scientific interpretation. Qwen has aligned gradient
differences under the historical SGD measurement, but the direct update
differences cancel after AdamW. The optimizer is therefore part of the
measured training behavior, not an interchangeable final step.

The separate Phi stochastic-rounding experiment must not be attached to the
Phi AdamW row in this table. Its `A=3.325 -> 0.956` result uses 16 common
states and stateless SGD with no moments; the `A=1.029` result here uses a
32-step zero-moment AdamW trajectory. The former supports a source-level
intervention under SGD. A matched source intervention for the AdamW result
has not been run.

These data do not locate the defect inside AdamW. Directionality is already
strong at the Phi gradient before the optimizer, while AdamW suppresses most
of it. The supported conclusion is that the optimizer changes whether an
existing gradient bias reaches the parameter update; the small remaining
AdamW signal has not yet been assigned to a mechanism.

The new exact Qwen seq128 three-stage result is `A=1.005` at the operator
output, `A=1.343` at the parameter gradient, and `A=0.961` after AdamW. Its
live direct update has `A=0.957`; feedback has `A=1.191`, and actual parameter
separation has `A=1.508`.

All three historical rows now have an exported direct/feedback/actual split
under AdamW. Liger has direct `A=1.720`, feedback `A=3.494`, and actual
`A=3.489`, with recurrence error below `8.5e-8`. Phi has direct `A=1.029`,
feedback `A=5.075`, and actual `A=1.711`. Thus a persistent direct effect can
exist while later training feedback is much larger; final distance must not
be attributed entirely to the operator.

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

The different-seed row is not used as a diffusion baseline: this controlled
carrier has no dropout, so changing the seed produces no change.  The actual
repeated-diffusion baseline is the separate five-seed, every-step,
norm/support-matched random-injection experiment.  Its `A` values are
`0.870--1.037` (mean `0.959`) and are recorded in
`results/property/joint_bias_formation_v1/four_scale_arms/phi_repeated_random_null.json`.

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
the narrower historical stateless-SGD headline above. The common-AdamW
comparison is counted separately.

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

The corrected retrospective evaluation has 15 rows and one optimizer:
AdamW. It contains all twelve result-blind sampled rows plus the three
historical rows. Nominally, the full 32-step test finds three rows above their
row-level null: Liger, Phi, and sampled row `0543`; Qwen is not positive under
AdamW. After the predeclared twelve-row Holm correction, `0543` is an
unresolved candidate rather than a confirmed positive.

Using only the first 16 local parameter-update differences:

- all three AdamW positives are selected;
- two of 12 negatives are also selected;
- precision is `3/5` and recall on this small set is `3/3`;
- the short score has AUROC `0.944`;
- short-horizon update RMS has AUROC `0.528`.

The old 14-row AUROC `1.00` is withdrawn because it mixed SGD positives with
AdamW controls and omitted sampled row `0543` after its full result was known.
Across a separate 32-row formation population, local RMS still has Pearson
correlation `0.018` (`p=0.921`) and Spearman correlation `0.243` (`p=0.178`)
with directionality.

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

The repository supports this measured workflow, three bounded historical SGD
records, and a corrected same-AdamW evaluation with two confirmed rows plus
one unresolved candidate. Two of the historical rows remain positive under
AdamW; Qwen does not, while one result-blind sampled row is nominally a
small-margin positive but does not survive the predeclared Holm correction. It does not yet
support a universal rule for unseen implementations, full-parameter training
failure, or a claim that every observed parameter separation is caused by a
persistent direct operator effect.

Machine-readable sources:

- `results/property/joint_bias_formation_v1/headline_case_evidence_scope_v1.json`
- `results/property/joint_bias_formation_v1/three_stage_summary.json`
- `results/property/bias_formation/consequence/phi4_lm_head_dx_seup.json`
- `results/property/joint_bias_formation_v1/source_persistence_reclassification.json`
- `results/property/joint_bias_formation_v1/oracle_repair_v3/same_optimizer_oracle_v3.json`
- `docs/oracle_repair_v3.md`
- `results/coverage/coverage_table_v1.json`

## Direct Persistence Screen v4

The current short selector is named the **Cold-start AdamW Direct Persistence
Screen**. It is a follow-up selector, not a safety classifier. The v4 package
separates the three predeclared rows from the twelve result-blind rows, keeps
`0543` as an unresolved candidate after Holm correction, and reports signed
direct/feedback/actual contributions.

See [`docs/direct_persistence_screen.md`](direct_persistence_screen.md), the
[v4 summary](../results/property/direct_persistence_v4/summary.json), and the
[evidence table](direct_persistence_evidence.md).
The optimizer boundary is stated separately in
[`docs/direct_persistence_optimizer.md`](direct_persistence_optimizer.md).

The old v3 records do not contain complete per-step vectors. The v4 package
therefore derives only the three-resultant inner-product matrix and explicitly
marks per-step cross-Grams unavailable. New measurements must save those
statistics or return `ABSTAIN`.
