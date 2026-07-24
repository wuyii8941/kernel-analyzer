# Oracle Contract Validation Cases v0.1 — 2026-07-16

> These are theory-level unit tests for the Oracle definition. “Pass” means the explicit contract/verdict rules force the expected result; it is not empirical evidence that a compiler implementation behaves correctly.

## 1. Verdict mechanics

| Case | Expected verdict | v0.1 result | Audit |
|---|---|---|---|
| no semantic envelope or compatibility margin | `UNINSTANTIATED` | missing contract fields fail closed | pass |
| intended candidate silently falls back to reference | `INVALID` | candidate-realization gate precedes behavior | pass |
| joint confidence set wholly inside acceptable set | `ACCEPT` within declared scope | explicit containment rule | pass |
| joint confidence set disjoint from acceptable set | `REJECT` contract | explicit disjointness rule | pass |
| confidence set overlaps boundary | `INDETERMINATE` | neither acceptance nor rejection is asserted | pass |
| no significant difference but interval is wide | `INDETERMINATE` | difference-test nonsignificance is not acceptance | pass |
| tiny statistically significant difference inside tolerance | `ACCEPT` | practical contract, not equality p-value, governs | pass |

## 2. Error versus discrepancy

| Case | Expected interpretation | v0.1 result | Audit |
|---|---|---|---|
| eager and compiled share the same wrong value | correctness reject if truth/spec catches it; relative discrepancy zero | conformance and discrepancy are separate | pass |
| eager wrong, compiled correct | compiled correctness accept; nonzero relative discrepancy | eager is baseline only | pass |
| both correct but choose different allowed floating results | correctness accept; compatibility may differ | semantic envelope may be non-singleton | pass |
| candidate is closer to truth but differs more from eager | correctness/non-degradation can favor candidate | truth-relative error is primary when available | pass |
| no independent truth or margin | measurement only | `UNINSTANTIATED`, not correctness | pass |

## 3. Bias, heterogeneity and runtime variability

| Case | Expected interpretation | v0.1 result | Audit |
|---|---|---|---|
| fixed nonzero discrepancy repeats identically | deterministic shift, runtime variability zero | repeat axis is separated | pass |
| two equally weighted strata have opposite shifts | global bias zero, heterogeneity nonzero | conditional/tail constraints prevent cancellation | pass |
| rare stratum violates universal spec | universal correctness reject | witness is not averaged away | pass |
| stochastic perturbation has zero mean but heavy violation tails | mean bias zero, stochastic/tail risk nonzero | tail/runtime constraints are conjunctive | pass |
| finite sample produces uncertain population mean | sampling uncertainty, not behavior variance | carried by confidence set | pass |
| autotuner choice is frozen after selection | configuration-specific deterministic mapping | configuration is contract identity | pass |
| autotuner is redrawn every repeat by protocol | higher-level runtime/configuration distribution | must be explicitly sampled | pass |

## 4. Structured operator semantics

| Case | Expected verdict | v0.1 result | Audit |
|---|---|---|---|
| `argmax` on identical finite operand returns nonmaximum index | correctness reject | P1 singleton relation | pass |
| `argmax` tie returns index other than documented first maximum | correctness reject in PyTorch P1 | documented tie relation is explicit | pass |
| `topk` returns different tied indices, both valid | correctness accept for both | P2 envelope is set-valued | pass |
| `topk` chooses value strictly outside valid boundary | correctness reject | outside P2 set relation | pass |
| eager/compiled argmax differ because upstream logits differ | argmax operator not blamed; upstream semantic impact reported | matched-operand gate separates source from exposure | pass |
| value-preserving structural operator changes integer/index field | exact-core reject | C1 finite relation | pass |

## 5. Numerical geometry

| Case | Expected interpretation | v0.1 result | Audit |
|---|---|---|---|
| reduction exact sum is near zero after cancellation | use `sum |x_i|`/conditioned bound, not relative-to-result error | reduction catalog specifies conditioning scale | pass |
| reduction output is inside conservative analytical bound but differs from eager | numerical conformance accept; compatibility separately judged | C3 envelope and C6 are separate | pass |
| matmul error large in absolute terms but within scale/conditioning bound | may conform | coordinate-conditioned envelope | pass |
| small raw matmul delta violates a strict residual/accuracy contract | reject | semantic geometry outranks raw magnitude | pass |
| softmax outputs sum to one but probability vector is wrong | invariant alone cannot accept | catalog declares invariant necessary, not sufficient | pass |
| all logits receive same shift and only softmax probabilities are exposed | probability contract may accept despite large raw-logit delta | geometry follows exposed semantics | pass |
| eigendecomposition flips valid eigenvector signs | coordinate mismatch does not reject | residual/subspace geometry | pass |
| ill-conditioned solve has small residual and large forward difference | report backward error and conditioning; no naive coordinate verdict | linalg contract separates them | pass |
| gradient norm matches but direction is wrong | gradient contract rejects under vector/directional geometry | norm alone is insufficient | pass |
| finite-difference reference is noisy at chosen step | reference invalid/indeterminate | reference error must be below resolution | pass |

## 6. Stochastic semantics

| Case | Expected interpretation | v0.1 result | Audit |
|---|---|---|---|
| same categorical law, different same-seed token under unspecified algorithm | no correctness rejection | P3 law and replay contracts are separate | pass |
| different law, finite sample happens to select same tokens | no automatic acceptance | distribution confidence governs | pass |
| exact target law with finite samples and zero tolerance | normally indeterminate for acceptance | C4/P3 state this limitation | pass |
| without-replacement sampler emits duplicate where prohibited | exact-core reject | structural/law support relation | pass |
| stochastic rounding mean is unbiased but tails exceed spec | reject tail contract | zero mean does not erase violations | pass |

## 7. Transition and impact

| Case | Expected interpretation | v0.1 result | Audit |
|---|---|---|---|
| optimizer parameter norm matches but momentum buffer is wrong | transition reject | complete declared next state is conjunctive | pass |
| AMP skip flag differs while final parameter happens not to change | exact transition/control violation or impact depending spec | discrete state is preserved | pass |
| one-step transition differs but long-run accuracy matches | local transition reject/impact without long-run harm claim | long-run is separate | pass |
| one-step transition passes but long-run runs diverge due unrelated randomness | no compiler long-run claim inferred | transition scope remains local | pass |
| operator repair removes drift but changes fusion/layout | operator impact `NOT_IDENTIFIABLE` | intervention integrity gate | pass |
| repair succeeds, injection fails due nonlinear context | intervention-dependent asymmetry, not contradiction | necessity/sufficiency contexts separated | pass |

## 8. Population and generalization

| Case | Expected interpretation | v0.1 result | Audit |
|---|---|---|---|
| stress inputs show many violations | robustness evidence, not deployment prevalence | nominal and stress populations separate | pass |
| reference population passes, candidate-reached population fails | two scoped verdicts, no averaging without weights | `Q_o` is contract identity | pass |
| common signatures pass, one admitted signature has exact witness | universal family reject | family quantifier preserves witness | pass |
| sampled signatures pass but domain is unbounded | no universal acceptance | coverage scope stays explicit | pass |
| operator version/configuration changes | prior verdict does not silently transfer | contract version/configuration identity changes | pass |

## 9. Remaining empirical validation

The definition passes these logical cases. The next validation must establish that concrete evidence pipelines preserve the required matched operand, candidate realization, configuration and semantic geometry. It must also compare Oracle rankings with raw numerical delta on independent seeded/confirmed semantic violations. Until then the Oracle is logically coherent but not empirically validated as a superior detector.
