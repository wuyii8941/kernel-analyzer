# Preliminary Survey: A Multi-Endpoint Discrepancy Oracle

## 1. Interim conclusion

The project should not search for one universal scalar endpoint. A defensible Oracle is a **scoped, multi-endpoint relational decision procedure**:

1. specify the implementation relation, target state population, observable geometry, configuration, and randomness/coupling protocol;
2. calibrate that the intended execution paths and matched state were actually realized;
3. estimate numerical, semantic, and transition-level discrepancies with their uncertainty;
4. apply predeclared endpoint-specific acceptance relations;
5. issue a correctness verdict only when an independent specification or truth reference exists.

The measurement profile can exist without a global pass/fail verdict. If an operational binary Oracle is required, it should be a transparent policy over endpoint-specific verdicts, not an unexplained weighted sum.

This direction is grounded in several mature literatures. The possible research contribution is their relational composition for matched training states, not invention of bias/variance analysis, differential testing, distribution testing, or causal debugging.

## 2. What “Oracle” means here

The software-testing oracle problem is the problem of distinguishing desired/correct behavior from potentially incorrect behavior; merely detecting a difference is not yet a correctness Oracle ([Barr et al., *The Oracle Problem in Software Testing: A Survey*](https://discovery.ucl.ac.uk/id/eprint/1471263/)). Classic differential testing treats disagreement among comparable systems as a **candidate** bug-exposing test, not proof that one side is wrong ([McKeeman 1998](https://www.cs.swarthmore.edu/~bylvisa1/cs97/f13/Papers/DifferentialTestingForSoftware.pdf)).

This distinction is especially important for floating-point compilers. NNSmith explicitly notes that correctly compiled models can be close but not identical and does not regard every such difference as a bug ([NNSmith, ASPLOS 2023](https://lingming.cs.illinois.edu/publications/asplos2023.pdf)). Metamorphic testing similarly supplies necessary relations when exact outputs are unavailable; it alleviates rather than magically eliminates the oracle problem ([Segura et al., TSE survey](https://eprints.whiterose.ac.uk/id/eprint/110335/)).

Therefore the project needs a claim ladder:

| Level | Supported statement | Required evidence |
| --- | --- | --- |
| M0 | measurement calibrated | matched state, execution identity, self-pair controls |
| M1 | numerical implementation discrepancy | calibrated paired observations |
| M2 | semantic/event distribution discrepancy | defined event map and randomness/coupling protocol |
| M3 | update/transition impact | matched one-step gradient/update/next-state evidence |
| M4 | operational non-equivalence | predeclared acceptance bounds or application loss |
| M5 | correctness violation | independent specification, high-precision truth, invariant, or confirmed wrong-code evidence |

M1--M3 do not imply M5.

## 3. Core comparison contract

For each study instance define a contract

```text
C = (R, C, Q, S, Y, G, Xi, Pi, Cfg, A)
```

where:

- `R`, `C`: fixed implementation paths;
- `Q`: target distribution over matched states;
- `S`: state definition and sampling unit;
- `Y`: observable hierarchy;
- `G`: geometry or loss used to compare each observable;
- `Xi`: allowed randomness;
- `Pi`: coupling protocol between implementations;
- `Cfg`: configuration and time hierarchy;
- `A`: endpoint-specific acceptance relations.

An additional empirical invariant is mandatory:

> Every recorded observation must prove that the claimed implementation path executed.

A string label such as `compiled` is not proof. Fallback, graph breaks, cache limits, library dispatch, or configuration leakage can change the treatment actually received. This is construct validity: without execution identity, the implementation contrast is not identified.

## 4. Measurement model: what is borrowed and what is changed

For observable `Y`, define the paired implementation discrepancy

```text
D(s, xi) = Y_C(s, xi_C) - Y_R(s, xi_R).
```

Without independent truth this is not mathematical error. With a fixed state population `Q` and randomness protocol:

```text
m(s) = E[D | S=s]
mu_Q = E_Q[m(S)]
h(s) = m(s) - mu_Q
eta = D - m(s).
```

The terms are:

- `mu_Q`: average implementation shift relative to the selected baseline and `Q`;
- `h(s)`: state-conditioned effect heterogeneity;
- `eta`: within-state runtime fluctuation under the specified protocol;
- estimator uncertainty: uncertainty caused by finite trajectories, states, and repeats, reported separately.

The metrology analogy is useful but bounded. VIM defines instrumental bias relative to a reference quantity value ([JCGM/VIM 4.20](https://jcgm.bipm.org/vim/en/4.20.html)); therefore `mu_Q` should not be called truth bias. NIST Gauge R&R supports repeated parts/operators/repeats as an experimental design for separating variation sources ([NIST Gauge R&R](https://www.itl.nist.gov/div898/handbook/mpc/section4/mpc4.htm)), but eager and compiled are normally fixed treatments, not a random sample of operators.

Bland--Altman analysis supports paired method differences and magnitude-dependent agreement, including repeated measurements ([Bland and Altman 1999](https://journals.sagepub.com/doi/10.1177/096228029900800204)). Basic limits of agreement rely on restrictive assumptions and should not be applied automatically to hierarchical, heteroscedastic training data ([Taffé 2021](https://doi.org/10.1016/j.jclinepi.2021.04.004)).

### Consequence for the plan

Gauge R&R is best treated as:

- an experimental-design analogy;
- a source of variance-component vocabulary;
- a warning to preserve repeats and crossed conditions.

It is not the complete probabilistic model, estimator, acceptance rule, or correctness theory.

## 5. Numerical endpoint family

Numerical analysis supplies the distinction among forward discrepancy, backward error, problem conditioning, and algorithmic stability ([Higham, *Accuracy and Stability of Numerical Algorithms*](https://epubs.siam.org/doi/abs/10.1137/1.9780898718027.fm)). A large output difference can reflect an ill-conditioned problem; a small output difference does not prove the implementation is correct.

Minimum numerical endpoints should include, as applicable:

- signed and absolute discrepancy;
- scale-normalized discrepancy;
- vector norm and angular discrepancy;
- predeclared task-relevant projection;
- quantiles and tail exceedance;
- conditional mean and within-state variability;
- covariance induced by the pairing protocol.

Tools such as Verificarlo use Monte Carlo Arithmetic to study numerical sensitivity and compiler-optimization influence ([Verificarlo](https://arxiv.org/abs/1509.01347)). Such injected perturbation distributions are sensitivity probes, not measurements of real GPU runtime nondeterminism. FLiT studies compiler-induced floating-point variability and controlled localization ([FLiT](https://arxiv.org/abs/1811.05618)); Herbgrind and pLiner demonstrate that observed error, error source, and propagation/localization point may differ ([Herbgrind](https://arxiv.org/abs/1705.10416), [pLiner](https://web.cs.ucdavis.edu/~rubio/includes/sc20.pdf)).

These projects justify source, propagation, and exposure as separate concepts. They do not justify labeling reduction as bias or ordinary floating-point rounding as variance before observation.

## 6. Semantic endpoint family

Let `E_I(s, xi)` be an event in a finite or structured event space.

For a binary event, two distinct estimands are mandatory:

```text
directional shift = P(E_C=1) - P(E_R=1)
                  = P(E_R=0,E_C=1) - P(E_R=1,E_C=0)

paired disagreement = P(E_R != E_C)
                    = P(E_R=0,E_C=1) + P(E_R=1,E_C=0).
```

Directional shift is a difference of marginal event rates. Paired disagreement depends on the chosen coupling and measures reproducibility under that coupling. One cannot replace the other.

For multiclass/ranking/routing events, retain at least:

- marginal distribution distance over outcomes;
- coupled disagreement under a declared coupling;
- application-specific cost between outcomes when labels have geometry;
- set overlap or assignment cost for top-k/MoE outputs.

For a finite event space, a natural marginal estimand is total variation:

```text
TV(P_R, P_C) = 1/2 * sum_e |P_R(e) - P_C(e)|.
```

For any coupling of the two implementations,

```text
P(E_R != E_C) >= TV(P_R, P_C).
```

Equality is attainable under a maximal coupling, but a shared-RNG implementation is not automatically maximal. Marginal event drift and coupled reproducibility therefore have a precise relationship while remaining different objects.

For sampling, common random numbers are a paired experimental design that can reduce uncertainty in differences between stochastic systems ([Yang and Nelson 1991](https://doi.org/10.1287/opre.39.4.583)). A common-RNG token disagreement is coupling-specific and does not by itself prove that the two marginal token distributions differ. Both the marginal law and the coupled disagreement must be reported.

## 7. Boundary conditioning and branch distance

Traditional search-based software testing uses branch distance as a nonnegative fitness estimating how close execution was to satisfying a target predicate; its role is guiding test-data generation ([McMinn 2011](https://philmcminn.com/publications/mcminn2011.pdf), tracing back to [Korel 1990](https://doi.org/10.1109/32.57624)).

Our object differs in three ways:

1. it is relational: two implementations are observed on the same state;
2. the signed event margin retains direction, while standard branch-distance fitness is normally nonnegative and target-branch-specific;
3. the estimand is a distributional implementation effect conditional on boundary geometry, not merely a search fitness.

Therefore “decision margin is branch distance” is too strong. The defensible relation is:

> Branch-distance ideas motivate boundary proximity, but the proposed Oracle studies boundary-conditioned relational discrepancy and event-law change.

## 8. Transition endpoint family

For training state `S`, define an implementation-specific one-step kernel

```text
K_I(. | s) = Law(S' | S=s, implementation=I).
```

The transition endpoint should be a profile rather than the impossible demand to estimate an arbitrary high-dimensional kernel from a few repeats:

- mean update difference;
- update norm and angle;
- predeclared loss/task projection;
- optimizer-state event or discrepancy;
- covariance/noise difference when repeated stochastic transitions exist;
- tail exceedance for application-relevant transition loss;
- optional multivariate distribution distance when sample size supports it.

Energy distance and MMD are legitimate multivariate two-sample distances/tests ([energy distance review](https://doi.org/10.1002/wics.1375), [Gretton et al. 2012](https://www.jmlr.org/beta/papers/v13/gretton12a.html)). They answer whether distributions differ under selected geometry/kernel; they do not specify practical importance, equivalence, or correctness. Kernel/metric choice, sample size, and hierarchical dependence remain part of the contract.

Long-term extrapolation requires additional dynamics assumptions. Markov-kernel perturbation theory can bound long-run differences from one-step differences only under stability/ergodicity and metric conditions ([Rudolf and Schweizer 2018](https://arxiv.org/abs/1503.04123)). Those assumptions are not currently established for deep-network training. Long-run training therefore remains validation or a separate dynamics question.

## 9. Difference testing is not equivalence testing

Three conclusions must remain distinct:

- evidence of difference;
- no detected difference at current resolution;
- evidence that the difference lies inside a prespecified acceptable region.

The third requires equivalence/non-inferiority logic. For simple paired means, two one-sided tests (TOST) are a standard option once a scientifically justified equivalence margin is fixed ([paired TOST overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC5382845/), [paired equivalence test study](https://doi.org/10.1080/03610918.2011.626545)). TOST is not automatically appropriate for heavy-tailed, hierarchical, multivariate, or rare-event endpoints; the key principle is predeclared practical bounds, not a specific t-test.

Endpoint-specific verdicts should use confidence sets:

| Confidence set relative to acceptable region | Verdict |
| --- | --- |
| fully inside | equivalent within declared tolerance |
| fully outside in a harmful direction | operational drift/non-equivalence |
| overlaps the boundary | indeterminate at current resolution |

Failure to reject a difference null is never enough to claim equivalence.

## 10. Proposed multi-endpoint Oracle profile

For contract `C`, return:

```text
Profile(C) = {
  calibration,
  numerical,
  stochastic/heterogeneity,
  semantic,
  transition,
  attribution,
  specification status,
  uncertainty,
  endpoint verdicts,
  claim level
}
```

### 10.1 Calibration

- matched-state integrity;
- execution-path identity;
- self-pair floors;
- pairing/coupling validity;
- missing/NaN/crash accounting.

Calibration failure makes downstream endpoint verdicts `invalid`, not `equivalent`.

### 10.2 Numerical

- `mu_Q`, conditional structure, heterogeneity;
- marginal and paired runtime variability;
- tails and geometry-specific summaries;
- sampling uncertainty.

### 10.3 Semantic

- marginal event-law shift;
- directional shift where a direction exists;
- coupled disagreement;
- boundary-conditioned and consequence-weighted results.

### 10.4 Transition

- gradient/update/optimizer/next-state profile;
- deterministic and stochastic components;
- task-relevant projections and tail loss.

### 10.5 Correctness

- unavailable without specification;
- calibrated against high-precision/invariant/confirmed-bug cases when available.

### 10.6 Global operational flag

If a single test-run flag is required:

```text
flag = any(endpoint confidence set violates its predeclared acceptance region)
```

The report must retain the endpoint that triggered the flag. No weighted aggregate should be introduced until an application supplies a defensible loss function.

## 11. Attribution after the Oracle is fixed

Repair and injection are interventions whose outcomes are the already-defined endpoint profile. Related causal-debugging work uses fault injection/interventions to go beyond correlation ([AID](https://www.microsoft.com/en-us/research/publication/causality-guided-adaptive-interventional-debugging/)); delta debugging isolates cause-effect chains by systematically altering program states ([Zeller 2002](https://www.st.cs.uni-saarland.de/papers/fse2002/)).

For this project:

- repair estimates effect removal under a compiled-context intervention;
- injection estimates effect creation under a reference-context intervention;
- disagreement between them is expected under interaction/nonlinearity;
- changing fusion, layout, dispatch, or downstream compilation invalidates a simple operator causal interpretation;
- without intervention integrity, report intervention-dependent attribution only.

## 12. SE evidence requirements

The ACM SIGSOFT empirical standards emphasize construct, internal, conclusion, and external validity, reliability, and reproducibility ([SIGSOFT Empirical Standards](https://www2.sigsoft.org/EmpiricalStandards/)). The benchmarking standard specifically requires realistic workload justification, sufficient repetition, raw-result preservation, and transparent reporting of execution problems ([detailed standards](https://www2.sigsoft.org/EmpiricalStandards/docs/standards)).

Minimum requirements for this project are therefore:

- explicit construct mapping from observable to intended Oracle claim;
- execution-path and matched-state validity;
- independent trajectory/state sampling units;
- effect sizes and uncertainty, not token-level pseudo-replication;
- discovery/confirmation separation and selection accounting;
- predeclared endpoint geometry and acceptance bounds;
- baseline comparison against simple discrepancy metrics;
- complete missing/fallback/NaN/crash accounting;
- environment/configuration provenance and raw artifacts;
- threat-to-validity table for every claim level.

## 13. What is established versus still open

### 13.1 Traceability to the current plan

| Plan stage | Main external foundation | What is borrowed | What remains project-specific |
| --- | --- | --- | --- |
| P0 comparison contract | Oracle survey, differential/metamorphic testing, VIM, Higham | oracle/specification boundary, reference-value discipline, numerical geometry | the matched training-state relational contract |
| proposed P0.5 execution identity | SIGSOFT construct validity, compiler differential testing | prove the treatment/system actually executed | backend/code-path canaries for dynamic DL compilation |
| P1 calibration | NIST Gauge R&R, repeated-measure agreement, common random numbers, SE benchmarking | crossed repeats, paired dependence, raw-data and stability requirements | complete DL training-state restoration and RNG partition |
| P2 discrepancy characterization | variance decomposition, Bland--Altman diagnostics, energy/MMD where applicable | conditional summaries, agreement/tail/distribution diagnostics | state-conditioned implementation-discrepancy atlas |
| P3 source interventions | Verificarlo, FLiT | controlled numerical perturbation and compiler-variability analysis | mapping source classes to shift/heterogeneity/runtime/transition endpoints |
| P4 conditioning | Higham | forward/backward error, conditioning, stability | applying those distinctions to compiled training steps |
| P5 propagation | Herbgrind, pLiner, PIE/RIPR intuition | source/propagation/exposure separation | propagation across model, gradient, optimizer, and semantic layers |
| P6 transition | distribution distances, transition-kernel perturbation theory | compare stochastic laws; conditions needed for long-run bounds | task-aware one-step training transition profile |
| P7 validation | NNSmith, metamorphic testing, confirmed bug benchmarks | baseline differential testing and partial specifications | incremental validity over raw delta for training impact |
| P8 verdict | equivalence/non-inferiority testing | prespecified acceptable region and indeterminate verdict | endpoint-specific operational losses and thresholds |
| P9 attribution | AID, causal testing, delta debugging | intervention rather than correlation | repair/injection integrity under fusion/layout/codegen interactions |

This table shows that most ingredients are established. The research risk is whether their combination yields a coherent, useful estimand and whether it adds information beyond simpler discrepancy checks.

### Established enough to proceed

- implementation-relative discrepancy is distinct from truth-relative error;
- average shift, state heterogeneity, runtime variability, and sampling uncertainty are different objects;
- semantic marginal shift and paired disagreement are different estimands;
- multiple endpoints are legitimate and should not be collapsed prematurely;
- transition impact is downstream evidence, not a replacement for numerical/semantic measurements;
- correctness requires independent specification;
- operator attribution requires intervention integrity.

### Open choices

- exact target state populations and transport/sensitivity policy;
- endpoint-specific geometries for gradient/update/optimizer state;
- practical equivalence margins and their source;
- which semantic structures are mandatory for generality validation;
- repeat/state allocation under deterministic and nondeterministic regimes;
- whether a global operational flag is required at all;
- minimum intervention unit that preserves compiler context.

## 14. Preliminary novelty assessment

| Candidate contribution | Preliminary strength |
| --- | --- |
| matched-state mean/variance decomposition | mature statistics applied to a new subject |
| use of differential testing for DL compilers | already established |
| boundary distance for finding sensitive tests | already established in SBST, though not identical |
| numerical-to-semantic-to-transition endpoint ladder | potentially useful abstraction if shown nonredundant |
| explicit separation of marginal event shift and coupling-dependent disagreement | potentially meaningful estimand discipline |
| execution-identity-calibrated relational measurement | strong methodological requirement, likely not alone a novelty claim |
| repair/injection over an endpoint profile | potentially new analysis problem if intervention integrity and interactions are handled |
| one universal scalar score | currently unjustified and should not be claimed |

The strongest defensible direction is not “Gauge R&R for compilers.” It is a calibrated relational Oracle profile connecting numerical discrepancy laws, semantic event laws, and one-step transition impact, with explicit claim and attribution boundaries.
