# Oracle external-completeness gates v0.1

Date: 2026-07-18

This document freezes the next empirical gates. Its purpose is not to accumulate
more clipping examples. Each gate answers a distinct unresolved claim in
`DISCREPANCY_ORACLE_THEORY_V1_2026-07-18.md`.

## 1. What is already established, and what is not

Established for the declared Qwen v0.4 finite bank:

- the paired eager/compiled construction is valid under a recorded compiler
  history;
- average implementation-relative shift, state-conditioned heterogeneity and
  within-state repeat variability are separately representable;
- two deterministic clipping disagreements exist;
- boundary-conditioned ranking is not identical to raw absolute numerical
  delta on that same frozen bank;
- a selected branch repair changes the one-step parameter update.

Not established:

- prevalence or stability over an independently selected state population;
- stability and prevalence over independently selected held-out states;
- a natural optimizer/scaler transition effect;
- correctness relative to an independent authority;
- operator-level causal effect;
- long-run training consequences.

The observed v0.4 bank is construction data. It must not also serve as external
validation data.

## 2. Gate P: held-out, multi-stage semantic validation

### Question

Does the frozen Oracle distinguish numerical magnitude from semantic risk on
new matched states sampled across training stages, rather than only on the bank
where its motivating events were found?

### Target population and sampling unit

- The operational population must be declared before execution: a finite set
  of eligible `(training run, pre-step checkpoint, rollout/prompt draw)` units.
- The primary independent cluster is the pre-step training state. Tokens from
  one state are not independent samples.
- At least early, middle and late training strata must be represented. Multiple
  snapshots from one trajectory improve stage coverage but do not establish
  cross-run generality.
- State selection and prompt/rollout selection must use a recorded rule that is
  independent of eager/compiled discrepancy. Event-enriched selection may be a
  separate stress-test stratum, never the prevalence stratum.

### Frozen endpoints

For every eligible nonzero-advantage token, record:

1. signed implementation-relative log-probability shift;
2. reference boundary distance and direction;
3. eager/compiled clipping event;
4. directional semantic shift and total disagreement;
5. within-state repeat variability under the frozen randomness protocol;
6. raw `abs(delta)` and the paired boundary-conditioned crossing diagnostic.

The paired boundary diagnostic is **post-execution**: it uses the candidate delta
observed on that state. For a scalar threshold event it algebraically reconstructs
crossing geometry. Its ranking is therefore construct evidence, not held-out
prediction. A predictive risk score is a separate optional claim and must not use
the held-out candidate delta or event label.

Report estimates by state cluster and training-stage stratum. A token-level
confidence interval that treats tokens as independent is invalid.

### Pass conditions

Gate P passes only if all of the following hold:

- construction validity and compiler/execution history are recorded for both
  implementations;
- the direction, heterogeneity and repeatability ledgers are estimable, even if
  an estimated component is zero;
- held-out results establish either (a) semantic disagreements, or (b) a
  calibrated upper bound under a declared sensitivity region;
- state-cluster uncertainty and stage/run heterogeneity are reported rather than
  token-pooled pseudo-replication;
- conclusions are explicitly limited to the operational population.

### Kill or downgrade conditions

- If the paired event ledger is always recoverable from raw delta alone without
  reference boundary or direction, the proposed boundary layer is empirically
  redundant for that subject.
- If effects reverse arbitrarily across independently selected runs or stages
  and no recorded state feature explains this, only local case-study claims
  remain.
- If disagreements occur only after selecting states using the discrepancy
  itself, prevalence and predictive claims are invalid.
- If repeats cannot distinguish implementation effect from runtime variation,
  stable-shift claims are invalid.

### Optional predictive-risk claim

If later work claims that the Oracle can prioritize unexecuted states or
operators, its score must be frozen using construction data and may use reference
features plus implementation effects estimated on other states. It may not use
the held-out candidate delta. Only this prospective score is evaluated by
held-out ranking, calibration or decision-curve metrics. Failure of that score
does not invalidate the post-execution semantic Oracle; it kills the predictive
extension.

## 3. Gate T: natural one-step transition validation

### Question

When the full pre-step training state is restored, do eager and compiled induce
different natural next-state distributions, beyond a manually repaired branch?

### Required frozen state

The state must include model parameters, optimizer moments and counters,
scheduler, scaler/overflow state, batch/tokens, old/reference log-probabilities,
RNG states and compiler/execution history. A model-only snapshot is insufficient.

### Endpoints

- optimizer step executed/skipped and scaler transition;
- gradient vector or declared summaries before clipping;
- clipping/overflow/routing decisions involved in the step;
- parameter update under scale-aware distances and at least one directional
  comparison;
- paired repeat distribution if any randomness remains open.

The endpoint is the transition under the declared implementation pair, not
long-run convergence.

### Pass conditions

- replay identity is demonstrated within each implementation before comparing
  implementations;
- all non-implementation state is paired or its coupling is explicitly open;
- event, update and numerical ledgers are reported separately;
- a null result is reported as a sensitivity-bounded result, not as equality.

### Kill or downgrade conditions

- Failure to restore optimizer/scaler/RNG state makes this a synthetic update,
  not a natural transition Oracle.
- If compiler history differs between arms, implementation attribution is
  confounded.
- A parameter distance without an acceptance authority is impact evidence, not
  correctness evidence.

## 4. Gate A: attribution comes after P and T

Repair and injection are not substitutes for Gates P or T. They become useful
only after the endpoint is externally stable enough to attribute.

Gate A requires:

- repair and injection for the same candidate unit;
- explicit treatment of alternative causes and pairwise/higher-order
  interactions;
- preservation of surrounding graph, layout, fusion and dispatch choices, or a
  downgrade to intervention-dependent attribution;
- separate labels for discrepancy generation, propagation and boundary
  conversion.

The existing v0.9 repair establishes an intervention-dependent
boundary-conversion effect. It does not establish an operator root cause.

## 5. Gate C: correctness is a separate authority

No amount of eager/compiled pairing creates ground truth. Correctness requires
at least one declared external authority, such as a specification, sufficiently
accurate reference, formal proof obligation or confirmed wrong-code instance.
Without it, all passed gates support discrepancy and semantic-impact claims only.

## 6. Resource and evidence policy

- Run at most one model-bearing arm per GPU process unless memory headroom has
  been measured.
- Release model, optimizer, compiled wrappers and accelerator references between
  arms; then collect Python and CUDA caches. A fresh process is preferred for
  independent arms.
- Check active GPU processes before and after each run.
- Keep compact manifests, hashes and summary records. Large artifacts may be
  deleted only when reproducible and not referenced by a retained result.
- Invalid runs stay labelled invalid. Their artifacts are not silently reused
  as evidence.

## 7. Order of execution

1. Build Gate P using independently selected states and the frozen v0.4 score.
2. Capture a new full pre-step state specifically for Gate T; do not pretend
   existing model-only snapshots contain optimizer state.
3. Attempt Gate A only for an endpoint that survives P and affects T.
4. Add Gate C only when an independent numerical or semantic authority exists.

This order is logical, not chronological dogma: an early feasibility check may
be used, but it cannot be counted as evidence for a later gate.
