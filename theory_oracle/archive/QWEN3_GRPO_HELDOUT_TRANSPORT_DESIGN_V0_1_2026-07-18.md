# Qwen3 GRPO held-out transport design v0.1

Date: 2026-07-18
Status: pre-execution design; not yet a frozen manifest

## Purpose

This bank addresses the first, fixed-start portion of Gate P in
`ORACLE_EXTERNAL_COMPLETENESS_GATES_V0_1_2026-07-18.md`. It asks whether the
implementation-relative discrepancy decomposition and clipping-event ledger
remain interpretable on independently selected trajectories. It is not an event
hunt, a correctness experiment, an operator experiment or a long-run experiment.

## Separation from construction data

The v0.4 construction bank used arithmetic prompt blocks at offsets 320 and 384
and starts retained from earlier trajectories. The new eligible prompt blocks are
the nine disjoint 64-row blocks with offsets

`[448, 512, 576, 640, 704, 768, 832, 896, 960]`.

Before any candidate outputs are observed, Python `random.Random(20260718)`
sampling without replacement selected offsets `[768, 448, 640]`. The next three
draws from the same generator produced training seeds
`[547400606, 781084057, 1648321674]`. The realized draws are part of this design;
re-running a library-dependent sampling algorithm is not required to interpret
them.

All three runs start from the same local frozen `data/phase0_policy_final` model
but have different prompt blocks and algorithmic RNG seeds. The original
`Qwen/Qwen3-0.6B` repository snapshot is not present in the offline cache, so it
is not silently fetched or substituted. These are independent continuation runs
conditional on one shared trained starting model and runtime. They do not
establish checkpoint-, model-family- or hardware-level generality.

## Training and measured states

Each run uses the same GRPO settings as the v0.4 bank except for start model,
prompt block and seed: FP16 Accelerate training with FP32 master parameters,
SDPA, 30 optimizer steps, four generations, three policy iterations, epsilon
0.2, learning rate 1e-6 and 128 completion tokens.

The grad-enabled paired scorer is measured at pre-minibatch optimizer steps
`[2, 5, 8, 11, 14, 17, 20, 23, 26, 29]`. The fixed stage labels are:

- early: 2, 5, 8;
- middle: 11, 14, 17;
- late: 20, 23, 26, 29.

The primary state cluster is `(run, optimizer_step, rollout_batch)`. Tokens and
four completions inside one cluster are nested observations. Run is the top-level
cluster for transport uncertainty; three runs imply low precision and must not be
hidden by 15,360 token rows.

## Matched execution contract

For every measured state, eager and candidate share model parameters/buffers,
batch tensors, old log probabilities, advantages, epsilon, temperature, model
mode, autograd, AMP wrapper, attention backend and restored RNG. Candidate
compiler graph hashes, specialization sequence, runtime-invocation counts and
fallback status are recorded. Two measured repeats per implementation are kept
after the candidate warm-up.

A state is invalid if pairing, gradient/tensor-version preservation, RNG restore,
candidate invocation or execution-history evidence fails. Invalid states remain
in the denominator of the validity audit but not in discrepancy estimands.

## Frozen outputs

### Numerical discrepancy ledger

- candidate-minus-eager current-token log-probability delta;
- token-weighted and state-weighted average shift, reported separately;
- state-conditioned effect summaries by step/stage/run;
- within-state repeat variability for each implementation;
- sampling uncertainty at state and run cluster levels.

The word `bias` is not used unless an independent numerical authority is added.

### Semantic-event ledger

For every nonzero-advantage token:

- reference signed clipping margin/boundary distance;
- eager and candidate clipping decisions;
- `0->1`, `1->0`, total disagreement and directional difference;
- deterministic versus repeat-unstable disagreement;
- exposure counts in predeclared absolute reference-boundary bands
  `[0,1e-4)`, `[1e-4,1e-3)`, `[1e-3,1e-2)`, `[1e-2,inf)`.

The paired boundary-crossing diagnostic may be reported to explain events, but
because it consumes observed candidate delta it is not scored as a held-out
predictor.

## Decision rules

The bank returns separate verdicts:

1. construction validity: `VALID` only if every required identity and preservation
   gate passes, otherwise `INVALID`;
2. finite-bank semantic compatibility: `REJECT` if at least one stable or
   stochastic distributional disagreement is established, `ACCEPT` only for the
   exact declared finite bank when none is observed, otherwise `INDETERMINATE`;
3. transport stability: `SUPPORTED`, `NOT_SUPPORTED` or `INDETERMINATE`, based on
   whether direction/heterogeneity patterns are compatible across the three runs
   under run-cluster uncertainty;
4. population prevalence: normally `INDETERMINATE` with only three run clusters;
   it cannot be promoted using token-level pseudo-replication;
5. correctness: always `UNINSTANTIATED` in this design.

No event-count-dependent stopping is allowed. All three valid runs are required.
If one run fails, it is rerun only from its original frozen configuration after
the cause is documented; its prompt block or seed is not replaced.

## Preselected natural-transition capture

After drawing the three prompt blocks and three training seeds, the next draw
from the same `random.Random(20260718)` stream selected optimizer step 29 from
the ten measured steps. Run B captures this state irrespective of its discrepancy
or event outcome. The choice is therefore not the known step-14 witness from the
construction bank.

The capture is evidence preparation for Gate T, not a Gate-T result. It contains
the exact step-29 minibatch, model, optimizer, scheduler, AMP scaler, Trainer and
Python/NumPy/Torch RNG states. It also retains the exact measured input packs at
steps 2, 5, 8, 11, 14, 17, 20, 23, 26 and 29 so a later candidate arm can rebuild
the observed compiler specialization history. Capture must preserve Torch RNG,
model gradients and parameter/buffer tensor versions exactly. Failure of any
preservation check invalidates the snapshot but does not silently remove run B
from the transport validity audit.

The snapshot does not by itself prove replay identity, a natural transition
effect or operator causality. Those require a separately frozen Gate-T executor
and independent replay audit.

## Kill and downgrade rules

- If the candidate path or compiler history differs from the declared pair, the
  affected run is invalid, not a zero-effect run.
- If state means vary but only a token-pooled mean is reported, the discrepancy
  analysis is incomplete.
- If repeats vary but are folded into implementation shift, the shift claim is
  invalid.
- If stable direction disappears across runs/stages without an explainable
  condition, the result supports local heterogeneity, not a global shift.
- If events occur only in a tiny boundary-exposure stratum, this is reported as
  boundary conditioning; it is not evidence of a universal compiler tendency.
- If no event occurs, absence is limited to the sampled exposure profile and
  sensitivity. It is not eager/compiled equivalence.

## Resource protocol

Runs execute sequentially in fresh processes. Before each run, host-visible
`nvidia-smi` must return within ten seconds and show adequate free memory and no
unowned compute process. Each run keeps one training model and one lazily compiled
wrapper in one process. After exit, GPU processes are checked again. No old result
directory is overwritten.

Large checkpoints are retained until validity hashes and compact evidence records
exist. Deletion is allowed only for reproducible duplicates not referenced by the
manifest or result cards.
