# Frozen localization-method plan v1.1

## Goal

Given a frozen semantic-contract violation, automatically reduce the search
space to the **narrowest supported compiler stage and replayable region**.
An operation or generated kernel is reported only when provenance and
controlled evidence support that extra resolution.  The system does not rank
operators by raw delta and does not promise a unique root cause.  Here
"minimal" means **minimum sufficient complexity**: a calibration case may be
small only if it retains the property being claimed.  For a training claim,
that includes matched state, backward, update, multiple candidate regions,
and a declared semantic endpoint.  A one-kernel reproducer is plumbing
calibration, not training-localization evidence.

## Evidence chain

```text
semantic-contract violation
  → stage screening
  → symptom-preserving graph/region reduction
  → provenance capture
  → same-input local replay (production)
  → fixed-suffix boundary substitution (mediation)
  → controlled intervention + no-op/context gates
  → certificate and stopping decision
```

Production and mediation are independent tests:

| same-input local output differs | boundary substitution changes endpoint | permitted interpretation |
|---|---|---|
| no | no | no current source or endpoint evidence |
| yes | no | local numerical producer without observed endpoint effect |
| no | yes | an upstream discrepancy reaches this boundary; this region is not a proven source |
| yes | yes | source-and-mediation candidate; still not a unique root cause |

## Why these checks exist

The design takes two reusable lessons from numerical debugging rather than
inventing a new attribution rule.  Herbgrind distinguishes local error sources
from their dynamic influence on a later output; pLiner narrows a numerical
failure by hierarchical, symptom-preserving interventions.  Our additions are
compiler-stage screening, cross-level provenance, and training-step endpoints.

| prior work | reusable mechanism | local invariant | not implied here |
|---|---|---|---|
| [Herbgrind (PLDI 2018)](https://www.cs.cornell.edu/~lerner/papers/herbgrind-pldi2018.pdf) | distinguish error production from output influence | same-input replay is separate from fixed-suffix mediation | unique compiler cause |
| [pLiner (SC 2020)](https://web.cs.ucdavis.edu/~rubio/includes/sc20.pdf) | hierarchical search using a failure predicate | each reduction preserves the declared contract violation | an arbitrary repair is local |
| [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html) | backend ablation (`eager`, `aot_eager`, `inductor`) | first observed failing stage is recorded with versions/config | source-line or kernel cause |

## Execution order and acceptance gates

### Phase 0 — evidence cleanup (complete)

Bind every retained case to artifacts, environment and allowed claim.  Correct
overclaims, but retain raw historical records.  Acceptance: this ledger is
the only current claim authority.

### Phase 1 — minimum-sufficient calibration pair (complete)

1. **Kernel plumbing microcase.** A hidden seeded generated-kernel mutation
   with an opaque case manifest.  It validates capture, provenance, same-input
   replay, no-op, direct intervention and context invariance.  It is a unit
   test, not a localization-accuracy result.
2. **Tiny training-step microcase.** A frozen `forward → loss → backward →
   clipping → optimizer update` program with multiple natural dataflow regions
   and a hidden deterministic discrepancy.  It validates stage screening,
   production/mediation separation, and endpoint preservation through update
   and optimizer state.

Acceptance for both: the instrumented no-op matches the original execution in
forward, backward, clipping, parameter update and optimizer state; the seeded
fault is found without a case-specific op/model/mutation rule; unrelated
regions are not licensed as sources.

The two cases answer different questions.  The kernel microcase only validates
capture/replay/intervention plumbing.  The training microcase is the smallest
complete localization loop: it must automatically reduce a one-step training
symptom across several natural regions while preserving the
forward/backward/update contract.  Neither is external localization-accuracy
evidence.

### Phase 2 — freeze generic locator (complete)

Freeze the stage matrix, contract comparator, region reducer, provenance
schema, replay/intervention protocol and claim gates.  Remove all Qwen,
`mm`, architecture, and mutation-specific branches from the locator core.
Acceptance: rerunning Phase 1 needs only a case manifest and declared
contract; the emitted certificate records every automatic and manual decision.

### Phase 3 — hidden historical external validation (active)

Use two withheld, independently patched cases: one whose correct stopping
level is frontend/AOT/lowering, and one whose evidence should reach
Inductor/generated kernel.  Patch/issue/fixed IR are unavailable to the
locator until the certificate hash is frozen.  Score stage coverage, candidate
coverage, search-space reduction, false localization, and correct stopping.

The case package given to the locator contains only the failing revision,
reproducer, declared contract, and allowed environment.  It must exclude issue
discussion, fixed revision, patch text, fixed IR, root-cause labels, and
operation-name hints.  The certificate and its artifact hash are frozen before
the evaluator reveals the patch.  If a lower-level case compatible with the
available runtime cannot meet these conditions, Phase 3 remains open rather
than promoting an analyst-guided or prospective case to ground truth.

The executable protocol is
`theory_oracle/historical_evaluation_protocol_v0_1.py`: `seal` binds the
generic locator certificate to a pre-reveal digest; `score` consumes an
evaluator-owned truth label only afterwards.  The score reports stage and
candidate-mechanism coverage, reduction size, erroneous kernel descent, and
stopping-decision agreement.  It is fail-closed on a modified certificate,
truth/certificate mismatch, or missing independent-evaluator attestation.  A
digest records an honest protocol boundary; it cannot by itself prove what an
analyst privately knew.

### Phase 4 — complex held-out training case

Run the frozen method on one Megatron matched step, not a free training
trajectory.  Success is a material automatic reduction from the whole step to
a small stage/region candidate set with explicit production/mediation evidence
and honest stopping.  It is not required to produce a unique source line.
Qwen remains a development/regression subject only.

The Phase-4 metric is not a claimed unique source line.  It is automatic
reduction from the whole matched step to a small auditable candidate set with
production, mediation, provenance, and stopping evidence.  The reported path
is `whole step -> stage -> regions -> optional operations/kernels`; actual
counts are measured, not promised in advance.

Current readiness is recorded in
`reports/PHASE4_MEGATRON_ASSET_AUDIT_V0_1.md`.  Absence of an installed
Megatron package or frozen state is an asset gap, not permission to substitute
Qwen or a free-running toy trajectory for this phase.

## Non-negotiable controls

- Eager is a comparison baseline unless an independent contract makes it
  normative.
- Do not infer source from first divergence or maximum delta.
- Do not infer source from mediation alone, or endpoint effect from production
  alone.
- A repair result is intervention-dependent unless graph, fusion, kernel,
  shape/stride/layout/dtype, autograd path and compiler configuration gates
  hold.
- A kernel-only result is invalid when the first observed failing stage is
  earlier and the lower-level path is not proven to preserve the realization.
- Raw artifacts are stored separately from derived verdicts and are keyed by
  case ID, state ID, Torch/compiler version, GPU and configuration.
