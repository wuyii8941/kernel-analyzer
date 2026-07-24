# Historical Bug Blind Localization Protocol v0.1

Status: protocol retained and superseded operationally by
`theory_oracle/historical_evaluation_protocol_v0_1.py`.  Case 003 is explicitly
marked `PATCH_EXCLUDED_REPLAY` because its fix was read during screening.  Case
004's fresh repeatability control later failed, so it is **not** a current
Phase-3 candidate; see `reports/TVM_GATHER_RERUN_INVALID_V0_1.md` and the
localization evidence ledger.  This document specifies the process, not the
current case roster.

## Purpose

This protocol evaluates whether the operator-analysis pipeline can localize a
reproducible compiler wrong-result witness. It is not a ranking benchmark and
it does not assume that every failure has one faulty ATen operator. The output
is an auditable localization certificate with an explicitly limited claim.

## What is hidden during localization

The analysis input may contain only:

- the buggy runtime/environment;
- a minimal reproducer and its input artifact;
- the reference/specification output contract;
- the compiled output and compiler artifacts produced by the run;
- the preregistered region inventory and stopping rules.

The following must be inaccessible until the certificate is frozen:

- fixed commit and patch files;
- issue comments, pull-request discussion, and root-cause notes;
- labels or filenames that reveal the expected compiler stage.

The evaluator may reveal the hidden patch only after localization and score
the certificate against it. A known historical bug is therefore an external
validation target, not information available to the locator.

## Precondition and fail-closed rules

Each case is first classified as one of:

- `REPRODUCIBLE`: reference/spec and compiled outputs disagree under the
  declared contract, in two independent processes;
- `FAIL_CLOSED`: compilation rejects the program or raises an explicit error;
- `INAPPLICABLE`: the declared environment or trigger condition cannot be
  instantiated;
- `INVALID`: the reference, inputs, or process controls are not stable.

`FAIL_CLOSED`, `INAPPLICABLE`, and `INVALID` are not localization successes.
They must not be converted into a silent-bug claim.

## Frozen success criteria

A successful certificate must contain all of the following:

1. Buggy revision/environment identity, GPU, CUDA, PyTorch/Triton versions,
   compiler configuration, and immutable hashes of the reproducer and inputs.
2. Two independent-process reproductions with stable reference output and
   stable compiled output. The reference may be eager, aot_eager, or an
   independently specified result, but its role must be declared.
3. A declared endpoint: exact output relation, tensor element relation, or a
   semantic event. Argmax/top-k/clipping are optional consequences, not a
   substitute for the output contract.
4. Aligned observation of the graph/IR/kernel boundaries. A first numerical
   difference is a candidate boundary, not automatically a fault location.
5. Same-input local replay for a candidate region, when the region can be
   isolated. This is the only evidence for a `local discrepancy producer`
   claim.
6. Fixed-suffix boundary mediation, when the boundary can be replayed. This
   tests whether the discrepancy can affect the endpoint; it does not prove
   where the discrepancy originated.
7. Provenance from source/module or FX/ATen nodes to the compiled region and,
   where available, the generated kernel. Missing links must be reported.
8. A controlled repair or injection whose non-target context is checked:
   graph count, shapes, strides/layout, dtypes, fusion/kernel inventory,
   launch metadata, and compiler configuration. If this context changes, the
   result is only `intervention-dependent attribution`.
9. Repeatability of the intervention and a no-op instrumentation control.
10. A claim level and limitations. The pipeline may output observation,
    local production, provenance-supported candidate, intervention-dependent
    attribution, stage localization, or cross-level localization. It may not
    output `root cause` merely because a repair removes the mismatch.

## Orthogonal evidence dimensions

For every region, record two separate predicates:

- **Production:** identical boundary input to reference and candidate region
  produces different local output.
- **Mediation:** replacing the boundary value while executing a fixed common
  suffix changes the endpoint.

The four combinations are interpreted as follows:

| Production | Mediation | Allowed interpretation |
|---|---|---|
| no | no | no current evidence for this region |
| yes | no | local numerical producer, no observed endpoint consequence |
| no | yes | upstream discrepancy is endpoint-relevant here; region is not a proven source |
| yes | yes | strong candidate, still requiring provenance and context invariance |

This prevents propagation nodes, amplification nodes, and decision nodes from
being conflated with the point where a discrepancy is introduced.

## Blind procedure

1. Build an immutable case package containing only the allowed inputs above.
2. Preregister regions, endpoint, tolerances, process count, and stopping
   rules. Do not choose a region from a hidden patch.
3. Run the no-op instrumentation control and two independent baseline runs.
4. Run aligned local replay and fixed-suffix mediation for the preregistered
   regions. Preserve raw artifacts separately from derived statistics.
5. Run at most the preregistered interventions. Stop if graph/autograd/kernel
   context is not invariant and downgrade the claim.
6. Seal the certificate with
   `historical_evaluation_protocol_v0_1.py seal` before revealing any patch
   metadata.  The digest is an auditable attestation boundary, not proof of
   private analyst knowledge.
7. Reveal the evaluator-owned compact patch-scope truth only in a separate
   step and run `historical_evaluation_protocol_v0_1.py score`.  It scores:
   stage coverage, mechanism agreement, localization granularity,
   intervention validity, and false localization of propagation nodes.

## Negative controls

The suite must include at least:

- a no-op intervention;
- a benign numerical mutation that changes a continuous value but not the
  declared endpoint;
- a propagation-only boundary where upstream values differ but same-input
  local production does not;
- a case that fails closed on the current runtime.

These controls are required to show that the method does not equate any delta,
first divergence, or successful repair with a compiler fault.

## Claim boundary

The historical bug provides a correctness witness only relative to its declared
contract and hidden external validation. Eager is not automatically a
mathematical truth for unrelated cases. If no independent specification exists,
the result remains implementation-relative discrepancy/localization evidence.
