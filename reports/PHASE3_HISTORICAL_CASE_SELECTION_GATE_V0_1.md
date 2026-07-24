# Phase-3 historical-case selection gate v0.1

This gate is evaluated *after* freezing the generic locator and *before*
running a purported external validation.  A reproducible mismatch is not
enough: the fixed revision and patch must be independently attributable and
withheld from the locator before certificate freeze.

## Higher-level stopping candidate: qualified

`tvm_onnx_gather_negative_index` was historically qualified, but its fresh
rerun is now invalid for frozen-core scoring.

- Buggy checkout: `/data1/tzh/tvm_bug_19436` (`6b27d1949`).
- Fixed checkout: `/data1/tzh/tvm_fix_19436` (`e370fc737`).
- Public metadata: Apache TVM issue `#19436` and fix PR `#19525`.
- Independent reference: ONNX Runtime semantic output.
- Existing pre-reveal certificate, fixed-run and post-reveal assessment are
  bound in `results/operator_oracle/tvm_gather_negative/`.
- Correct outcome is a frontend/Relax stopping decision, not a kernel claim.

The fresh rerun is documented in `TVM_GATHER_RERUN_INVALID_V0_1.md`: two
buggy runs have different undefined negative-index outputs, so the
repeatability gate rejects the certificate.  Its former reports are historical
evidence only and it may not be counted as the first Phase-3 validation unless
a deterministic witness is recovered without changing the semantic case.

## Lower-level generated-kernel candidate: not yet qualified

`pytorch_adaptive_avgpool_flatten_sum` is reproducible on PyTorch 2.11/T4 and
has same-input replay plus an intervention-dependent generated-kernel result.
However, public PyTorch issue `#180956` has no independently verified merged
developer patch in the current evidence ledger.  Its repository-local
expression repair is a mechanism hypothesis, not external ground truth.

Therefore it remains a **prospective kernel regression case**.  It must not be
used to report Phase-3 localization accuracy, even if the frozen locator
returns the same kernel candidate.

### GPU mechanism candidate: PyTorch #122260

On the bound PyTorch `2.2.0+cu121` / Tesla T4 runtime, the compact GPU witness
from PyTorch #122260 is deterministic: eager, Dynamo+eager and AOT-eager
produce finite `1`, whereas Inductor produces `inf` in two repeats.  The raw
screen is
`results/historical_candidate_screen/fma_context_122260_v0_1/screen.json`.

This is the current best **GPU kernel-pipeline mechanism case** because it
creates a real lower-level numerical contract violation suitable for FX →
Inductor → Triton provenance, same-input replay and controlled intervention.
It still cannot score Phase 3: the issue and mechanism were used to select the
candidate, and its linked ghstack PR stack is closed/unmerged.  It is retained
to test whether the method can honestly descend to a generated kernel, not to
claim hidden external localization accuracy.

## Retrospective merged-lowering development candidate: PyTorch #141538

PyTorch #141538 passes the environment qualification gate on isolated,
version-matched PyTorch `2.5.1+cu121` / Triton `3.1.0` / Tesla T4.  The
original RNG-reset witness is exact and repeatable in eager, Dynamo+eager, and
AOT-eager, while Inductor is repeatably wrong (`max_abs = 2.6380972862243652`).
The raw record is
`results/historical_candidate_screen/fractional_maxpool_141538_v0_1/screen.json`.

Public commit `ccc2878c9782` / PR #144395 is merged and changes
`torch/_inductor/lowering.py` plus its test.  It is useful for checking the
lower-level provenance/replay path, but it cannot be a Phase-3 blind score in
this workspace: its public issue and patch mechanism were consulted while
selecting it.  A later certificate can only be labelled *retrospective*.
Removing patch text from a package after the analyst has read it would not
restore blinding.  A separately withheld evaluator must supply the actual
scored lower-level case.

## Post-reveal negative calibration: PyTorch #105929

PyTorch #105929 has a reproducible old-runtime witness and an independently
merged fix (PR #131828), but it **does not qualify as a Phase-3 success**.  The
frozen backend matrix showed the failure under the Inductor backend, whereas
the merged fix is in Dynamo frame capture.  The difference is informative:
backend labels are not by themselves a strict transformation bisection when a
TorchDispatch mode changes capture semantics.  In addition, the FX graph has
one natural compute node, so it cannot score reduction.

The full pre/post-reveal audit is
`reports/PYTORCH_105929_PRE_POST_REVEAL_AUDIT_V0_1.md`.  Retain it as a
negative calibration test for claim gating; do not count it as either of the
two external validations.

## Rejected substitutes

- Qwen high-order-gradient and Qwen/TorchTitan cases are development/regression
  evidence under the current subject policy, not independent Phase-3 scoring.
- BF16 hardtanh is inapplicable on Tesla T4.
- Current expanded-index candidate fails closed rather than instantiating a
  silent wrong-result witness.
- TVM ScatterElements (`#19435` / `#19527`) has external revisions but is
  another frontend converter fix, not a lower-level kernel case.

## Required next input

To execute the intended two-case external evaluation without weakening its
claim, an independent evaluator must supply one withheld higher-level case and
one withheld, reproducible lower-level Inductor/Triton/kernel case, each with
a buggy revision, minimal reproducer, and later fixed revision/PR.  Until then
the project has calibration and retrospective mechanism evidence, but no
external Phase-3 accuracy score.
