# Historical Bug Ground-Truth Audit v0.1

This audit was performed only after the blind case package and localization
certificate were frozen. It separates external validation from our own
mechanism hypothesis.

## Case 001 (`pytorch_adaptive_avgpool_flatten_sum`)

The public upstream issue is [PyTorch issue #180956](https://github.com/pytorch/pytorch/issues/180956).
The issue is a valid external wrong-result witness, and our target environment
reproduces it. However, no developer-confirmed fixed commit or merged patch was
identified in the available upstream metadata during this audit.

The repository's private `issue_draft.md` contains a mechanism explanation and
a one-expression repair experiment. That material was not available to the
blind locator and is not independent ground truth. It can be used only as a
post-hoc mechanism comparison.

Post-hoc, its described stride correction agrees with the expression tested in
our certificate. This agreement is useful calibration evidence, but it must not
be reported as an independently scored localization result because both the
case notes and the tested generated artifact are repository-local evidence.

Therefore Case 001 is classified as:

`PROSPECTIVE_UNKNOWN_CASE_WITH_BLIND_CERTIFICATE`

It demonstrates that the pipeline can produce a constrained localization
certificate, but it cannot be used to report stage/localization accuracy
against an independent developer patch.

## Other candidates

- The expanded `index_add` candidate is currently fail-closed on the available
  nightly and is not a silent wrong-result witness there.
- The layernorm/reciprocal candidate is a reproducible wrong-result fallback,
  but no independent fixed patch has been verified in this audit.
- The hardtanh/bfloat16 boundary-gradient case is locally marked closed/fixed,
  but the current Tesla T4 warns that bfloat16 compilation is unsupported;
  eager and compiled gradients match there. It is therefore `INAPPLICABLE_ON_T4`
  and cannot serve as the second benchmark without native-BF16 hardware.

## Consequence for claims

The current certificate is valid evidence of observation, local production,
provenance, and intervention-dependent attribution. It is not a measured
historical localization accuracy result. A future externally fixed case must be
run with the same frozen protocol before any accuracy claim is made.
