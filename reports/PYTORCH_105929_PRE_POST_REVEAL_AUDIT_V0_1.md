# PyTorch #105929 — pre/post-reveal audit v0.1

## Bound pre-reveal execution

The blind run used the isolated `pt220_clean` environment with PyTorch
`2.2.0+cu121` on CPU.  The declared semantic contract was intentionally
non-numerical: a `TorchDispatchMode` that rewrites `aten.add` to `aten.mul`
must remain observable under compilation.

Artifact:
`results/historical_blind/torchdispatch_105929_v0_1/pre_reveal_certificate.json`.
The certificate was frozen before reading the issue discussion, fixed
revision, or patch.  Its hash is:

```text
8e39fe3fba5205f1f27a9790c35c244612c4a35eea2f96a6052116019628aefd
```

The generic stage screen observed the contract in eager, Dynamo+eager, and
`aot_eager`, and observed a repeatable violation in Inductor.  FX inventory
contained a single natural compute node, `add`; consequently there is no real
search-space reduction to score.  The certificate stopped at
`STAGE_FX_CANDIDATE_PRE_REVEAL` and explicitly did not claim a first bad pass,
root cause, source line, kernel, or patch agreement.

The certificate was emitted before the generic core gained the explicit
`UNREDUCIBLE_SINGLETON_INVENTORY` status.  Its historical
`ONE_MINIMAL_CANDIDATE_SET` field is therefore retained as raw provenance but
is superseded by this audit: an inventory of one candidate is enumeration, not
delta reduction.

## Post-reveal ground truth

After certificate freeze, public PyTorch metadata was inspected.  Issue
[#105929](https://github.com/pytorch/pytorch/issues/105929) is fixed by merged
commit
[`93979e70631a`](https://github.com/pytorch/pytorch/commit/93979e70631ae90afe26c25ef620b311c9b6a8f5),
PR [#131828](https://github.com/pytorch/pytorch/pull/131828), titled
`Skip frame if torch dispatch mode enabled`.  The actual semantic repair is
in Dynamo frame-capture logic (`torch/_dynamo/convert_frame.py`), with dispatch
mode infrastructure and tests.  It is not an Inductor generated-kernel patch.

## What this comparison teaches

This is an **external counterexample to an overly strong stage inference**.
On this old runtime, the chosen backend matrix made the violation first
visible under the `inductor` backend.  That does *not* prove that Inductor is
the source or that Dynamo has been excluded: backend invocation is not a
strictly nested sequence of compiler transformations when a dispatch mode can
alter capture decisions.

The case is therefore not a Phase-3 success and not a failed implementation
detail to hide.  It demonstrates two required corrections:

1. `first observed failing backend` is an observation, not a first-bad-stage
   claim, unless the exact transformation nesting is independently verified.
2. A one-node FX inventory is an operation witness, not a
   symptom-preserving reduction result; no candidate-set shrinkage may be
   reported when no executable variant removed or replaced that node.

## Allowed conclusion

The run establishes a repeatable historical semantic-contract witness and a
frozen blind **backend observation**.  It does not cover the merged patch
mechanism and does not validate operation/kernel localization.  Its permanent
role is a negative calibration case for stage semantics and claim gating.
