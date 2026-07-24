# PyTorch #115260 — pre/post-reveal audit v0.1

## Bound pre-reveal execution

The pre-reveal experiment ran on CPU with PyTorch `2.2.0+cu121` in the
isolated `pt220_clean` environment.  Its contract is contextual equivalence:
adding observability of existing FX values must not alter the final `tan`
output.

Artifact: `results/historical_blind/tan_output_context_115260_v0_1/pre_reveal_certificate.json`.
Frozen certificate hash:

```text
1d342152cdac593e8e7659d629eee734bd6fef7b7aa8c9f0c8ec440feec7a948
```

Before reading the issue discussion or any patch material, the frozen generic
locator reported:

- eager, Dynamo-eager, and AOT-eager satisfy the contract in two repeats;
- Inductor violates it repeatably (`max_abs = 13.0625`);
- 9 FX nodes reduce to the one-minimal context-exposure set `{mul}` (88.9%
  reduction);
- the certificate explicitly stops at `STAGE_FX_CANDIDATE_PRE_REVEAL`.

It did not claim same-input local production, a generated-kernel mapping, a
source line, or a root cause.

## Post-certificate provenance replay

The frozen witness was replayed with Inductor debug artifacts enabled.  The
replay independently reproduces the contextual endpoint (`max_abs = 13.0625`)
and binds the FX candidate to two generated `output_code.py` artifacts whose
explicit C++ fusion symbol is:

```text
cpp_fused_cat_maximum_mul_tan_0
```

The artifact report, including hashes, is
`results/historical_blind/tan_output_context_115260_v0_1/post_certificate_provenance.json`.
This is an auditable FX-to-fused-C++-kernel relation: it is based on the
compiler-emitted fusion symbol, not a filename guess.  It still does not show
which expression inside that fused kernel is faulty, and it does not establish
same-input isolated-region production or a unique cause.

## Post-reveal comparison

Only after certificate freeze, the public issue comments were read.  Developer
analysis associates the symptom with conversion behavior across fused `mul`
and `tan` in Inductor C++ scheduling.  That mechanism overlaps the frozen FX
candidate `{mul}` and stage candidate `Inductor`.

The issue links PR `pytorch/pytorch#118365`.  Public PR metadata records it as
**closed, not merged**, with proposed changes in
`torch/_inductor/codegen/cpp.py` and a CPU repro test.

## Allowed conclusion

This is a successful **blind stage-to-fused-C++-kernel candidate with
post-reveal developer mechanism agreement**.  It demonstrates that the
automatic stage screen and FX context-exposure reducer can narrow a real
historical symptom without reading the later mechanism, then follow explicit
Inductor provenance to a fused generated kernel.

It is **not** `EXTERNAL_PATCH_VALIDATED`: the observed PR was not merged, and
we have not bound a fixed released revision that clears the witness.  It does
not satisfy the Phase-3 merged-patch accuracy gate.  No root-cause or
generated-kernel claim is licensed from this artifact alone.
