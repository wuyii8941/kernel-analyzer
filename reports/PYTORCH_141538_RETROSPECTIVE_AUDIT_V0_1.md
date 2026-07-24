# PyTorch #141538 — retrospective lower-level audit v0.1

## Status

This is a **retrospective development case**, not a Phase-3 hidden external
score.  The public issue and merged PR `#144395` (commit `ccc2878c9782`) were
read while choosing the case.  Removing that information from a later package
would not make the present analysis blind.

## Old-runtime witness

The bound failing environment is PyTorch `2.5.1+cu121`, Triton `3.1.0`, CUDA
12.1, Tesla T4.  With an explicit, content-fingerprinted `_random_samples`
input, eager, Dynamo-eager and AOT-eager agree while Inductor is repeatably
wrong by `2.6380972862243652`.  The original-RNG and explicit-boundary screens
are respectively:

- `results/historical_candidate_screen/fractional_maxpool_141538_v0_1/screen.json`
- `results/historical_candidate_screen/fractional_maxpool_explicit_samples_141538_v0_1/screen.json`

Making the random samples an explicit input matters: it turns a stochastic
API parameter into a same-input replay boundary.  It does not make eager a
mathematical truth; here equality is the declared implementation contract.

## Local production and provenance

The fresh cache capture binds the two exact input fingerprints, runs eager and
compiled twice, and records a reproducible local discrepancy.  Its exported
ATen graph contains `aten.fractional_max_pool2d`; the compiler-emitted wrapper
contains the matching source annotation and invokes:

```text
triton_poi_fused_fractional_max_pool2d_0
```

Artifact: `results/historical_blind/fractional_maxpool_141538_v0_1/local_evidence_report.json`.
The generated code and debug IR are retained under the same case directory's
`inductor_cache_r3/` and `torch_compile_debug_r3/` paths.

This establishes a same-input, reproducible **local discrepancy producer**
with auditable `ATen -> emitted wrapper -> Triton symbol` provenance.  It does
not establish that an individual line in the wrapper is uniquely faulty.

## Fixed-runtime control

On PyTorch `2.11.0+cu126`, Triton `3.6.0`, CUDA 12.6, Tesla T4, the identical
explicit-boundary contract clears in two compiled repeats (`max_abs = 0`).
Artifact: `results/historical_post_reveal/fractional_maxpool_141538_v0_1/fixed_runtime_control.json`.

This is a useful post-reveal compatibility control, but it is **not** a
one-commit causal proof: Torch, Triton and CUDA versions all changed together.
The merged patch is relevant only to the retrospective mechanism comparison.

## Claim gate and stopping decision

The machine verifier returns `LOCAL_INJECTION`.  It deliberately does not
upgrade because this one-output functional program has no fixed nontrivial
suffix for mediation and no context-preserving repair/no-op intervention.

Not established:

- a Phase-3 blind localization score;
- endpoint mediation beyond the pool output itself;
- a non-target-context-preserving intervention;
- a unique lowering pass, operator, kernel expression, or root cause.

The correct next action is not to force a source-line diagnosis.  Use this
case as a lower-level provenance/replay regression while an independent party
prepares genuinely withheld historical cases for Phase 3.
