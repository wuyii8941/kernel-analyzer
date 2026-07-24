# Qwen3 subject scope and repository cleanup policy

Updated 2026-07-23.

## Active subject ladder

The project will use Qwen3 before DeepSeek.  The model size is part of the
validation design, not an implementation detail:

1. **Qwen3-0.6B** — calibration only.  It is small enough to debug replay,
   provenance, and intervention code quickly.  A result on this model is not
   treated as evidence that the method scales.
2. **Qwen3-1.7B (dense)** — first substantive subject.  It keeps the same
   architecture family while increasing the number of blocks, activation
   ranges, and generated kernels.  The first blind bug-localization cases
   should use this tier when the runtime and checkpoint are available.
3. **A larger dense Qwen3 (4B or 8B, subject to the available GPU budget)** —
   transfer check.  This tests whether the analysis survives larger graphs,
   without introducing MoE routing as a new confounder.
4. **Qwen3-MoE or DeepSeek-V2-Lite** — later stress subjects.  These are
   valuable for routing/parallelism effects, but should not be mixed into the
   first pipeline validation because they add expert dispatch, distributed
   layout, and more complicated provenance.

The official Transformers documentation describes dense Qwen3 models from
0.6B through 32B and documents Qwen3-MoE separately.  Thus the dense ladder
is a controlled scale-up, while MoE and DeepSeek are deliberate transfer
tests rather than replacements for the baseline.

## What counts as a usable Qwen3 bug case

A case is eligible for blind localization only when it has:

- a reproducible buggy revision and a fixed revision;
- an input/state artifact that can be replayed without the issue description;
- a declared endpoint (tensor, semantic event, gradient/update, or loss);
- an independently documented fix or root-cause discussion;
- enough compiler artifacts to compare stage and provenance evidence.

Qwen-specific reports that are only RNG/configuration mismatches are retained
as negative controls.  They must not be advertised as compiler correctness
bugs.  Open issues without a fixed revision are prospective cases, not ground
truth for localization accuracy.

## Repository cleanup boundary

The active path is now:

```text
matched-state Oracle v0
  -> blind Qwen3 case package
  -> generic observation/stage/provenance capture
  -> region replay and controlled intervention
  -> post-reveal comparison with the real fix
```

The following are not active decision procedures and must not be used to rank
operators or claim a root cause:

- fork-count-only and raw-delta-only experiments;
- synthetic twin-trajectory safety claims;
- exploratory one-trajectory significance tests;
- model-specific candidate lists that select a region before blind capture.

They remain in the tree only when their reports or manifests are needed to
reproduce historical evidence.  Generated caches and process logs are
cleanable; compact reports, manifests, and old definitions are retained as an
audit trail.  This prevents a cleanup from silently changing the evidence
base.

