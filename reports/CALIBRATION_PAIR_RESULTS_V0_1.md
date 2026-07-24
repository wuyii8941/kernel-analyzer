# Minimum-sufficient calibration pair — results v0.1

These are instrumentation calibrations.  They validate evidence plumbing, not
compiler-bug discovery or localization accuracy.

## A. Generated-kernel plumbing

Environment: PyTorch `2.11.0+cu126`, Tesla T4, CUDA 12.6.  Artifact:
`results/calibration/kernel_plumbing_v0_2/seeded_fault/seeded_fault_record.json`.

- Integer-valued FP32 input made the unmodified eager/Inductor reduction exact.
- The captured wrapper and an independent no-op copy both matched reference.
- A calibration-only hidden `tl.sum` expression mutation changed all four
  outputs by exactly `1.0` under the same input.
- Restoring the declared expression recovered exact reference output.
- The captured non-target wrapper signature was invariant.

This passes generated-code capture, provenance, replay, no-op and direct
intervention plumbing.  It does **not** establish graph reduction, automatic
candidate selection, or localization accuracy: there is only one meaningful
generated region and the seed recipe is known to this harness.

## B. One-step training

Environment: PyTorch `2.11.0+cu126`, Tesla T4, CUDA 12.6.  Artifact:
`results/calibration/tiny_training_v0_1/training_calibration_record.json`.

- The frozen state contains model, AdamW state and deterministic batch hashes.
- Regions are encoder (`linear → layernorm → gelu`), head, loss/backward,
  clipping and AdamW update.
- A declared calibration-only head-boundary shift of `0.125` changed the
  fixed-suffix gradient (`max abs 0.01041667`).
- The unseeded model was screened through eager, Dynamo-eager, AOT-eager and
  Inductor.  The declared *forward* stage contract is
  `allclose(rtol=1e-5, atol=1e-6)`, because an exact-equality screen initially
  (correctly) exposed harmless reduction-order rounding in Inductor.  All four
  unseeded stages satisfy the declared envelope; the only failing stage in the
  calibration certificate is the explicitly labelled seeded boundary.
- With a pre-recorded midpoint clip threshold, the reference norm `1.29048`
  did not clip and the seeded candidate norm `1.31234` did clip.
- Parameter update and optimizer next state changed; no-op and repair restored
  the complete one-step transition exactly.
- The frozen generic reducer issued four subset queries.  For **each** query,
  the adapter restored model/optimizer/batch state and ran the complete
  `forward → loss → backward → clip → AdamW` transition; its predicate was the
  declared clip/update/optimizer-state contract, not a raw-delta threshold.
  It reduced four anonymous dataflow regions to `r_head` (75% reduction).

This confirms that the method's production and mediation vocabulary and
symptom-preserving reduction can be instantiated through a real training step.
It is not a compiler localization result because the seed is explicitly
injected at a known boundary; only the calibration adapter knows how to
dispatch that seeded implementation variant.

## Consequence

The calibration pair clears Phase 1.  The generic, case-agnostic core was then
frozen in Phase 2; it owns stage screening, delta reduction, certificate
assembly and claim gating, while case adapters own only execution/replay.
The active gate is Phase 3: externally patched cases must now demonstrate that
the same core localizes a real historical violation without a seeded fault.
