# Qwen3 cross-state transition capture contract v0.1

## Purpose

Create two additional complete pre-minibatch training states for testing whether
operator repair effects transport beyond heldout-transport-B step 29.

The existing A and C artifacts contain final checkpoints and paired scorer
measurements, but not the exact step-29 optimizer/scaler/RNG state, target
minibatch and prior compiled-shape history.  They cannot be reconstructed by
combining final checkpoints with token logs.  A and C are therefore replayed
from the same frozen initial model, config and seed with transition capture
enabled.

## Frozen replay arms

- A: `configs/oracle_qwen3_grpo_heldout_transport_v01_a.yaml`, capture step 29;
- C: `configs/oracle_qwen3_grpo_heldout_transport_v01_c.yaml`, capture step 29.

Each replay writes to new paths and refuses to resume from or overwrite the
historical A/C runs.  It also records a fresh grad-enabled eager/compiled anchor
bank so the compiled treatment identity can be verified against the replay's
own trajectory rather than borrowed from an earlier run.  GPU arms run
serially.

## Snapshot validity

A captured state is valid only if the snapshot verifier confirms:

- model parameters/buffers, AdamW, scheduler, native FP16 GradScaler and full
  Python/NumPy/Torch RNG are present;
- the exact target minibatch and every declared prior shape-history minibatch
  exist and match their manifests;
- capture preserved RNG, gradients and tensor versions;
- the target is policy iteration 2 at optimizer step 29;
- the corresponding fresh grad-state anchor is candidate-valid, preserves
  training state and matches the target scorer hashes.

Historical A/C scorer hashes are a reproducibility diagnostic, not a validity
gate.  If replay differs from the old trajectory but passes all frozen-state
and candidate-identity gates, it remains an independent sample from the same
declared config/seed protocol and must be labelled as a replay state.

## Claim boundary

Two extra snapshots do not estimate a population distribution by themselves.
They enable a three-state transport check for predeclared operator
interventions.  Eager remains a baseline rather than a correctness authority.
