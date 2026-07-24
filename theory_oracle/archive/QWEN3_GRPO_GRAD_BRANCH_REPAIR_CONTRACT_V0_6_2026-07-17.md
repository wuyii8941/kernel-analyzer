# Qwen3 GRPO Grad-Branch Repair Contract v0.6 — 2026-07-17

## Status

Frozen before v0.6 execution. This supersedes neither the v0.4 event bank nor
the preserved v0.5 `INVALID` result.

## Why v0.6 exists

The v0.5 executor compared a hash of a flattened scorer tensor with the frozen
hash of the native Trainer tensor. The 512 float32 values were independently
shown to be exactly equal, while the hashed shapes were `[512]` and `[4,128]`.
Because shape is part of the hash domain, v0.5 correctly rejected its own
mis-specified gate.

v0.6 makes exactly one semantic change to the executor: before hashing, the
complete scorer is put into the frozen native batch shape `[4,128]`. It does not
change any value, selected event, branch treatment, optimizer probe, compiler
path, tolerance, or acceptance threshold.

## Frozen witness and intervention

Use trajectory A, optimizer step 14, rollout batch 4, policy iteration 2 and
flat token index 277 from the valid v0.4 bank. The eager event is unclipped and
the compiled event is clipped for a negative-advantage token.

Run three one-step arms from the same saved parameters and batch:

1. `A_reference`: eager scorer and its ordinary branch;
2. `B_candidate`: compiled scorer and its ordinary branch;
3. `C_branch_repair`: the same compiled scorer tensor and graph as B, changing
   only the selected token's branch action to the reference action.

The optimizer is a deliberately controlled SGD probe, not a reconstruction of
the natural AdamW/GRPO training update.

## Acceptance rules

- all eager calls hash exactly to the frozen native `[4,128]` eager hash;
- all candidate calls hash exactly to the frozen native `[4,128]` candidate hash;
- B and C have identical full scorer hashes and compiled graph identities;
- the selected scalar, old log-probability and event directions exactly match
  the bank;
- the ordinary compiled branch has zero selected-logp loss derivative while the
  repaired branch has a nonzero derivative;
- an independent verifier recomputes model distances from saved weights.

No post-execution tolerance or witness replacement is allowed.

## Claim boundary

Passing estimates a branch-intervention effect for one frozen state under this
controlled optimizer probe. It does not establish compiler incorrectness,
operator cause, natural-update magnitude, prevalence, or long-run harm.
