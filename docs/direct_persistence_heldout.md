# What the first unseen implementation test shows

Gemma 4 was frozen as a new implementation before its trajectory was run.  It
is kept separate from the later atlas-derived pool because that pool is missing
exact repair and state identities.

The v4 adapter checks the model revision, the frozen wrapper hashes, the
separate 16-state formation bank, the separate 32-state trajectory bank, the
repair target, and the AdamW settings.  It then reports:

| item | result |
|---|---:|
| 16-step direct/local score | `A=0.9860` |
| 16-step local null upper 95% | `1.0125` |
| 32-step direct/local score | `A=1.0003` |
| 32-step actual trajectory score | `A=3.2312` |
| 32-step feedback score | `A=3.2340` |

The frozen source prediction was negative for direct persistence, and the
full confirmation agrees: Gemma is **not** a direct-persistence positive under
the declared screen.  Its actual trajectory is strongly separated, but that
separation is feedback-dominated and outside the direct screen's claim.

This is one genuine `NEW_IMPL` negative.  It does not provide recall or AUROC:
there is no positive in this one-row pool.  More mechanically frozen unseen
implementations are required before making a generalization claim.

Artifacts:

- `results/property/direct_persistence_v4/heldout_gemma_pool.json`
- `results/property/direct_persistence_v4/heldout_gemma_predictions.json`
- `results/property/direct_persistence_v4/heldout_gemma_confirmation.json`
- `results/property/direct_persistence_v4/heldout_gemma_validation.json`
- `results/property/direct_persistence_v4/heldout/gemma4_e2b_norm_short_screen.json`
- `results/property/direct_persistence_v4/heldout/gemma4_e2b_norm_consequence32.json`
