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

## Two additional Gemma target checks

Two more targets from the pre-frozen Gemma pool were run in a fresh process
that built its own runtime release. The source result was written before the
32-step consequence run.

| target | direct/local result | actual result | status |
|---|---:|---:|---|
| softmax backward, `k_norm` | `A32=0.000`, no carrier effect | `1.5e-8` | not applicable: no observed carrier difference |
| GELU/loss backward, projection | `A32=1.0002` | `A32=3.0267` | feedback control, not direct persistence |
| another GELU backward region (`backward:1860`) | `A32=0.000`, no carrier effect | `A32=0.000` | not applicable: no observed carrier difference |

These add controls, not new positives. The first target had no measurable
effect on the chosen carrier, and the second had a large final separation only
after feedback; neither can be used to claim that the whole model is safe.
The compact audit is
`results/property/direct_persistence_v4/heldout/new_impl_targets_v2.json`.
The third row was selected by a rule frozen before its consequence run; its
runtime release and state banks are recorded in
`results/property/direct_persistence_v4/heldout/new_impl_pool_v3.json`.

Artifacts:

- `results/property/direct_persistence_v4/heldout_gemma_pool.json`
- `results/property/direct_persistence_v4/heldout_gemma_predictions.json`
- `results/property/direct_persistence_v4/heldout_gemma_confirmation.json`
- `results/property/direct_persistence_v4/heldout_gemma_validation.json`
- `results/property/direct_persistence_v4/heldout/gemma4_e2b_norm_short_screen.json`
- `results/property/direct_persistence_v4/heldout/gemma4_e2b_norm_consequence32.json`
- `results/property/direct_persistence_v4/heldout/new_impl_pool_v3.json`
