# Current conclusion

The project asks when a numerical difference introduced by one LLM training
operator becomes a parameter-update error that keeps accumulating.

## What is established

1. Four core models were audited systematically at sequence lengths 64, 128,
   and 256. Six additional models were used for targeted or held-out checks.
2. Three bounded historical records show short-horizon directional direct operator or
   backward differences under stateless SGD: Liger fused CE, Phi
   `lm_head dX`, and the Qwen `lm_head dX` family. They must be described as
   short, optimizer-specific results.
3. A corrected same-AdamW 16/32-step evaluation contains all twelve mechanically
   sampled rows plus the three historical rows. Liger and Phi trigger the
   short screen; Qwen does not, while one sampled Phi row is a small-margin
   candidate. This table is no longer used as the final long-horizon label.
4. In the warm-state 4096-step review, Liger, Phi, Qwen, Llama and Ministral
   `lm_head dX`/fused CE records retain robust direct direction. Llama and
   Ministral reproduce the same lm-head family direction on new operand
   distributions. Qwen3-VL SiLU shows long-run feedback separation with a
   paired loss gap. The Qwen64 `v_proj`, Qwen seq64 `v_proj`, and saved-P
   direct directions diffuse, but their live candidate/repair trajectories
   show long-run paired loss splits; they are therefore retained as consequence
   controls, not persistent-bias cases. Mamba's completed 4096-step direct audit
   does not exceed its own random baseline, and its separate live paired stage
   did not produce a safe artifact; Gemma4's compatible replay stopped before
   producing a complete artifact. Both remain unresolved/non-positive rather
   than being called negative. Layer-23 abstains because its historical implementation
   identity cannot be replayed, while DeepSeek `dV` remains unresolved for
   formation reasons. The complete audit contains 100 rows across 23 unique
   matrix IDs. The frozen short-screen candidate pool contributes 69 rows;
   they remain unresolved until their 4096-step replay completes and are not
   treated as negatives.
5. Error size alone does not identify short directional formation. Across 32 reachable,
   nonzero rows, local RMS has Pearson correlation `0.018` with persistence.
6. On the corrected 15-row same-AdamW evaluation, the 16-step local-update
   score has AUROC `0.944`, recall `3/3`, and false positives `2/12`.
   Same-level local-update RMS has AUROC `0.528`. The old 14-row AUROC is
   withdrawn because it mixed optimizers and omitted a sampled positive row.

## What is not established

- The historical short-horizon rows and current 4096-step results do not
  prove one universal property for every operator.
- Liger, Phi, Qwen, Llama and Ministral have a robust 4096-step direct audit
  and an observed paired parameter/loss split under controlled one-carrier
  training. SiLU has a feedback-sustained long-horizon bias and a recorded
  paired loss gap, but is not a direct-source case. Qwen64 `v_proj`, Qwen seq64 `v_proj`
  and saved-P are separate long-loss-split consequence controls whose direct
  source directions are not persistent; they are not counted as persistent
  bias. These remain controlled runs, not full-parameter training.
- The 15-row result is retrospective and measured on declared parameter
  carriers. The 4096-step review measures same-state direct updates; neither
  is full-parameter training or a converged-loss result.
- Training feedback is common in the sampled controls: 11/12 have locally
  canceling operator updates but persistent later feedback. Those rows are
  not counted as operator-local source bias.
- A generic three-part predictor cannot yet be evaluated fairly. Five cases
  were checked for complete inputs and all five required abstention.

- The audit has also promoted 69 frozen short-screen candidates to a separate
  4096-step long-replay queue. They are not included in the six completed
  result cases and are not called negatives while their replays are pending or
  unresolved.

The defensible result is therefore a measured operator-to-gradient-to-update
map plus a two-level workflow: short direction triage followed by 4096-step
long-horizon review. It is not a universal LLM-training safety certificate or
a claim that loss converges to a different endpoint.
