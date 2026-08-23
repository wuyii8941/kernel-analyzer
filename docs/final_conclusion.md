# Current conclusion

The project asks when a numerical difference introduced by one LLM training
operator becomes a parameter-update error that keeps accumulating.

## What is established

1. Four core models were audited systematically at sequence lengths 64, 128,
   and 256. Six additional models were used for targeted or held-out checks.
2. Three bounded historical records show persistent direct operator or
   backward differences under stateless SGD: Liger fused CE, Phi
   `lm_head dX`, and the Qwen `lm_head dX` family. They must be described as
   optimizer-specific results.
3. A corrected same-AdamW evaluation contains all twelve mechanically sampled
   rows plus the three historical rows. Liger and Phi remain locally
   persistent; Qwen does not, while one sampled Phi row is a small-margin
   positive. The exact Qwen seq128 rerun measures output, gradient, AdamW
   update and the live 32-step result in one invocation.
4. Error size alone does not identify these cases. Across 32 reachable,
   nonzero rows, local RMS has Pearson correlation `0.018` with persistence.
5. On the corrected 15-row same-AdamW evaluation, the 16-step local-update
   score has AUROC `0.944`, recall `3/3`, and false positives `2/12`.
   Same-level local-update RMS has AUROC `0.528`. The old 14-row AUROC is
   withdrawn because it mixed optimizers and omitted a sampled positive row.

## What is not established

- The three historical SGD cases and the three AdamW-positive rows do not
  prove one universal property for every operator.
- Liger, Phi and Qwen now all have exported direct-operator, later-state
  feedback, and actual parameter-separation sequences under AdamW. They remain
  one-parameter controlled training runs, not full-parameter training.
- The 15-row result is retrospective and measured on declared parameter
  carriers, not full-parameter training.
- Training feedback is common in the sampled controls: 11/12 have locally
  canceling operator updates but persistent later feedback. Those rows are
  not counted as operator-local source bias.
- A generic three-part predictor cannot yet be evaluated fairly. Five cases
  were checked for complete inputs and all five required abstention.

The defensible result is therefore a measured operator-to-gradient-to-update
map plus a useful short persistence screen in a declared scope. It is not a
universal LLM-training safety certificate.
