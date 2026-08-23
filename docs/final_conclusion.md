# Current conclusion

The project asks when a numerical difference introduced by one LLM training
operator becomes a parameter-update error that keeps accumulating.

## What is established

1. Four core models were audited systematically at sequence lengths 64, 128,
   and 256. Six additional models were used for targeted or held-out checks.
2. The current paper headline contains three operator-local source/transport
   persistence cases: Liger fused CE, Phi `lm_head dX`, and Qwen `lm_head dX`.
3. On the same ordered 32-state measurements, Liger is already directional at
   the operator output. Phi and Qwen become more directional after the real
   backward pass reaches a parameter gradient.
4. Error size alone does not identify these cases. Across 32 reachable,
   nonzero rows, local RMS has Pearson correlation `0.018` with persistence.
5. On the frozen 14-row Oracle evaluation, the 16-step effective-update score
   has AUROC `1.00`, recall `3/3`, and false positives `2/11`. Local RMS has
   AUROC `0.242`; BF16 dtype alone has AUROC `0.50`.

## What is not established

- The three cases do not prove one universal property for every operator.
- The 14-row result is retrospective and measured on declared parameter
  carriers, not full-parameter training.
- Training feedback is common in the sampled controls: 11/12 have locally
  canceling operator updates but persistent later feedback. Those rows are
  not counted as operator-local source bias.
- A generic three-part predictor cannot yet be evaluated fairly. Five cases
  were checked for complete inputs and all five required abstention.

The defensible result is therefore a measured operator-to-gradient-to-update
map plus a useful short persistence screen in a declared scope. It is not a
universal LLM-training safety certificate.
