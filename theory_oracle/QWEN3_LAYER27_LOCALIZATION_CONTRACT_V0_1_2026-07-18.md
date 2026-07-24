# Qwen3 decoder layer 27 localization contract v0.1

## Purpose and terminology

This experiment determines whether the final decoder layer (zero-based layer
27) contains an implementation-specific contribution to the frozen step-29
Qwen3 scorer discrepancy.

`Qwen3DecoderLayer` is a composite module, not a single operator. Therefore this
experiment is explicitly **localization**, not operator attribution. A nonzero
result only authorizes follow-up interventions on the layer's individual
RMSNorm, Linear, attention and residual-add invocations.

## Arms

The model is split into prefix (embedding through layer 26), target (layer 27),
and tail (final RMSNorm plus `lm_head`):

- `split_EEE`: all three eager;
- `split_CCC`: all three compiled;
- `repair_CEC`: compiled prefix, eager layer 27, compiled tail;
- `injection_ECE`: eager prefix, compiled layer 27, eager tail;
- `whole_eager`: original endpoint anchor.

## Fail-closed gates

- `whole_eager` equals the frozen eager scorer hash.
- `split_EEE` equals `whole_eager` bit-for-bit.
- `split_CCC` equals the frozen whole-compiled scorer hash bit-for-bit.
- all arms are bit-exact over two repetitions.

If splitting the decoder changes either endpoint, mixed-arm contrasts are not
interpretable for the original candidate. No fusion/layout root-cause claim is
allowed from an invalid split.

## Interpretation if valid

- `split_CCC -> repair_CEC` is a selected-state repair contrast for the entire
  layer-27 realization.
- `split_EEE -> injection_ECE` is the corresponding injection contrast.
- zero in both directions localizes the discrepancy before layer 27;
- nonzero contrasts motivate within-layer operator experiments, but do not by
  themselves identify an operator or prove necessity/sufficiency.

State, runtime protocol, anchors and selected-token observable are inherited
unchanged from the preceding Qwen3 operator pilots. Correctness, population and
long-run claims remain out of scope.
