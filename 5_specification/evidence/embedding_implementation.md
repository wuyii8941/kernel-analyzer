# Public dense embedding implementation review — 2026-10-19

The active EMB contract covers one public PyTorch 2.10 `torch.nn.functional.embedding` call and the complete dense
weight VJP. The selected inputs are finite physical f16, bf16, f32 or f64 weight and upstream tensors, nonempty int64
indices, optional valid `padding_idx`, and the public `scale_grad_by_freq` choice. `max_norm` is null and `sparse` is
false.

The retained public documentation defines embedding as row lookup, states that padding positions do not contribute to
the gradient, and defines frequency scaling as inverse word frequency in the mini-batch. The retained PyTorch 2.10
functional source fixes negative `padding_idx` normalization. The retained ATen source independently exposes the
forward `index_select` and the dense backward loop: every non-padding occurrence adds its upstream row, divided by the
complete occurrence count when frequency scaling is enabled.

The active contract does not certify `embedding_bag`, `max_norm` mutation, sparse gradient layout, module
initialization, special values or layout and alias behavior. Those were mixed into the former draft, including two
different renormalization profiles. That broader draft and adapter remain under `contracts/draft`; none of those
branches is used to judge a kernel through the active EMB profile.

Eighteen examples cover repeated and multidimensional indices, positive and negative padding rows, both frequency
choices, complete outputs, physical dtype restrictions and refused tasks. Seven deliberately wrong readings are
rejected. A separate NumPy coordinate construction agrees with the reference, and 20 public PyTorch 2.10 calls cover
four floating dtypes and five index, padding and frequency cases. Additional checks reject missing fields, wrong output
dtype and wrong shapes.

The strict task value uses exact real addition and exact division by an integer count. Device accumulation rounding is
measured separately. The evidence is therefore strong for the declared dense task and deliberately silent about the
removed branches.
