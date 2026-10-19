# Transformers Llama supplied-table RoPE implementation review — 2026-10-19

The active ROPE contract covers one versioned public function:
`transformers.models.llama.modeling_llama.apply_rotary_pos_emb` from Transformers 4.57.3. It receives complete
rank-4 q and k tensors, complete `[B,L,D]` cos and sin tables, `unsqueeze_dim=1` and a null deprecated
`position_ids`. The profile also returns the complete VJP with respect to q and k for explicit upstream tensors. All
six tensors are finite physical f16, bf16, f32 or f64 values, and the last dimension is even.

The retained 4.57.3 source gives the task directly. `rotate_half` splits the last dimension in half and concatenates
the negated second half with the first. `apply_rotary_pos_emb` broadcasts the supplied tables over the head dimension
and applies `x*cos + rotate_half(x)*sin` independently to q and k. Differentiating that linear map gives, for paired
coordinates `p` and `s=p+D/2`,

```
out[p] = x[p] * cos[p] - x[s] * sin[p]
out[s] = x[s] * cos[s] + x[p] * sin[s]
dx[p]  = g[p] * cos[p] + g[s] * sin[s]
dx[s]  = -g[p] * sin[p] + g[s] * cos[s]
```

This review corrected an important error in the former draft. The usual `LlamaRotaryEmbedding` producer constructs
equal table halves, but the public table-application function consumes every supplied coordinate. Equal halves are
therefore a property of that producer, not an input condition of `apply_rotary_pos_emb`. An optimized implementation
that reads only one table half implements the selected task only when the binding separately establishes that stronger
condition. The active reference preserves unequal table halves and uses them in both the forward value and VJP.

Sixteen contract examples cover unequal halves, multiple q/k head counts, batches, tokens, four physical dtypes,
complete outputs and rejected shapes or task variants. Six deliberately wrong readings are rejected. A separate
coordinate implementation checks all four results without calling the adapter. Twenty public PyTorch formula and
autograd cases cover four dtypes and five shapes/table patterns. The installed Transformers 4.57.6 public function was
also called for the same 20 cases; its selected function bodies match the retained 4.57.3 source and all results agree.

The active profile does not certify table generation, position lookup, partial rotary dimensions, adjacent-pair
interleaving, another broadcast axis, another model family, gradients with respect to cos/sin, special values, or
layout and alias effects. The former five-profile contract and adapter remain under `contracts/draft` and cannot be
used by the active runtime.
