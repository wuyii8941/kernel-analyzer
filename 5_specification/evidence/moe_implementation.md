# Mixtral top-2 routing and combination implementation review — 2026-10-19

The active MOE contract covers one versioned subtask from
`transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock.forward` in Transformers 4.57.3. It receives
complete router logits `[T,E]` and already computed expert outputs `[T,E,D]`. For each token it selects the two
largest logits, normalises the two selected weights to sum to one, sets the other dense weights to zero, and forms the
weighted sum of the supplied expert outputs. The selected logits must be distinct. Capacity limits, dropped tokens,
router jitter, auxiliary loss and the expert MLP itself are outside this profile.

The retained Mixtral paper defines the sparse block as a weighted sum over the selected experts and gives
`Softmax(TopK(router_logits))` as the gate. The retained Transformers source computes a full float32 softmax, takes
the largest two values, then divides those two by their sum before combining expert results. With finite distinct
logits these are the same routing task in exact real arithmetic. The installed Transformers 4.57.6
`MixtralSparseMoeBlock.forward` has the same parsed function body as the retained 4.57.3 source for this task.

The contract deliberately uses the exact real task as the specification. Transformers explicitly requests float32
for its router softmax, even when the caller supplies another floating dtype, so its intermediate rounding is an
execution property rather than a change to the mathematical target. Tests compare against the public PyTorch formula
with a tolerance that includes this documented float32 step; the strict reference itself continues to enclose the
exact exponential ratios and weighted sums.

Fourteen contract examples cover routing order, dense zero weights, expert combination, four physical floating
dtypes, complete outputs, rejected ties and malformed inputs. Six deliberately wrong readings are rejected. A
separate 768-bit Arb evaluation checks the weights and combination without calling the adapter, and 20 public PyTorch
formula cases cover four dtypes and five shapes or routing patterns. Complete-output checks reject missing, extra,
wrong-shape and wrong-dtype results.

The previous draft combined four Switch and Mixtral profiles, including capacity and auxiliary-loss choices that
cannot be inferred from an unfamiliar kernel. It is retained under `contracts/draft` for history and is not active.
The reviewed contract makes no claim about other MoE routers, ties, capacity, expert dispatch, distributed execution,
expert MLP computation, load-balancing loss, special values, layout or alias effects.
