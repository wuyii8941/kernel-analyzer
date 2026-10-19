# vLLM 0.16.0 deterministic top-k then top-p filter review — 2026-10-19

The active SAMPLING contract covers one narrow deterministic task from the installed vLLM 0.16.0
`apply_top_k_top_p` function. It receives complete finite float32 logits with distinct values in every row and
complete per-row `k` and `p` parameters. It first keeps the `k` largest logits, then applies top-p to a softmax
renormalised over that remainder, and returns the complete Boolean keep mask. The contract does not return masked
logits, probabilities or a randomly sampled token.

The retained nucleus-sampling paper defines top-p as the smallest highest-probability set whose cumulative mass
reaches `p`, and top-k as the size-`k` set with maximum probability mass. The retained installed vLLM source fixes the
implementation choices needed here: ascending sort, top-k before top-p, softmax after the top-k mask, removal while
ascending cumulative mass is at most `1-p`, and unconditional retention of the largest entry. On distinct logits,
this equals keeping a token exactly when the exact-real mass at larger retained logits is less than `p`.

The old draft mixed this deterministic filter with masked logits, output probabilities, exponential random noise and
a sampled token, and cited an unverified development commit. It remains under `contracts/draft` and is not active.
Equal-logit ordering is also refused instead of assigning an undocumented tie result. The active specification uses
exact-real softmax for the task; float32 exponent, accumulation and comparison behavior remain measurable execution
properties.

Twenty contract examples cover top-k alone, top-p alone, their composition, row-specific parameters, `p=1`, complete
Boolean output and refused inputs. Six deliberately wrong readings are rejected. A separate 768-bit Arb computation
checks all 20 deterministic cases without calling the adapter, and a public PyTorch formula agrees on all cases.
The installed vLLM 0.16.0 `apply_top_k_top_p` function was called on the same 20 cases and returned the same complete
mask every time. Complete-output checks reject missing, extra, wrong-shape and wrong-dtype results.

This profile does not certify ties, special values, other dtypes, an identity call with no filter, masked-logit
materialisation, probabilities, random sampling, per-request generators, FlashInfer or AITER alternatives,
gradients, layout or alias effects.
