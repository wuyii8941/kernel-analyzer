# ATTN: ATen-call interface for the fused SDPA overloads (path A closure, package B, 2026-10-20/21)

Draft files: `contracts/draft/ATTN_path_a_A2_20261020.yaml`, `contracts/draft/ATTN_adapter_path_a_A2_20261020.py`
(helpers `kernel_spec/attn_aten_semantics.py`, mutant-only `kernel_spec/attn_aten_indep.py`; tests
`tests/test_attn_aten_interface.py`). The top-level `contracts/ATTN.yaml` / `adapters/ATTN.py` are unchanged.

## What was added

Profile `torch210_aten_sdpa_fused_dropout0` (`task_choice_required: false`, output `out` = list of ATen return values):

| overload | recorded call (blocked_keys.txt) | positions | not compared |
|---|---|---|---|
| `aten._scaled_dot_product_flash_attention` | qwen bf16, q [1,14,128,64] strided view, k/v [1,2,128,64], dropout 0.0, causal, scale 0.125 | 9 | 2 cum_seq_q, 3 cum_seq_k, 6 rng_state, 7 unused, 8 debug_attn_mask (AT.ATEN_AUX) |
| `aten._scaled_dot_product_flash_attention_backward` | qwen bf16, same GQA shapes, lse f32 [1,14,128], cum_seq None, max 128/128, dropout 0.0, causal, seed i64[2], offset i64[] | 3 | none |
| `aten._scaled_dot_product_efficient_attention_backward` | bert-tiny / gpt2 f32 [1,H,128,64], attn_bias None, **dropout 0.1**, mask [T,T,T,F], causal False/True | 4 | none (position 3 is null because attn_bias is None, a per-call null) |

`ATEN_RETURNS` in the adapter states the same for the rule-v2 generator.

Decisions (all `drafted`): AT.ATEN_TASK (output = public SDPA result; logsumexp = natural log of the softmax
normaliser of the scaled, masked scores, f32 (B,Hq,L)), AT.ATEN_HEADS (flash: contiguous GQA groups; efficient: equal
heads), AT.ATEN_CAUSAL (j <= i, only L = S), AT.ATEN_SAVED (backward uses supplied out and logsumexp; exact VJP kept as
reading `recompute_saved`), AT.ATEN_AUX (return positions), AT.ATEN_RNG (dropout_p > 0 unsupported), AT.ATEN_DOMAIN.

Consequence for the recorded keys: the two qwen flash keys are inside the profile; the two efficient-backward keys
(bert-tiny#73, gpt2#87) have dropout_p = 0.1 and receive `unsupported` (AT.ATEN_RNG). They are therefore expected to
end as "outside contract scope" in A2, with the stated reason, not as a semantic verdict.

## Sources read

- `TS_NATIVE_FUNCTIONS.SDPA_FUSED_SCHEMAS` (schemas, return names, `nondeterministic_seeded` tags).
- `TS_AUTOGRAD_DERIVATIVES.SDPA_FUSED`: the backward ops receive the forward's own `output` and `logsumexp`.
- `TS_META_SDPA` (new, torch/_meta_registrations.py): flash returns (attention, lse f32 (B,H,L), None, None,
  L, S, seed (2,), offset (), empty debug mask); efficient forward pads lse to ceil(L/32)*32; backward shapes.
- `FA2_PAPER` (new, arXiv 2307.08691v1 HTML): L = m + log(l); Algorithm 2 uses supplied O and L
  (D = rowsum(dO o O), P = exp(S - L)); GQA gradients summed over the duplicated heads.
- `TS_ATTN_CU` (new, reference program): flash CUDA returns undefined cum_seq tensors; efficient maps is_causal to
  CausalFromTopLeft.
- `TS_MEMEFF_FWD_H` (new, reference program) and `T_RANDOMNESS` (new): see RNG below.

## RNG question (efficient backward with dropout_p = 0.1)

Searched: the SDPA page (T_SDPA: "dropout applied", "pass 0.0 to disable"), torch.nn.functional.dropout (Bernoulli(p)
only), the 2.10 reproducibility note ("not guaranteed across releases ... or platforms ... even when using identical
seeds"), native_functions.yaml (schema only, tagged nondeterministic_seeded), derivatives.yaml, meta registrations.
None defines the mapping from (philox_seed, philox_offset) to the dropout mask. The mapping exists only in the CUDA
kernel source: `curand_init(seed, 0, offset + batch/head linear offset)` on a cuRAND Philox4_32_10 state, one stream
element per (b, h, i, j) entry, keep with probability 1-p and scale 1/(1-p), plus cuRAND's internal uniform
conversion; flash uses a different layout. Defining f from that would define the task by one implementation's code.
Decision AT.ATEN_RNG therefore refuses dropout_p > 0 (unsupported); alternatives (kernel-emulating task-choice profile,
distributional reading) are listed and not implemented. No RNG model is assumed.

## PyTorch 2.10 cross-check (A6000 sm_86, torch 2.10.0+cu128; scripts in `.cache/attn_aten/probe*.py`)

Flash forward, recorded structure (bf16, q a strided [1,128,14,64] view transposed to [1,14,128,64], stride
(114688,64,896,1), k/v [1,2,128,64], dropout 0, causal, scale 0.125), against an independent float64 formula:

- The flash ATen op accepts 14 query / 2 key-value heads directly; the public SDPA with `enable_gqa=True` dispatches
  to exactly `aten._scaled_dot_product_flash_attention.default` and its backward (TorchDispatchMode trace).
- Returns: output bf16 [1,14,128,64]; logsumexp f32 [1,14,128]; cum_seq_q/k None; max_q = max_k = 128 (int);
  rng_state uint64 [2] = [0,0]; unused uint64 [] = 0; debug mask bf16 shape (0,).
- max |lse - natural LSE| = 6.7e-7; against LSE/ln 2 = 2.56 (base 2 rejected). max |out - ref| = 8.5e-3 (bf16).
- Contiguous GQA mapping: round-robin mapping differs by 3.55 (rejected).
- Backward vs float64 autograd of public SDPA (enable_gqa): grad_q/k/v max error 0.010 / 0.025 / 0.050 at
  magnitudes 2.6 / 5.7 / 9.3 (bf16); grad_key / grad_value have the kv shapes [1,2,128,64].
- Supplied values are used: logsumexp + 1 scales grad_value by 0.3677 (exp(-1) = 0.3679); supplying out = 0 changes
  grad_query (by 4.97) but not grad_value. This supports AT.ATEN_SAVED (it is not its definition).
- Refusals: f32 ("FlashAttention only support fp16 and bf16"); Hq=3, Hkv=2 ("must divide"); return_debug_mask=True
  with dropout 0 ("return_softmax is only supported when p_dropout > 0.0"). scale=None equals scale=1/8 for E=64.
- Non-square causal (L=4, S=9): flash ATen output matches bottom-right (diff 4.3e-3) not upper-left (3.6); the
  public dispatcher refuses flash for that case. Efficient non-square causal matches upper-left (2.4e-7). Hence
  AT.ATEN_CAUSAL restricts causal calls to L = S.
- Flash backward ignores max_q/max_k values for dense inputs (7/9 gave identical gradients); the profile still
  requires max_q = L, max_k = S (forward's returned values) and leaves other values unsupported.

Efficient forward/backward (f32, [1,2,128,64] non-causal and [1,12,128,64] causal, dropout 0):

- lse f32 [1,H,128]; seed/offset int64 scalars (0, 0). out error 1.3e-6, lse error 9.8e-7 vs float64.
- Backward returns (grad_q, grad_k, grad_v, None); errors vs float64 autograd ~1.7e-6. logsumexp + 1 scales
  grad_value by exp(-1) (supplied value used).
- grad_input_mask [F,T,F,F] still returns all three gradients unchanged (mask entries 0..2 do not change the result;
  the profile requires them true). grad_input_mask[3] = True with attn_bias None crashes the process (segfault):
  unsupported in the profile.
- Grouped heads rejected ("Expected query.size(2) == key.size(2)"): unsupported.
- L = 5: lse has shape [1,2,32] with +inf padding; unpadded lse rejected ("LSE is not correctly aligned"); NaN in
  the padding produces NaN grad_value, +inf padding contributes 0. The profile accepts width L or ceil(L/32)*32 with
  +inf padding only.
- dropout 0.1: same seed reproduces, a different seed changes the output by 0.59; the backward with another seed /
  offset changes grad_value by 0.51: the result depends on the RNG inputs.

Reference vs PyTorch (tests): reduced recorded structures (Hq=14/Hkv=2, causal, scale 0.125) match independent
float64 formulas to 1e-12 (forward) / 1e-11 (supplied-value backward); the `recompute_saved` reading matches float64
PyTorch autograd to 1e-10; the CUDA flash kernel agrees with the reference within bf16 tolerance (out 2e-2, lse 1e-5,
gradients 6e-2) on an 8x8 GQA case.

## Checks

- `tools/check_contracts.py --contracts .cache/spec_a2_stage/B/contracts --adapters .cache/spec_a2_stage/B/adapters
  --only ATTN`: consistent, examples 40/40, readings 2/2, mutations 18/18 (new AT.M9-AT.M18 each rejected by its
  witness); after the review follow-up 42/42, 2/2, 19/19.
- `pytest -q 5_specification/tests/test_attn_aten_interface.py`: 15 passed.

## Review follow-up (2026-10-21, reviews/2026-10-20/A2_package_B_review.md)

- Flash forward/backward now refuse a value head dimension Ev != E (invalid_input, AT.ATEN_DOMAIN, example AT.E41,
  mutant AT.M19): TS_META_SDPA.FLASH_FWD gives attention = empty_like(query) and PyTorch refuses Ev != E. The
  efficient backward keeps Ev != E (AT.E42).
- AT.ATEN_DOMAIN: domain limited by dtype only (backend head-dimension alignment is not part of it); the efficient
  backward deliberately also accepts the unpadded logsumexp width L (PyTorch's CUDA kernel rejects it unless L is a
  multiple of 32; alternative stated).
- AT.ATEN_RNG: added the alternative "condition on an observed mask", impossible because no covered overload
  returns or receives the mask (the backward regenerates it from philox seed/offset).
- Re-check on a fresh stage (`.cache/spec_a2_stage/B2`): consistent, examples 42/42, readings 2/2, mutations 19/19.
  Tests: `tests/test_attn_aten_interface.py` (16) and reviewer `tests/test_attn_aten_a2_review.py` (49, strict xfail
  removed after XPASS): 65 passed.

## Limits

- Full-size recorded calls (14 x 128 x 128 x 64) are evaluated element-wise with exact rationals; speed is a bulk
  backend question (kernel_spec/bulk.py), not addressed here.
- attn_bias, packed variable-length inputs, non-square causal calls and dropout are outside the profile.
