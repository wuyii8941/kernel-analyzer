# [Bug] Released Liger (<= 0.8.4) silently computes wrong RoPE for Phi-4-mini (phi3 with partial_rotary_factor 0.75)

## Summary

Phi-4-mini-instruct has `model_type = "phi3"`, `head_dim = 128`, `partial_rotary_factor = 0.75`: HF's
`modeling_phi3.apply_rotary_pos_emb` rotates the first `rotary_dim = 96` dims (pairs `(i, i + 48)`) and passes the
last 32 through. `apply_liger_kernel_to_phi3(rope=True)` (the default, also reached through HF Trainer / TRL
`use_liger_kernel=True` and `AutoLigerKernelForCausalLM`) replaces it with the Triton RoPE kernel, which in every
release up to 0.8.4 assumes `rotary_dim == head_dim`: it rotates the whole head with pairs `(i, i + 64)`, reading the
96-wide cos/sin as if they were 128 wide. No error or warning; GLM-4, Qwen3-Next, Qwen3.5 and Nemotron are guarded
for exactly this reason, phi3 is not.

Main is fixed by #1451, which presents the change as new support for Qwen3-Next / GLM-4 / GPT-NeoX / Phi-2 /
StableLM and does not mention that phi3 checkpoints with partial rotary were silently wrong in releases.

## Measurement

Phi-4-mini-instruct, bf16, wikitext-103 validation (16 x 2 x 1024 tokens), only `rope=True` applied to the loaded
instance:

| | val loss |
|---|---|
| HF | 2.700 |
| Liger 0.8.4 (transformers 5.18.0) | 12.928 |
| Liger 0.7.0 (transformers 4.57.3) | 12.928 |
| Liger main (b297821787) | 2.700 |

Relative error of the RoPE output for q on one layer: 0.84; on the 32 pass-through dims: 0.91.

## Requests

1. A patch release containing #1451's partial-RoPE support (but see the separate GPT-OSS regression report, which the
   same change introduces), or a guard that raises for phi3 configs with `partial_rotary_factor < 1` in released
   branches.
2. A regression test with a phi3 mini model using `partial_rotary_factor=0.75`; the current `mini_phi3` convergence
   config uses the default full rotary, so it cannot catch this.
