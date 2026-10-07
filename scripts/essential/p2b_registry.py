"""2b search space (docs/protocol_essential_bugs_phase2_20261007.md section 3; task book G1-G8): the family registry from which
the candidate list (G4, results/essential/phase2b/candidates.json) and the coverage plan (G3, coverage_plan.json) are written.

Search unit: contract x actual implementation path x input condition x call / state scenario x check method.
F and mode-B FR stay closed for every family until the reviewer's independent spec for it is delivered and merged (G8); E,
the P checks whose preconditions are stated here, and mode-A FR (K - K_R, Triton candidates) run.

    python scripts/essential/p2b_registry.py      # writes the two JSON files
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/essential/phase2b"

ENVS = {
    "ka_main": "torch 2.10.0+cu128, Triton 3.6.0 (TTIR mapping locked to this build)",
    "liger": "torch 2.10.0, transformers 4.57.3, accelerate 1.7.0, liger-kernel 0.7.0, unsloth 2026.3.3, torchao 0.16.0, "
             "bitsandbytes 0.49.2, xformers 0.0.35",
    "nightly_cuda": "torch 2.15.0.dev20260907+cu126, Triton 3.8.0 (newest CUDA nightly runnable on driver 535)",
    "nightly_cpu": "torch 2.15.0.dev20261005+cpu",
    "hf_ga_4452 / hf_ga_4460 / hf_ga_5190": "transformers 4.45.2 / 4.46.0 / 5.19.0 on torch 2.10",
}
UNAVAILABLE = {"deepspeed": "not installed (environment unavailable)", "apex": "not installed (environment unavailable)",
               "flash-attn (standalone)": "not installed (environment unavailable)",
               "megatron-core": "not installed (Megatron-style reference written in-repo for the MoE construction)"}


def cand(cid, lib, api, device, dtype, backend, env, ttir=False, shares=None, note=None):
    return {"id": cid, "library": lib, "api": api, "device": device, "dtype": dtype, "requested_backend": backend,
            "env": env, "ttir": ttir, "shares": shares or [], "actual_backend": "recorded at run time", "note": note}


def eager_inductor(api, dtypes=("float64", "float32", "bfloat16"), cpu=True, nightly=True):
    out = []
    for dt in dtypes:
        if cpu and dt != "bfloat16":
            out.append(cand(f"eager_cpu_{dt}", "torch 2.10", api, "cpu", dt, "aten", "ka_main"))
        out.append(cand(f"eager_cuda_{dt}", "torch 2.10", api, "cuda", dt, "aten", "ka_main"))
        if dt != "float64":
            out.append(cand(f"inductor_cuda_{dt}", "torch 2.10", f"torch.compile({api})", "cuda", dt, "inductor", "ka_main",
                            ttir=True, shares=["decomposition rules of torch 2.10"]))
    if nightly:
        out.append(cand("nightly_eager_cuda_float32", "torch nightly", api, "cuda", "float32", "aten", "nightly_cuda"))
        out.append(cand("nightly_inductor_cuda_float32", "torch nightly", f"torch.compile({api})", "cuda", "float32",
                        "inductor", "nightly_cuda", ttir=False, note="Triton 3.8.0: TTIR mapping not locked -> black box"))
    return out


# ------------------------------------------------------------------------------------------------ families (tier 1)

FAMILIES = {
    # ---------------------------------------------------------------- basic ops (low spec cost)
    "matmul_linear": {
        "group": "basic", "contract": "torch.mm/bmm/matmul/addmm/F.linear docs",
        "points": ["transpose and broadcast", "bias", "zero and one-element reduced dims", "dims not multiples of a tile",
                   "dX/dW/db", "gradient accumulation into .grad"],
        "factors": {"shape": ["(1,k)x(k,1)", "(m,0)x(0,n)", "(37,129)x(129,61)", "batch-broadcast"],
                    "layout": ["contiguous", "transposed", "offset"], "bias": [None, "vector", "broadcast"],
                    "values": ["gauss", "small_ints", "cancel", "mixed_scale"]},
        "high_risk": [{"shape": "(m,0)x(0,n)", "bias": "vector"}, {"layout": "transposed", "values": "small_ints"}],
        "candidates": eager_inductor("F.linear / torch.matmul"),
        "P": ["linearity in each argument (exact for small ints)", "transpose identity (A B)^T = B^T A^T",
              "gradient accumulation: two backward calls add (.grad semantics)"],
        "FR": "mode A (Inductor; matmul itself is cuBLAS/extern unless fused, so mostly not Triton)",
    },
    "reductions": {
        "group": "basic", "contract": "sum/mean/prod/amax/amin/var/std/softmax/log_softmax/logsumexp/cumsum docs",
        "points": ["multi-axis", "empty set", "effective count (correction for var/std)", "blocked merge",
                   "all -inf row (softmax / logsumexp)", "ties (amax/amin gradient)"],
        "factors": {"op": ["sum", "mean", "prod", "amax", "amin", "var", "std", "softmax", "log_softmax", "logsumexp", "cumsum"],
                    "dims": ["single", "multi", "all", "empty-extent"], "values": ["gauss", "small_ints", "all_equal", "neg_inf_row",
                                                                                    "zeros", "cancel"],
                    "size": ["1", "1027", "4099"]},
        "high_risk": [{"op": "logsumexp", "values": "neg_inf_row"}, {"op": "amax", "values": "small_ints"},
                      {"op": "var", "dims": "empty-extent"}],
        "candidates": eager_inductor("torch.<reduction>") + [cand("liger_softmax", "liger 0.7.0", "liger_kernel softmax", "cuda",
                                                                  "float32", "triton", "liger", ttir=True)],
        "P": ["shift invariance (softmax, log_softmax)", "logsumexp(x + c) = logsumexp(x) + c", "D1 for amax/amin ties",
              "cumsum last element = sum", "permutation invariance of full reductions (exact for small ints)"],
        "FR": "mode A (Inductor, Liger)",
    },
    "activations": {
        "group": "basic", "contract": "ReLU/SiLU/GELU (approximate='none' erf form, 'tanh' form)/SwiGLU/GeGLU docs",
        "points": ["threshold and branch at 0", "broadcast", "fused vs unfused forward and backward",
                   "GELU erf vs tanh declared formulas"],
        "factors": {"op": ["relu", "silu", "gelu_erf", "gelu_tanh", "swiglu", "geglu"], "values": ["gauss", "zeros", "huge", "tiny"],
                    "layout": ["contiguous", "offset"]},
        "high_risk": [{"op": "relu", "values": "zeros"}, {"op": "gelu_tanh", "values": "huge"}],
        "candidates": eager_inductor("F.<act>") + [cand("liger_swiglu", "liger 0.7.0", "LigerSiLUMulFunction", "cuda", "float32",
                                                        "triton", "liger", ttir=True),
                                                   cand("unsloth_swiglu", "unsloth 2026.3.3", "swiglu_fg_kernel", "cuda", "bfloat16",
                                                        "triton", "liger", ttir=True)],
        "P": ["relu derivative convention at 0 is 0 (documented subgradient)", "odd/even identities (gelu(x) - gelu(-x) = x)",
              "fused == composed (silu(a) * b)"],
        "FR": "mode A",
    },
    "gather_layout": {
        "group": "basic", "contract": "gather/index_select/take_along_dim/narrow/cat/split/transpose/reshape/view docs",
        "points": ["repeated indices (backward accumulation)", "non-contiguous layouts", "non-zero storage offset", "aliasing views"],
        "factors": {"op": ["gather", "index_select", "take_along_dim", "cat_split", "view_alias"],
                    "index": ["distinct", "repeated", "all_same"], "layout": ["contiguous", "transposed", "offset"]},
        "high_risk": [{"op": "gather", "index": "all_same", "layout": "offset"}],
        "candidates": eager_inductor("torch.gather / index_select / ..."),
        "P": ["counting: backward of a gather with repeated indices accumulates (ones -> counts)",
              "round trip split -> cat is the identity", "views share storage (write through)"],
        "FR": "mode A",
    },
    # ---------------------------------------------------------------- saving and recomputation
    "checkpoint": {
        "group": "recompute", "contract": "torch.utils.checkpoint docs (use_reentrant=False/True); dropout p = 0",
        "points": ["checkpoint on vs off: same forward value and gradients", "shared parameters", "an intermediate used twice"],
        "factors": {"impl": ["non_reentrant", "reentrant", "hf_gradient_checkpointing"], "graph": ["mlp", "shared_param", "reuse_twice"],
                    "compile": [False, True]},
        "high_risk": [{"impl": "reentrant", "graph": "shared_param"}, {"impl": "non_reentrant", "compile": True}],
        "candidates": [cand("torch_checkpoint_cpu", "torch 2.10", "torch.utils.checkpoint", "cpu", "float64", "aten", "ka_main"),
                       cand("torch_checkpoint_cuda", "torch 2.10", "torch.utils.checkpoint", "cuda", "float32", "aten", "ka_main"),
                       cand("torch_checkpoint_compiled", "torch 2.10", "compile(checkpoint)", "cuda", "float32", "inductor",
                            "ka_main", ttir=True),
                       cand("hf_gradient_checkpointing", "transformers 4.57.3", "model.gradient_checkpointing_enable", "cuda",
                            "float32", "aten", "liger")],
        "P": ["on == off (bitwise for float64 on CPU; tau otherwise)"], "FR": "not applicable (property check, no f needed)",
    },
    # ---------------------------------------------------------------- embeddings
    "embedding": {
        "group": "embedding", "contract": "F.embedding / F.embedding_bag docs",
        "points": ["padding_idx (no gradient)", "max_norm in-place renormalisation", "scale_grad_by_freq", "empty bag",
                   "mode sum/mean/max", "per_sample_weights"],
        "factors": {"op": ["embedding", "embedding_bag"], "padding_idx": [None, 0], "max_norm": [None, 1.0],
                    "scale_grad_by_freq": [False, True], "mode": ["sum", "mean", "max"], "bags": ["normal", "empty", "single"],
                    "per_sample_weights": [False, True]},
        "high_risk": [{"op": "embedding_bag", "bags": "empty", "mode": "mean"}, {"op": "embedding", "max_norm": 1.0, "padding_idx": 0}],
        "candidates": eager_inductor("F.embedding / F.embedding_bag", dtypes=("float64", "float32")),
        "P": ["padding_idx row gets zero gradient", "scale_grad_by_freq: gradient divided by the index frequency",
              "embedding_bag(mode=sum) == sum of embedding rows", "empty bag output is zero (sum/mean)"],
        "FR": "mode A",
    },
    # ---------------------------------------------------------------- attention
    "attention": {
        "group": "attention", "contract": "F.scaled_dot_product_attention docs (attn_mask, is_causal, scale, enable_gqa)",
        "points": ["causal alignment when q_len != k_len", "fully masked rows", "GQA", "softcap", "sliding window", "scale"],
        "factors": {"q_len_k_len": ["equal", "q<k", "q>k", "q=1"], "mask": ["none", "causal", "padding", "fully_masked_row",
                                                                             "sliding_window"],
                    "gqa": [False, True], "scale": [None, 0.5], "head_dim": [16, 64, 72]},
        "high_risk": [{"q_len_k_len": "q<k", "mask": "causal"}, {"mask": "fully_masked_row"}, {"q_len_k_len": "q=1", "mask": "causal"}],
        "candidates": [cand(f"sdpa_{b}_{dt}", "torch 2.10", "F.scaled_dot_product_attention", "cuda", dt, b, "ka_main")
                       for b in ("math", "efficient", "flash", "cudnn") for dt in ("float32", "bfloat16")
                       if not (b in ("flash", "cudnn") and dt == "float32")]
                      + [cand("sdpa_math_cpu_float64", "torch 2.10", "F.scaled_dot_product_attention", "cpu", "float64", "math", "ka_main"),
                         cand("inductor_attention_float32", "torch 2.10", "compile(manual softmax(QK^T)V)", "cuda", "float32",
                              "inductor", "ka_main", ttir=True),
                         cand("flex_attention_float32", "torch 2.10", "torch.nn.attention.flex_attention", "cuda", "float32",
                              "inductor (flex)", "ka_main", ttir=True),
                         cand("hf_eager_attention", "transformers 4.57.3", "LlamaAttention eager path", "cuda", "float32", "aten", "liger"),
                         cand("xformers_memory_efficient", "xformers 0.0.35", "memory_efficient_attention", "cuda", "bfloat16",
                              "cutlass", "liger")],
        "P": ["masked keys do not influence the output", "softmax rows sum to 1 (no fully masked row)",
              "causal: output at position i does not depend on keys > i (+ offset when q_len != k_len)",
              "GQA == repeated KV heads"],
        "FR": "mode A for Inductor / flex (Triton); SDPA backends are CUDA kernels -> black box",
    },
    "packing": {
        "group": "attention", "contract": "block-diagonal attention mask with position_ids reset per document",
        "points": ["no cross-document leakage", "position ids reset"],
        "factors": {"docs": [1, 2, 5], "lengths": ["equal", "unequal", "one_token_doc"], "impl": ["sdpa_mask", "flex_block_mask",
                                                                                                 "hf_packed_position_ids"]},
        "high_risk": [{"lengths": "one_token_doc"}],
        "candidates": [cand("sdpa_block_mask", "torch 2.10", "SDPA with block-diagonal mask", "cuda", "float32", "efficient", "ka_main"),
                       cand("flex_block_mask", "torch 2.10", "flex_attention with document mask", "cuda", "float32",
                            "inductor (flex)", "ka_main", ttir=True),
                       cand("hf_packed", "transformers 4.57.3", "Llama with packed position_ids (sdpa)", "cuda", "float32",
                            "sdpa", "liger")],
        "P": ["packed output of each document == the document run alone (no leakage)"], "FR": "mode A (flex)",
    },
    "rope": {
        "group": "attention", "contract": "RoPE: rotate-half (GPT-NeoX / HF Llama) and interleaved (GPT-J / original) conventions",
        "points": ["convention differences separated from real errors", "position offset", "scaling variants (linear, ntk)"],
        "factors": {"convention": ["rotate_half", "interleaved"], "offset": [0, 7], "scaling": [None, "linear2"],
                    "head_dim": [64, 72]},
        "high_risk": [{"convention": "interleaved", "offset": 7}],
        "candidates": [cand("ref_torch_rotate_half", "in-repo reference", "rotate-half formula", "cuda", "float32", "aten", "ka_main"),
                       cand("hf_apply_rotary", "transformers 4.57.3", "apply_rotary_pos_emb", "cuda", "float32", "aten", "liger"),
                       cand("liger_rope", "liger 0.7.0", "liger_rotary_pos_emb", "cuda", "float32", "triton", "liger", ttir=True),
                       cand("unsloth_rope", "unsloth 2026.3.3", "fast_rope_embedding", "cuda", "float32", "triton", "liger", ttir=True)],
        "P": ["norm preservation per pair", "relative position: <R(m) q, R(n) k> depends on m - n only", "R(0) = identity"],
        "FR": "mode A (Liger, Unsloth)",
    },
    # ---------------------------------------------------------------- normalisation
    "normalization": {
        "group": "norm", "contract": "LayerNorm / RMSNorm / GroupNorm / BatchNorm docs (eps inside the sqrt; BN training uses the "
                                     "biased batch variance, running_var the unbiased one)",
        "points": ["eps position", "variance definition", "BN running statistics", "BN eval vs train"],
        "factors": {"op": ["layer_norm", "rms_norm", "group_norm", "batch_norm_train", "batch_norm_eval"],
                    "values": ["gauss", "constant_rows", "huge_offset", "tiny"], "affine": [True, False], "batch": [1, 2, 7]},
        "high_risk": [{"op": "batch_norm_train", "batch": 1}, {"op": "rms_norm", "values": "constant_rows"}],
        "candidates": eager_inductor("F.<norm>") + [cand("liger_rmsnorm", "liger 0.7.0", "LigerRMSNorm", "cuda", "float32", "triton",
                                                         "liger", ttir=True),
                                                    cand("liger_layernorm", "liger 0.7.0", "LigerLayerNorm", "cuda", "float32",
                                                         "triton", "liger", ttir=True)],
        "P": ["scale invariance (norm(a x) = norm(x) for a > 0, eps -> 0 limit)", "BN running_var uses n / (n - 1)",
              "BN eval uses running statistics only"],
        "FR": "mode A",
    },
    # ---------------------------------------------------------------- optimisers (state)
    "optimizers": {
        "group": "optimizer", "contract": "torch.optim AdamW/Adam/SGD/RMSprop docs (algorithm boxes); Adafactor (paper / implementation docs)",
        "points": ["weight decay position (decoupled vs L2)", "amsgrad max over v (not v_hat)", "maximize", "eps position",
                   "nesterov / dampening", "RMSprop centered", "Adafactor factorisation and clipping threshold",
                   "parameter-group differences"],
        "factors": {"opt": ["adamw", "adam", "adam_amsgrad", "sgd_nesterov", "sgd_dampening", "rmsprop_centered", "adafactor"],
                    "impl": ["for_loop", "foreach", "fused"], "maximize": [False, True], "weight_decay": [0.0, 0.1],
                    "state": ["cold", "warm", "zero_grad", "grad_none", "nonfinite_skip", "save_restore"]},
        "high_risk": [{"opt": "adam_amsgrad", "state": "save_restore"}, {"opt": "adamw", "state": "grad_none"},
                      {"opt": "sgd_nesterov", "maximize": True}],
        "candidates": [cand(f"torch_{impl}_{dev}", "torch 2.10", f"torch.optim.<opt>({impl})", dev, "float32", impl, "ka_main")
                       for impl in ("for_loop", "foreach", "fused") for dev in ("cpu", "cuda")]
                      + [cand("torch_compiled_step", "torch 2.10", "torch.compile(opt.step)", "cuda", "float32", "inductor",
                              "ka_main", ttir=True),
                         cand("bnb_adamw32bit", "bitsandbytes 0.49.2", "bnb.optim.AdamW32bit", "cuda", "float32", "cuda", "liger"),
                         cand("torchao_adamw_fp32", "torchao 0.16.0", "torchao.optim._AdamW (fp32 state)", "cuda", "float32",
                              "compiled", "liger"),
                         cand("hf_adafactor", "transformers 4.57.3", "transformers.optimization.Adafactor", "cuda", "float32",
                              "aten", "liger"),
                         cand("deepspeed_fused_adam", "deepspeed", "FusedAdam", "cuda", "float32", "-", "unavailable")],
        "P": ["for_loop == foreach == fused (same contract)", "maximize(g) == minimize(-g)", "zero grad vs grad None: documented "
              "difference (None skips the parameter)", "save/restore mid-run continues identically", "amsgrad: max_exp_avg_sq "
              "non-decreasing"],
        "FR": "mode A (compiled step)",
        "state_sequences": ["init -> normal -> zero-grad -> restore", "init -> steps -> save/restore -> continue",
                            "init -> skipped (non-finite) -> next normal"],
    },
    "schedulers": {
        "group": "optimizer", "contract": "torch.optim.lr_scheduler docs (closed forms), transformers get_cosine_schedule_with_warmup",
        "points": ["closed form vs recursion", "last_epoch / resume", "phase boundaries (warmup end, cycle end)"],
        "factors": {"sched": ["warmup_cosine_hf", "cosine_annealing", "one_cycle", "linear_warmup_torch"],
                    "resume": [False, True], "steps": [1, 10, 1000]},
        "high_risk": [{"sched": "cosine_annealing", "resume": True}, {"sched": "one_cycle", "steps": 1}],
        "candidates": [cand("torch_lr_scheduler", "torch 2.10", "lr_scheduler.*", "cpu", "float64", "python", "ka_main"),
                       cand("torch_nightly_lr_scheduler", "torch nightly", "lr_scheduler.*", "cpu", "float64", "python", "nightly_cpu"),
                       cand("hf_schedulers", "transformers 4.57.3", "get_*_schedule_with_warmup", "cpu", "float64", "python", "liger")],
        "P": ["recursive (step) == closed form (get_closed_form / documented formula)", "resume at k == continuous run",
              "values at phase boundaries"],
        "FR": "not applicable",
    },
    "clip_amp": {
        "group": "optimizer", "contract": "clip_grad_norm_ / clip_grad_value_ / GradScaler docs",
        "points": ["norm_type (2, inf, 1)", "non-finite handling (error_if_nonfinite)", "unscale before clip", "skipped steps",
                   "merging sharded norms (sum of squares, not sum of norms)"],
        "factors": {"op": ["clip_norm", "clip_value", "scaler_step"], "norm_type": [2.0, "inf", 1.0], "grads": ["normal", "with_nan",
                                                                                                                 "with_inf", "zero"],
                    "foreach": [False, True], "shards": [1, 2]},
        "high_risk": [{"op": "clip_norm", "grads": "with_inf", "norm_type": "inf"}, {"op": "clip_norm", "shards": 2}],
        "candidates": [cand(f"torch_clip_{dev}", "torch 2.10", "nn.utils.clip_grad_norm_", dev, "float32", "aten", "ka_main")
                       for dev in ("cpu", "cuda")]
                      + [cand("torch_gradscaler_cuda", "torch 2.10", "torch.amp.GradScaler", "cuda", "float16", "aten", "ka_main"),
                         cand("accelerate_clip", "accelerate 1.7.0", "Accelerator.clip_grad_norm_", "cuda", "float32", "aten", "liger"),
                         cand("deepspeed_clip", "deepspeed", "engine clipping", "cuda", "float32", "-", "unavailable")],
        "P": ["clip(c) leaves norms <= c unchanged and scales others to c", "shard merge: sqrt(sum of squares) == global norm",
              "a non-finite step is skipped and the scale halves (GradScaler)"],
        "FR": "not applicable",
    },
    # ---------------------------------------------------------------- training-program layer
    "training_program": {
        "group": "program", "contract": "label shift and token counting; accumulation-window normalisation (specs/phase1/"
                                        "spec_accumulation.py); loss averaging across ranks",
        "points": ["mean over unequal token counts", "label shift", "cross-rank averaging (2 gloo processes on CPU)"],
        "factors": {"impl": ["hf_trainer_4452", "hf_trainer_4460", "hf_trainer_4573", "hf_trainer_5190", "accelerate_loop", "own_loop"],
                    "tokens": ["equal", "unequal", "empty_microbatch"], "ranks": [1, 2]},
        "high_risk": [{"tokens": "unequal", "ranks": 2}, {"tokens": "empty_microbatch"}],
        "candidates": [cand(f"hf_trainer_{v}", f"transformers {v}", "Trainer (gradient_accumulation_steps)", "cpu", "float64",
                            "aten", env) for v, env in (("4.45.2", "hf_ga_4452"), ("4.46.0", "hf_ga_4460"), ("4.57.3", "liger"),
                                                       ("5.19.0", "hf_ga_5190"))]
                      + [cand("accelerate_ddp_gloo", "accelerate 1.7.0", "Accelerator + 2-process gloo", "cpu", "float64", "gloo",
                              "liger"),
                         cand("torch_ddp_gloo", "torch 2.10", "DistributedDataParallel (2 gloo processes)", "cpu", "float64",
                              "gloo", "ka_main")],
        "P": ["split invariance", "padding-free == padded (TRL-style invariance)", "rank invariance: 2 ranks == 1 rank on the "
              "concatenated batch"],
        "FR": "not applicable", "F_spec_available": "spec_accumulation.py (phase-1 package) covers the accumulation window",
    },
    # ---------------------------------------------------------------- MoE (small construction)
    "moe": {
        "group": "moe", "contract": "top-k routing with renormalised weights; permute / unpermute; capacity drop; auxiliary "
                                    "load-balancing loss (Switch / Megatron-style definitions)",
        "points": ["top-k routing ties", "permute / unpermute round trip", "empty expert", "capacity boundary", "weighted combine",
                   "aux-loss counting"],
        "factors": {"topk": [1, 2], "experts": [4, 8], "capacity": [None, "exact", "overflow"], "routing": ["random", "ties", "one_expert"]},
        "high_risk": [{"routing": "one_expert", "capacity": "overflow"}, {"routing": "ties", "topk": 2}],
        "candidates": [cand("ref_loop_cpu", "in-repo reference", "per-token loop", "cpu", "float64", "python", "ka_main"),
                       cand("vectorised_scatter_cuda", "in-repo (Megatron-style)", "sort-based permute + index_add", "cuda",
                            "float32", "aten", "ka_main"),
                       cand("vectorised_compiled", "in-repo (Megatron-style)", "torch.compile(...)", "cuda", "float32", "inductor",
                            "ka_main", ttir=True)],
        "P": ["permute -> unpermute is the identity", "combine weights per token sum to 1", "dropped tokens contribute zero",
              "aux loss equals its documented formula on counts"],
        "FR": "mode A (compiled)",
    },
}

TIER2 = {"queued": ["KLDiv batchmean", "BCEWithLogits pos_weight", "CTC", "adaptive pooling bin edges", "interpolate align_corners",
                    "conv same padding"],
         "registered_subset_or_deferred": {"quantised-training scale rounding": "deferred to the extension queue",
                                           "selective-scan chunk boundaries and state reset": "deferred to the extension queue"},
         "not_checked": ["dropout randomness (only the 1/(1-p) scaling under a fixed mask)", "autocast dtype policy",
                         "inference-time KV-cache quantisation"]}


def coverage_plan(fam):
    """single-factor boundaries (each level of each factor with the others at their first level), all pairwise
    combinations of factor levels for the first two factors, and the registered high-risk combinations."""
    f = fam["factors"]
    keys = list(f)
    base = {k: v[0] for k, v in f.items()}
    plan = []
    for k in keys:
        for v in f[k]:
            plan.append(dict(base, **{k: v}))
    if len(keys) >= 2:
        for a, b in itertools.product(f[keys[0]], f[keys[1]]):
            plan.append(dict(base, **{keys[0]: a, keys[1]: b}))
    for hr in fam.get("high_risk", []):
        plan.append(dict(base, **hr, high_risk=True))
    uniq, seen = [], set()
    for c in plan:
        key = json.dumps(c, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cands = {name: fam["candidates"] for name, fam in FAMILIES.items()}
    (OUT / "candidates.json").write_text(json.dumps({"environments": ENVS, "unavailable": UNAVAILABLE, "families": cands},
                                                    indent=1, default=str) + "\n")
    plan = {}
    for name, fam in FAMILIES.items():
        cp = coverage_plan(fam)
        plan[name] = {"group": fam["group"], "contract": fam["contract"], "points": fam["points"], "factors": fam["factors"],
                      "planned_conditions": len(cp), "conditions": cp,
                      "methods": {"E": "yes", "P": fam["P"], "FR": fam["FR"],
                                  "F": fam.get("F_spec_available", "closed until the reviewer's spec is delivered and merged")},
                      "state_sequences": fam.get("state_sequences")}
    (OUT / "coverage_plan.json").write_text(json.dumps({"tier1": plan, "tier2": TIER2}, indent=1, default=str) + "\n")
    for name, p in plan.items():
        print(f"{name:18s} conditions {p['planned_conditions']:4d}  candidates {len(cands[name]):3d}")


if __name__ == "__main__":
    main()
