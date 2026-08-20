#!/usr/bin/env python3
"""Causally split the layer-23 query cotangent at actual AOT bmm_76.

The target program is G_q = S_bwd @ K.  Reference/candidate operands are
crossed at the exact generated bmm call, while the real downstream query
RoPE/RMSNorm VJP and q_proj weight VJP continue unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD_SRC = REPO / "archive" / "round1_code" / "src"
for path in (OLD_SRC, REPO, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.long_horizon_trigger import atomic_json, build_model, load_eval_states, load_milestone, under_root
from kernel_analyzer.seup import adamw_update


PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280)
COLUMNS = slice(1664, 1792)
TARGET_NODE_MARKER = "bmm_76]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--state-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/final/l23_attention_bmm_decomposition.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--antithetic-only", action="store_true",
        help="Run only the fixed-state S_bwd +epsilon/-epsilon matched intervention.",
    )
    return parser.parse_args()


def projection(value, direction, torch) -> float:
    return float(torch.dot(value.reshape(-1).float(), direction.reshape(-1).float()) / direction.float().norm())


def main() -> None:
    args = parse_args()
    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    direction_path = under_root(args.direction, "direction")
    output_path = under_root(args.output, "output")
    state_indices = args.state_index or list(range(8, 24 if args.antithetic_only else 40))
    if min(state_indices) < 8 or len(set(state_indices)) != len(state_indices):
        raise ValueError("requires unique held-out state indices >= 8")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/tile_causal_compile")

    import torch
    import torch.nn.functional as F
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.select_algorithm import extern_kernels
    from transformers import AutoTokenizer
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    bank = json.loads(bank_path.read_text())
    milestone = next(row for row in bank["milestones"] if int(row["step"]) == args.step)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    all_states, evaluation = load_eval_states(tokenizer, 1024, max(state_indices) + 1, device)
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)

    direction_payload = torch.load(direction_path, map_location="cpu", weights_only=False)
    if direction_payload.get("schema") != "kernel-analyzer-frozen-tile-direction-v1":
        raise ValueError("direction schema mismatch")
    direction = direction_payload["direction"].float().to(device)

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__()
            self.subject = subject

        def forward(self, input_ids, labels):
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm_loss = candidate(*all_states[state_indices[0]])
    warm_loss.backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])
    source_matches = []
    for module in modules:
        source_path = Path(module.__file__)
        source = source_path.read_text()
        if TARGET_NODE_MARKER in source and "mm_267" in source:
            source_matches.append((module, source_path, source))
    if len(source_matches) != 1:
        raise RuntimeError(f"expected one exact bmm_76 backward source, got {len(source_matches)}")
    source_module, source_path, source = source_matches[0]
    marker_position = source.index(TARGET_NODE_MARKER)
    call_start = source.rfind("def call(", 0, marker_position)
    target_ordinal = source[call_start:marker_position].count("extern_kernels.bmm(")
    d_mm_marker = source.index("mm_262]", call_start, marker_position)
    target_d_mm_ordinal = source[call_start:d_mm_marker].count("extern_kernels.mm(")
    softmax_calls = list(re.finditer(
        r"([A-Za-z0-9_]*softmax[A-Za-z0-9_]*backward[A-Za-z0-9_]*)\.run\(",
        source[call_start:marker_position],
    ))
    if not softmax_calls:
        raise RuntimeError("softmax-backward kernel preceding bmm_76 was not found")
    target_softmax_symbol = softmax_calls[-1].group(1)
    target_softmax_position = call_start + softmax_calls[-1].start()
    target_softmax_ordinal = source[call_start:target_softmax_position].count(
        f"{target_softmax_symbol}.run("
    )
    target_softmax_kernel = getattr(source_module, target_softmax_symbol)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    original_attention = modeling_qwen3.eager_attention_forward
    eager_capture: dict[str, torch.Tensor] = {}

    def captured_eager_attention(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
        raw_scores = torch.matmul(query, key_states.transpose(2, 3))
        if module.layer_idx == 23:
            eager_capture["K"] = key_states.detach().reshape(16, 1024, 128).clone()
            eager_capture["V"] = value_states.detach().reshape(16, 1024, 128).clone()

            def raw_hook(gradient):
                eager_capture["S"] = gradient.detach().reshape(16, 1024, 1024).clone()
                return gradient

            def query_hook(gradient):
                eager_capture["Gq"] = gradient.detach().reshape(16, 1024, 128).clone()
                return gradient

            raw_scores.register_hook(raw_hook)
            query.register_hook(query_hook)
        attn_weights = raw_scores * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        probability_saved_fp32 = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = probability_saved_fp32.to(query.dtype)
        if module.layer_idx == 23:
            eager_capture["P"] = probability_saved_fp32.detach().reshape(16, 1024, 1024).clone()

            def probability_hook(gradient):
                eager_capture["U"] = gradient.detach().reshape(16, 1024, 1024).clone()
                return gradient

            attn_weights.register_hook(probability_hook)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if module.layer_idx == 23:
            def attention_output_hook(gradient):
                eager_capture["D"] = gradient.detach().reshape(16, 1024, 128).clone()
                return gradient

            attn_output.register_hook(attention_output_hook)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    q_proj_parameter = dict(model.named_parameters())[PARAMETER]
    original_bmm = extern_kernels.bmm
    original_mm = extern_kernels.mm

    def run_candidate(
        inputs,
        replacement=None,
        capture_operands=False,
        softmax_u_replacement=None,
        capture_softmax=False,
        u_bmm_replacement=None,
        capture_u_bmm=False,
        d_mm_left_replacement=None,
        capture_d_mm=False,
    ):
        model.zero_grad(set_to_none=True)
        loss = candidate(*inputs)
        counter = {"bmm": 0}
        mm_counter = {"mm": 0}
        softmax_counter = {"call": 0}
        observed = {}
        original_softmax_run = target_softmax_kernel.run

        def wrapped_softmax_run(*values, **kwargs):
            ordinal = softmax_counter["call"]
            softmax_counter["call"] += 1
            if ordinal != target_softmax_ordinal:
                return original_softmax_run(*values, **kwargs)
            if len(values) < 4 or tuple(values[1].shape) != (16, 1024, 1024):
                raise RuntimeError("exact layer-23 softmax backward has unexpected operands")
            observed["softmax_target"] = True
            if capture_softmax:
                observed["U"] = values[1].detach().clone()
            if softmax_u_replacement is None:
                return original_softmax_run(*values, **kwargs)
            replaced = list(values)
            replaced[1] = softmax_u_replacement
            return original_softmax_run(*replaced, **kwargs)

        def wrapped_bmm(*values, **kwargs):
            ordinal = counter["bmm"]
            counter["bmm"] += 1
            if ordinal == target_ordinal - 2:
                left, right = values[:2]
                out = kwargs.get("out")
                expected = ((16, 1024, 128), (16, 128, 1024), (16, 1024, 1024))
                if (tuple(left.shape), tuple(right.shape), tuple(out.shape)) != expected:
                    raise RuntimeError("exact bmm_74 ordinal has unexpected operands")
                observed["u_bmm_target"] = True
                if capture_u_bmm:
                    observed["D"] = left.detach().clone()
                    observed["Vt"] = right.detach().clone()
                selected_left, selected_right = (
                    u_bmm_replacement if u_bmm_replacement is not None else (left, right)
                )
                return original_bmm(selected_left, selected_right, out=out)
            if ordinal != target_ordinal:
                return original_bmm(*values, **kwargs)
            left, right = values[:2]
            out = kwargs.get("out")
            expected = ((16, 1024, 1024), (16, 1024, 128), (16, 1024, 128))
            if (tuple(left.shape), tuple(right.shape), tuple(out.shape)) != expected:
                raise RuntimeError("exact bmm_76 ordinal has unexpected operands")
            observed["target"] = True
            if capture_operands:
                observed["S"] = left.detach().clone()
                observed["K"] = right.detach().clone()
            selected_left, selected_right = replacement if replacement is not None else (left, right)
            return original_bmm(selected_left, selected_right, out=out)

        def wrapped_mm(*values, **kwargs):
            ordinal = mm_counter["mm"]
            mm_counter["mm"] += 1
            if ordinal != target_d_mm_ordinal:
                return original_mm(*values, **kwargs)
            left, right = values[:2]
            out = kwargs.get("out")
            expected = ((1024, 2048), (2048, 2048), (1024, 2048))
            if (tuple(left.shape), tuple(right.shape), tuple(out.shape)) != expected:
                raise RuntimeError("exact mm_262 ordinal has unexpected operands")
            observed["d_mm_target"] = True
            if capture_d_mm:
                observed["Go"] = left.detach().clone()
                observed["Wo"] = right.detach().clone()
            selected_left = d_mm_left_replacement if d_mm_left_replacement is not None else left
            return original_mm(selected_left, right, out=out)

        extern_kernels.bmm = wrapped_bmm
        extern_kernels.mm = wrapped_mm
        target_softmax_kernel.run = wrapped_softmax_run
        try:
            loss.backward()
            torch.cuda.synchronize(device)
        finally:
            extern_kernels.bmm = original_bmm
            extern_kernels.mm = original_mm
            target_softmax_kernel.run = original_softmax_run
        if not observed.get("target"):
            raise RuntimeError("actual bmm_76 was not observed")
        if (capture_softmax or softmax_u_replacement is not None) and not observed.get("softmax_target"):
            raise RuntimeError("actual layer-23 softmax backward was not observed")
        if (capture_u_bmm or u_bmm_replacement is not None) and not observed.get("u_bmm_target"):
            raise RuntimeError("actual bmm_74 was not observed")
        if (capture_d_mm or d_mm_left_replacement is not None) and not observed.get("d_mm_target"):
            raise RuntimeError("actual mm_262 was not observed")
        return float(loss.detach().float().cpu()), q_proj_parameter.grad.detach()[ROWS, COLUMNS].clone(), observed

    rows = []
    try:
        for state_index in state_indices:
            inputs = all_states[state_index]
            eager_capture.clear()
            model.zero_grad(set_to_none=True)
            def o_proj_hook(_module, _values, output):
                def output_hook(gradient):
                    eager_capture["Go"] = gradient.detach().reshape(1024, 2048).clone()
                    return gradient
                output.register_hook(output_hook)

            o_proj_handle = model.model.layers[23].self_attn.o_proj.register_forward_hook(o_proj_hook)
            modeling_qwen3.eager_attention_forward = captured_eager_attention
            try:
                eager_loss = model(input_ids=inputs[0], labels=inputs[1], use_cache=False, return_dict=False)[0]
                eager_loss.backward()
            finally:
                # torch.compile guards this global.  Restore it before every
                # candidate call so the already compiled graph remains valid.
                modeling_qwen3.eager_attention_forward = original_attention
                o_proj_handle.remove()
            torch.cuda.synchronize(device)
            eager_tile = q_proj_parameter.grad.detach()[ROWS, COLUMNS].clone()
            if set(eager_capture) != {"S", "K", "V", "D", "Go", "Gq", "P", "U"}:
                raise RuntimeError(f"eager operand capture incomplete: {sorted(eager_capture)}")
            reference_s = eager_capture["S"]
            reference_k = eager_capture["K"]
            reference_gq = eager_capture["Gq"]
            replay_reference_gq = torch.empty_like(reference_gq)
            original_bmm(reference_s, reference_k, out=replay_reference_gq)
            replay_reference_s = torch.ops.aten._softmax_backward_data.default(
                eager_capture["U"].float(), eager_capture["P"].float(), -1, torch.float32
            ).to(torch.bfloat16) * (128 ** -0.5)
            replay_reference_u = torch.empty_like(eager_capture["U"])
            original_bmm(
                eager_capture["D"], eager_capture["V"].transpose(1, 2), out=replay_reference_u
            )
            replay_reference_d = torch.empty((1024, 2048), dtype=torch.bfloat16, device=device)
            original_mm(
                eager_capture["Go"],
                model.model.layers[23].self_attn.o_proj.weight.detach(),
                out=replay_reference_d,
            )
            eager_d_flat = eager_capture["D"].reshape(1, 16, 1024, 128).transpose(1, 2).contiguous().reshape(1024, 2048)
            probability_fp32 = eager_capture["P"].float()
            upstream_fp32 = eager_capture["U"].float()
            product_fp32 = upstream_fp32 * probability_fp32
            dot_fp32 = product_fp32.sum(dim=-1, keepdim=True)
            replay_reference_s_fma = torch.ops.prims.fma.default(
                -probability_fp32, dot_fp32, product_fp32
            ).to(torch.bfloat16) * (128 ** -0.5)

            candidate_loss, tile_cc, candidate_operands = run_candidate(
                inputs, capture_operands=True, capture_softmax=True, capture_u_bmm=True, capture_d_mm=True
            )
            candidate_s = candidate_operands["S"]
            candidate_k = candidate_operands["K"]
            candidate_u = candidate_operands["U"]
            candidate_d = candidate_operands["D"]
            candidate_vt = candidate_operands["Vt"]
            candidate_go = candidate_operands["Go"]
            _, tile_rc, _ = run_candidate(inputs, replacement=(reference_s, candidate_k))
            if args.antithetic_only:
                natural_local_plus = candidate_s.float() - reference_s.float()
                raw_antithetic_s = (
                    reference_s.float().mul(2.0).sub(candidate_s.float())
                ).to(reference_s.dtype)

                # A BF16 value need not have an exactly representable reflection
                # around another BF16 value (notably at exponent boundaries).  Do
                # not call the rounded reflection an antithetic control.  Project
                # each natural nonzero residual to the nearest value in a fixed,
                # local BF16 search orbit that *does* have an exact additive inverse.
                # The orbit and selection metric use only local representability,
                # never the downstream gradient or trajectory.
                natural_nonzero = natural_local_plus.ne(0)
                best_error = torch.full_like(natural_local_plus, float("inf"))
                matched_plus_s = reference_s.clone()
                matched_minus_s = reference_s.clone()

                def consider_exact_pair(proposal):
                    reflected = (
                        reference_s.float().mul(2.0).sub(proposal.float())
                    ).to(reference_s.dtype)
                    proposal_delta = proposal.float() - reference_s.float()
                    reflected_delta = reflected.float() - reference_s.float()
                    valid = natural_nonzero & proposal_delta.ne(0)
                    valid &= reflected_delta.eq(-proposal_delta)
                    valid &= torch.isfinite(proposal_delta)
                    error = (proposal.float() - candidate_s.float()).abs()
                    use = valid & error.lt(best_error)
                    best_error.copy_(torch.where(use, error, best_error))
                    matched_plus_s.copy_(torch.where(use, proposal, matched_plus_s))
                    matched_minus_s.copy_(torch.where(use, reflected, matched_minus_s))

                consider_exact_pair(candidate_s)
                toward = candidate_s
                away = candidate_s
                away_direction = torch.where(
                    natural_local_plus.gt(0),
                    torch.full_like(candidate_s, float("inf")),
                    torch.full_like(candidate_s, float("-inf")),
                )
                for _ in range(8):
                    toward = torch.nextafter(toward, reference_s)
                    away = torch.nextafter(away, away_direction)
                    consider_exact_pair(toward)
                    consider_exact_pair(away)
                local_plus = matched_plus_s.float() - reference_s.float()
                local_minus = matched_minus_s.float() - reference_s.float()

                _, tile_plus, _ = run_candidate(
                    inputs, replacement=(matched_plus_s, candidate_k)
                )
                _, tile_minus, _ = run_candidate(
                    inputs, replacement=(matched_minus_s, candidate_k)
                )
                _, tile_sham, _ = run_candidate(
                    inputs,
                    replacement=(matched_plus_s.clone(), candidate_k.clone()),
                )

                natural_gradient_plus = tile_cc.float() - tile_rc.float()
                gradient_plus = tile_plus.float() - tile_rc.float()
                gradient_minus = tile_minus.float() - tile_rc.float()
                parameter_tile = q_proj_parameter.detach()[ROWS, COLUMNS].float()
                zero_moment = torch.zeros_like(parameter_tile)

                def adam_value(gradient):
                    return adamw_update(
                        gradient.float(), zero_moment, zero_moment,
                        parameter_tile, step=1, learning_rate=1.0e-5,
                        betas=(0.9, 0.95), epsilon=1.0e-8,
                        weight_decay=0.0,
                    )["value"]

                update_zero = adam_value(tile_rc)
                update_plus = adam_value(tile_plus) - update_zero
                update_minus = adam_value(tile_minus) - update_zero

                def pair_geometry(plus, minus):
                    even = (plus.float() + minus.float()).mul(0.5)
                    odd = (plus.float() - minus.float()).mul(0.5)
                    plus_l2 = float(plus.float().norm())
                    minus_l2 = float(minus.float().norm())
                    even_l2 = float(even.norm())
                    odd_l2 = float(odd.norm())
                    return {
                        "plus_l2": plus_l2,
                        "minus_l2": minus_l2,
                        "even_l2": even_l2,
                        "odd_l2": odd_l2,
                        "balanced_mean_over_natural": even_l2 / max(plus_l2, 1.0e-30),
                        "balanced_mean_suppression": 1.0 - even_l2 / max(plus_l2, 1.0e-30),
                        "norm_ratio_minus_over_plus": minus_l2 / max(plus_l2, 1.0e-30),
                        "pair_sum_max_abs": float((plus.float() + minus.float()).abs().max()),
                    }

                local_geometry = pair_geometry(local_plus, local_minus)
                gradient_geometry = pair_geometry(gradient_plus, gradient_minus)
                sgd_geometry = pair_geometry(
                    gradient_plus.mul(-1.0e-5), gradient_minus.mul(-1.0e-5)
                )
                adam_geometry = pair_geometry(update_plus, update_minus)
                natural_local_energy = float(natural_local_plus.square().sum())
                matched_local_energy = float(local_plus.square().sum())
                natural_support = int(torch.count_nonzero(natural_local_plus))
                matched_support = int(torch.count_nonzero(local_plus))
                local_projection_residual = float(
                    (local_plus - natural_local_plus).norm()
                    / natural_local_plus.norm().clamp_min(1.0e-30)
                )
                gradient_projection_residual = float(
                    (gradient_plus - natural_gradient_plus).norm()
                    / natural_gradient_plus.norm().clamp_min(1.0e-30)
                )
                rows.append({
                    "state_index": state_index,
                    "offset": evaluation["offsets"][state_index],
                    "token_sha256": evaluation["token_sha256"][state_index],
                    "eager_loss": float(eager_loss.detach().float().cpu()),
                    "candidate_loss": candidate_loss,
                    "local_s_bwd": local_geometry,
                    "gradient_q_proj_tile": gradient_geometry,
                    "stateless_sgd_q_proj_tile": sgd_geometry,
                    "adamw_zero_moment_step1_q_proj_tile": adam_geometry,
                    "natural_source_fidelity": {
                        "natural_local_nonzero_coordinates": natural_support,
                        "matched_local_nonzero_coordinates": matched_support,
                        "matched_support_fraction": matched_support / max(natural_support, 1),
                        "matched_energy_fraction": matched_local_energy / max(natural_local_energy, 1.0e-30),
                        "matched_local_relative_residual": local_projection_residual,
                        "matched_local_fidelity": 1.0 - local_projection_residual,
                        "matched_gradient_relative_residual": gradient_projection_residual,
                        "matched_gradient_fidelity": 1.0 - gradient_projection_residual,
                    },
                    "local_support_equal": bool(torch.equal(
                        local_plus != 0, local_minus != 0
                    )),
                    "local_antithetic_exact": bool(torch.equal(local_plus, -local_minus)),
                    "candidate_restoration_sham_max_abs": float(
                        (tile_sham.float() - tile_plus.float()).abs().max()
                    ),
                    "candidate_restoration_sham_exact": bool(torch.equal(tile_sham, tile_plus)),
                    "reference_s_candidate_k_is_zero_arm": True,
                })
                print(json.dumps({
                    "event": "ANTITHETIC_STATE_COMPLETE",
                    "state_index": state_index,
                    "local_exact": rows[-1]["local_antithetic_exact"],
                    "gradient_suppression": gradient_geometry["balanced_mean_suppression"],
                    "adam_suppression": adam_geometry["balanced_mean_suppression"],
                }, sort_keys=True), flush=True)
                continue
            _, tile_cr, _ = run_candidate(inputs, replacement=(candidate_s, reference_k))
            _, tile_rr, _ = run_candidate(inputs, replacement=(reference_s, reference_k))
            _, tile_sham, _ = run_candidate(inputs, replacement=(candidate_s.clone(), candidate_k.clone()))

            # Second split inside S_bwd.  "U" is the probability/output-path
            # cotangent.  "P/program" contains saved forward probability and
            # the local softmax VJP arithmetic/materialization schedule.
            reference_p_candidate_u_s = torch.ops.aten._softmax_backward_data.default(
                candidate_u.float(), eager_capture["P"], -1, torch.float32
            ).to(torch.bfloat16) * (128 ** -0.5)
            _, tile_ur_pc, _ = run_candidate(inputs, softmax_u_replacement=eager_capture["U"])
            _, tile_uc_pr, _ = run_candidate(
                inputs, replacement=(reference_p_candidate_u_s, candidate_k)
            )
            _, tile_ur_pr, _ = run_candidate(inputs, replacement=(reference_s, candidate_k))

            # Third split inside U = D @ V^T at actual bmm_74.  Each hybrid
            # continues through the candidate softmax VJP and bmm_76.
            reference_d = eager_capture["D"]
            reference_vt = eager_capture["V"].transpose(1, 2)
            _, tile_dr_vc, _ = run_candidate(
                inputs, u_bmm_replacement=(reference_d, candidate_vt)
            )
            _, tile_dc_vr, _ = run_candidate(
                inputs, u_bmm_replacement=(candidate_d, reference_vt)
            )
            _, tile_dr_vr, _ = run_candidate(
                inputs, u_bmm_replacement=(reference_d, reference_vt)
            )
            _, tile_u_sham, _ = run_candidate(
                inputs, u_bmm_replacement=(candidate_d.clone(), candidate_vt.clone())
            )
            _, tile_go_reference, _ = run_candidate(
                inputs, d_mm_left_replacement=eager_capture["Go"]
            )
            _, tile_go_sham, _ = run_candidate(
                inputs, d_mm_left_replacement=candidate_go.clone()
            )

            delta_cc = tile_cc.float() - eager_tile.float()
            delta_rc = tile_rc.float() - eager_tile.float()
            delta_cr = tile_cr.float() - eager_tile.float()
            delta_rr = tile_rr.float() - eager_tile.float()
            s_shapley = 0.5 * ((tile_cc.float() - tile_rc.float()) + (tile_cr.float() - tile_rr.float()))
            k_shapley = 0.5 * ((tile_cc.float() - tile_cr.float()) + (tile_rc.float() - tile_rr.float()))
            total_removal = tile_cc.float() - tile_rr.float()
            u_shapley = 0.5 * (
                (tile_cc.float() - tile_ur_pc.float())
                + (tile_uc_pr.float() - tile_ur_pr.float())
            )
            p_program_shapley = 0.5 * (
                (tile_cc.float() - tile_uc_pr.float())
                + (tile_ur_pc.float() - tile_ur_pr.float())
            )
            softmax_total_removal = tile_cc.float() - tile_ur_pr.float()
            d_shapley = 0.5 * (
                (tile_cc.float() - tile_dr_vc.float())
                + (tile_dc_vr.float() - tile_dr_vr.float())
            )
            v_shapley = 0.5 * (
                (tile_cc.float() - tile_dc_vr.float())
                + (tile_dr_vc.float() - tile_dr_vr.float())
            )
            u_bmm_total_removal = tile_cc.float() - tile_dr_vr.float()
            rows.append({
                "state_index": state_index,
                "offset": evaluation["offsets"][state_index],
                "token_sha256": evaluation["token_sha256"][state_index],
                "eager_loss": float(eager_loss.detach().float().cpu()),
                "candidate_loss": candidate_loss,
                "reference_bmm_replay_matches_eager_query_gradient": bool(torch.equal(replay_reference_gq, reference_gq)),
                "reference_softmax_vjp_replay_matches_eager_s": bool(torch.equal(replay_reference_s, reference_s)),
                "reference_softmax_vjp_replay_max_abs": float((replay_reference_s.float() - reference_s.float()).abs().max()),
                "reference_softmax_fma_replay_matches_eager_s": bool(torch.equal(replay_reference_s_fma, reference_s)),
                "reference_softmax_fma_replay_max_abs": float((replay_reference_s_fma.float() - reference_s.float()).abs().max()),
                "reference_u_bmm_replay_matches_eager_u": bool(torch.equal(replay_reference_u, eager_capture["U"])),
                "reference_o_proj_input_vjp_replay_matches_eager_d": bool(
                    torch.equal(replay_reference_d, eager_d_flat)
                ),
                "reference_o_proj_input_vjp_replay_max_abs": float(
                    (replay_reference_d.float() - eager_d_flat.float()).abs().max()
                ),
                "candidate_minus_eager_projection": projection(delta_cc, direction, torch),
                "reference_s_candidate_k_residual_projection": projection(delta_rc, direction, torch),
                "candidate_s_reference_k_residual_projection": projection(delta_cr, direction, torch),
                "reference_s_reference_k_residual_projection": projection(delta_rr, direction, torch),
                "s_shapley_removal_projection": projection(s_shapley, direction, torch),
                "k_shapley_removal_projection": projection(k_shapley, direction, torch),
                "total_bmm_operand_removal_projection": projection(total_removal, direction, torch),
                "shapley_closure_max_abs": float((s_shapley + k_shapley - total_removal).abs().max()),
                "candidate_restoration_sham_max_abs": float((tile_sham.float() - tile_cc.float()).abs().max()),
                "candidate_restoration_sham_projection": projection(tile_sham.float() - tile_cc.float(), direction, torch),
                "s_delta_l2": float((candidate_s.float() - reference_s.float()).norm()),
                "k_delta_l2": float((candidate_k.float() - reference_k.float()).norm()),
                "rr_residual_l2": float(delta_rr.norm()),
                "reference_u_candidate_p_program_residual_projection": projection(tile_ur_pc.float() - eager_tile.float(), direction, torch),
                "candidate_u_reference_p_program_residual_projection": projection(tile_uc_pr.float() - eager_tile.float(), direction, torch),
                "reference_u_reference_p_program_residual_projection": projection(tile_ur_pr.float() - eager_tile.float(), direction, torch),
                "u_shapley_removal_projection": projection(u_shapley, direction, torch),
                "p_program_shapley_removal_projection": projection(p_program_shapley, direction, torch),
                "softmax_factor_total_removal_projection": projection(softmax_total_removal, direction, torch),
                "softmax_factor_shapley_closure_max_abs": float(
                    (u_shapley + p_program_shapley - softmax_total_removal).abs().max()
                ),
                "reference_d_candidate_v_residual_projection": projection(tile_dr_vc.float() - eager_tile.float(), direction, torch),
                "candidate_d_reference_v_residual_projection": projection(tile_dc_vr.float() - eager_tile.float(), direction, torch),
                "reference_d_reference_v_residual_projection": projection(tile_dr_vr.float() - eager_tile.float(), direction, torch),
                "d_shapley_removal_projection": projection(d_shapley, direction, torch),
                "v_shapley_removal_projection": projection(v_shapley, direction, torch),
                "u_bmm_total_removal_projection": projection(u_bmm_total_removal, direction, torch),
                "u_bmm_shapley_closure_max_abs": float(
                    (d_shapley + v_shapley - u_bmm_total_removal).abs().max()
                ),
                "u_bmm_candidate_restoration_sham_max_abs": float(
                    (tile_u_sham.float() - tile_cc.float()).abs().max()
                ),
                "reference_go_residual_projection": projection(
                    tile_go_reference.float() - eager_tile.float(), direction, torch
                ),
                "go_removal_projection": projection(
                    tile_cc.float() - tile_go_reference.float(), direction, torch
                ),
                "go_candidate_restoration_sham_max_abs": float(
                    (tile_go_sham.float() - tile_cc.float()).abs().max()
                ),
            })
    finally:
        modeling_qwen3.eager_attention_forward = original_attention

    if args.antithetic_only:
        minimum_suppression = 0.90
        minimum_natural_source_fidelity = 0.90
        validity_gates = {
            "sixteen_fixed_conditions": len(rows) == 16,
            "all_local_antithetic_exact": all(
                row["local_antithetic_exact"] for row in rows
            ),
            "all_local_support_equal": all(row["local_support_equal"] for row in rows),
            "all_shams_exact": all(
                row["candidate_restoration_sham_exact"] for row in rows
            ),
            "natural_source_fidelity_every_condition": all(
                row["natural_source_fidelity"]["matched_energy_fraction"]
                >= minimum_natural_source_fidelity
                and row["natural_source_fidelity"]["matched_local_fidelity"]
                >= minimum_natural_source_fidelity
                and row["natural_source_fidelity"]["matched_gradient_fidelity"]
                >= minimum_natural_source_fidelity
                for row in rows
            ),
        }
        mechanism_gates = {
            "gradient_balanced_mean_suppressed_every_condition": all(
                row["gradient_q_proj_tile"]["balanced_mean_suppression"]
                >= minimum_suppression for row in rows
            ),
            "gradient_response_even_nonzero_every_condition": all(
                row["gradient_q_proj_tile"]["even_l2"] > 0.0 for row in rows
            ),
            "adamw_zero_moment_response_even_nonzero_every_condition": all(
                row["adamw_zero_moment_step1_q_proj_tile"]["even_l2"] > 0.0
                for row in rows
            ),
        }
        validity_complete = all(validity_gates.values())
        result = {
            "schema": "kernel-analyzer-l23-s-bwd-antithetic-v1",
            "status": (
                "MATCHED_ANTITHETIC_MECHANISM_DECOMPOSITION"
                if validity_complete else "ANTITHETIC_INTERVENTION_UNRESOLVED"
            ),
            "parameter": PARAMETER,
            "tile": {"rows": [ROWS.start, ROWS.stop], "columns": [COLUMNS.start, COLUMNS.stop]},
            "checkpoint_step": args.step,
            "state_indices": state_indices,
            "binding": {
                "actual_backward_node": "bmm_76",
                "equation": "G_q = S_bwd @ K; dW_q = G_q^T H",
                "intervention": (
                    "S_ref +/- epsilon after a fixed nearest-BF16 projection onto "
                    "residuals with an exact additive inverse; S_ref is the zero arm"
                ),
                "right_operand_fixed": "candidate K",
                "generated_source_sha256": source_sha256,
                "candidate_values_used_to_select_boundary": False,
            },
            "optimizer_probe": {
                "name": "AdamW",
                "learning_rate": 1.0e-5,
                "betas": [0.9, 0.95],
                "epsilon": 1.0e-8,
                "weight_decay": 0.0,
                "moments": "ZERO_STEP1",
                "natural_mature_moments_measured": False,
            },
            "minimum_balanced_mean_suppression": minimum_suppression,
            "minimum_natural_source_fidelity": minimum_natural_source_fidelity,
            "validity_gates": validity_gates,
            "mechanism_gates": mechanism_gates,
            "mechanism_verdicts": {
                "source_event_asymmetry_dominates_gradient_response": (
                    "SUPPORTED_EVERY_CONDITION"
                    if validity_complete and mechanism_gates[
                        "gradient_balanced_mean_suppressed_every_condition"
                    ] else "NOT_SUPPORTED_UNDER_FROZEN_90_PERCENT_GATE"
                ),
                "fb_response_rectification": (
                    "SUPPORTED_EVERY_CONDITION"
                    if validity_complete and mechanism_gates[
                        "gradient_response_even_nonzero_every_condition"
                    ] else "UNRESOLVED"
                ),
                "adamw_zero_moment_step1_response_rectification": (
                    "SUPPORTED_EVERY_CONDITION"
                    if validity_complete and mechanism_gates[
                        "adamw_zero_moment_response_even_nonzero_every_condition"
                    ] else "UNRESOLVED"
                ),
            },
            "rows": rows,
            "claim_boundary": (
                "The matched source claim projects each natural residual within a fixed "
                "eight-neighbor BF16 orbit to the nearest exactly invertible residual. "
                "It requires at least 90% local approximation, local-energy, and natural-"
                "gradient fidelity, exact local +/-epsilon support, "
                "and at least 90% gradient balanced-mean suppression in every fixed "
                "condition to call event asymmetry dominant. Exact-pair gradient and "
                "AdamW even components separately identify response rectification; they "
                "are not relabeled as source asymmetry. The stateless-SGD row is the "
                "expected linear scaling and is not a duplicate gate. AdamW uses zero "
                "moments at step 1 and does not claim mature-optimizer behavior."
            ),
            "tensor_values_saved": False,
        }
    else:
        result = {
            "schema": "kernel-analyzer-l23-attention-bmm-decomposition-v1",
            "status": "COMPLETE",
            "parameter": PARAMETER,
            "tile": {"rows": [ROWS.start, ROWS.stop], "columns": [COLUMNS.start, COLUMNS.stop]},
            "checkpoint_step": args.step,
            "state_indices": state_indices,
            "binding": {
                "actual_backward_node": "bmm_76",
                "equation": "G_q = S_bwd @ K",
                "left_operand": "buf192: scaled softmax-backward output",
                "right_operand": "permute_510: repeated layer-23 post-RoPE key",
                "backward_bmm_zero_based_ordinal": target_ordinal,
                "attention_output_vjp_mm": "mm_262: D = Go @ Wo",
                "attention_output_vjp_mm_zero_based_ordinal": target_d_mm_ordinal,
                "softmax_backward_symbol": target_softmax_symbol,
                "softmax_backward_same_symbol_zero_based_ordinal": target_softmax_ordinal,
                "generated_source_sha256": source_sha256,
                "candidate_values_used_to_select_boundary": False,
            },
            "hybrids": {"CC": "candidate S/K", "RC": "reference S, candidate K", "CR": "candidate S, reference K", "RR": "reference S/K"},
            "rows": rows,
            "tensor_values_saved": False,
        }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"output": str(output_path), "rows": len(rows), "target_ordinal": target_ordinal}, sort_keys=True))


if __name__ == "__main__":
    main()
