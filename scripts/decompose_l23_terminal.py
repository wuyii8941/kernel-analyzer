#!/usr/bin/env python3
"""Decompose the terminal cotangent that drives the layer-23 q_proj carrier.

The real chain is

    Gz = d NLL(log_softmax(logits)) / d logits,
    Dn = Gz @ W_lm_head,
    T  = J_RMSNorm(H)^T Dn.

Interventions are nested in that causal order at actual mm_198 and the actual
final-RMSNorm VJP kernel.  The real layer-27..23 backward then continues to the
fixed q_proj weight tile.
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
for path in (OLD_SRC, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.long_horizon_trigger import atomic_json, build_model, load_eval_states, load_milestone, under_root


PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280)
COLUMNS = slice(1664, 1792)


def projection(value, direction, torch) -> float:
    return float(torch.dot(value.reshape(-1).float(), direction.reshape(-1).float()) / direction.float().norm())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--state-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    direction_path = under_root(args.direction, "direction")
    output_path = under_root(args.output, "output")
    state_indices = args.state_index or list(range(8, 40))
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
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.select_algorithm import extern_kernels
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
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
    parameter = dict(model.named_parameters())[PARAMETER]
    direction = torch.load(direction_path, map_location="cpu", weights_only=False)["direction"].float().to(device)

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__(); self.subject = subject
        def forward(self, input_ids, labels):
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm = candidate(*all_states[state_indices[0]]); warm.backward(); torch.cuda.synchronize(device)
    matches = []
    for module in list(PyCodeCache.modules[module_start:]):
        path = Path(module.__file__); source = path.read_text()
        if "mm_198]" in source and "bmm_76]" in source:
            matches.append((module, path, source))
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact terminal backward source, got {len(matches)}")
    source_module, source_path, source = matches[0]
    call_start = source.index("def call(")
    mm_position = source.index("mm_198]", call_start)
    mm_ordinal = source[call_start:mm_position].count("extern_kernels.mm(")
    nll_calls = list(re.finditer(r"([A-Za-z0-9_]*log_softmax[A-Za-z0-9_]*nll_loss[A-Za-z0-9_]*)\.run\(", source[call_start:mm_position]))
    if len(nll_calls) != 1:
        raise RuntimeError(f"expected one terminal NLL/log-softmax VJP kernel, got {len(nll_calls)}")
    nll_symbol = nll_calls[0].group(1)
    nll_abs_position = call_start + nll_calls[0].start()
    nll_ordinal = source[call_start:nll_abs_position].count(f"{nll_symbol}.run(")
    nll_kernel = getattr(source_module, nll_symbol)
    norm_position = source.index("mm_199]", mm_position)
    norm_region = source[mm_position:norm_position]
    norm_calls = list(re.finditer(r"([A-Za-z0-9_]*add_div_expand_mul_pow_sum_view_3)\.run\(", norm_region))
    if len(norm_calls) != 1:
        raise RuntimeError("exact final RMSNorm VJP kernel was not found")
    norm_symbol = norm_calls[0].group(1)
    norm_abs_position = mm_position + norm_calls[0].start()
    norm_ordinal = source[call_start:norm_abs_position].count(f"{norm_symbol}.run(")
    norm_kernel = getattr(source_module, norm_symbol)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    eager_capture = {}
    def reference_logits_vjp(logits, labels):
        flat_logits = logits.reshape(1024, 151936)
        targets = labels.reshape(-1)[1:]
        valid = targets.ne(-100)
        log_probability = torch.log_softmax(flat_logits[:-1].float(), dim=-1)
        gradient = log_probability.exp()
        if not bool(valid.all()):
            gradient[~valid] = 0.0
        rows = torch.arange(1023, device=logits.device)[valid]
        gradient[rows, targets[valid]] -= 1.0
        gradient.div_(valid.sum())
        result = torch.zeros_like(flat_logits)
        result[:-1].copy_(gradient.to(result.dtype))
        return result

    def capture_eager(inputs):
        eager_capture.clear()
        def norm_hook(_module, values, output):
            def input_grad(g):
                eager_capture["T"] = g.detach().clone()
                return g
            def output_grad(g):
                eager_capture["Dn"] = g.detach().clone()
                return g
            values[0].register_hook(input_grad)
            output.register_hook(output_grad)
        def head_hook(_module, _values, output):
            eager_capture["Logits"] = output.detach().clone()
            def logits_grad(g):
                eager_capture["Gz"] = g.detach().clone()
                return g
            output.register_hook(logits_grad)
        handles = [model.model.norm.register_forward_hook(norm_hook), model.lm_head.register_forward_hook(head_hook)]
        model.zero_grad(set_to_none=True)
        try:
            loss = model(input_ids=inputs[0], labels=inputs[1], use_cache=False, return_dict=False)[0]
            loss.backward(); torch.cuda.synchronize(device)
        finally:
            for handle in handles: handle.remove()
        if set(eager_capture) != {"Logits", "Gz", "Dn", "T"}:
            raise RuntimeError(f"terminal eager capture incomplete: {sorted(eager_capture)}")
        return float(loss.detach().float().cpu()), parameter.grad.detach()[ROWS, COLUMNS].clone()

    original_mm = extern_kernels.mm
    original_norm_run = norm_kernel.run
    def run_candidate(inputs, arm, reference=None):
        if arm not in {"C", "LOCAL_G_REF", "G_REF", "DN_REF", "T_REF", "SHAM"}:
            raise ValueError(arm)
        model.zero_grad(set_to_none=True); loss = candidate(*inputs)
        mm_count = {"value": 0}; norm_count = {"value": 0}; nll_count = {"value": 0}; observed = {}
        original_nll_run = nll_kernel.run
        def wrapped_nll(*values, **kwargs):
            ordinal = nll_count["value"]; nll_count["value"] += 1
            if ordinal != nll_ordinal: return original_nll_run(*values, **kwargs)
            if tuple(values[0].shape) != (1, 1024, 151936):
                raise RuntimeError("terminal logits buffer shape changed")
            observed["nll"] = True
            observed["Logits"] = values[0].detach().clone()
            if arm == "LOCAL_G_REF":
                values[0].copy_(reference_logits_vjp(values[0], inputs[1]).reshape_as(values[0]))
                return None
            return original_nll_run(*values, **kwargs)
        def wrapped_mm(*values, **kwargs):
            ordinal = mm_count["value"]; mm_count["value"] += 1
            if ordinal != mm_ordinal: return original_mm(*values, **kwargs)
            left, right = values[:2]; out = kwargs.get("out")
            if tuple(left.shape) != (1024, 151936) or tuple(right.shape) != (151936, 2048) or tuple(out.shape) != (1024, 2048):
                raise RuntimeError("terminal mm_198 operands changed")
            observed["mm"] = True; observed["Gz"] = left.detach().clone()
            if arm == "G_REF": result = original_mm(reference["Gz"].reshape(1024, 151936), right, out=out)
            elif arm in {"DN_REF", "T_REF"}: out.copy_(reference["Dn"].reshape(1024, 2048)); result = None
            else:
                if arm == "SHAM": left = left.clone()
                result = original_mm(left, right, out=out)
            observed["Dn"] = out.detach().clone()
            return result
        def wrapped_norm(*values, **kwargs):
            ordinal = norm_count["value"]; norm_count["value"] += 1
            if ordinal != norm_ordinal: return original_norm_run(*values, **kwargs)
            if tuple(values[0].shape) != (1, 1024, 2048):
                raise RuntimeError("final RMSNorm VJP output shape changed")
            observed["norm"] = True
            if arm == "T_REF": values[0].copy_(reference["T"]); result = None
            else: result = original_norm_run(*values, **kwargs)
            observed["T"] = values[0].detach().clone()
            return result
        extern_kernels.mm = wrapped_mm; norm_kernel.run = wrapped_norm; nll_kernel.run = wrapped_nll
        try:
            loss.backward(); torch.cuda.synchronize(device)
        finally:
            extern_kernels.mm = original_mm; norm_kernel.run = original_norm_run; nll_kernel.run = original_nll_run
        if not observed.get("nll") or not observed.get("mm") or not observed.get("norm"):
            raise RuntimeError("terminal chain was not fully observed")
        return float(loss.detach().float().cpu()), parameter.grad.detach()[ROWS, COLUMNS].clone(), observed

    rows = []
    for state_index in state_indices:
        inputs = all_states[state_index]
        eager_loss, eager_tile = capture_eager(inputs)
        reference = {name: value for name, value in eager_capture.items()}
        analytic_reference_g = reference_logits_vjp(reference["Logits"], inputs[1]).reshape_as(reference["Gz"])
        replay_dn = torch.empty_like(reference["Dn"].reshape(1024, 2048))
        original_mm(reference["Gz"].reshape(1024, 151936), model.lm_head.weight.detach(), out=replay_dn)
        candidate_loss, tile_c, obs_c = run_candidate(inputs, "C")
        _, tile_local_g, _ = run_candidate(inputs, "LOCAL_G_REF", reference)
        _, tile_g, _ = run_candidate(inputs, "G_REF", reference)
        _, tile_dn, _ = run_candidate(inputs, "DN_REF", reference)
        _, tile_t, obs_t = run_candidate(inputs, "T_REF", reference)
        _, tile_sham, obs_sham = run_candidate(inputs, "SHAM")
        logits_stage = tile_c.float() - tile_g.float()
        local_logits_stage = tile_c.float() - tile_local_g.float()
        upstream_logits_stage = tile_local_g.float() - tile_g.float()
        mm_stage = tile_g.float() - tile_dn.float()
        norm_stage = tile_dn.float() - tile_t.float()
        terminal_total = tile_c.float() - tile_t.float()
        row = {
            "state_index": state_index,
            "offset": evaluation["offsets"][state_index],
            "token_sha256": evaluation["token_sha256"][state_index],
            "eager_loss": eager_loss,
            "candidate_loss": candidate_loss,
            "reference_lm_head_input_vjp_replay_matches": bool(torch.equal(replay_dn, reference["Dn"].reshape(1024, 2048))),
            "reference_lm_head_input_vjp_replay_max_abs": float((replay_dn.float() - reference["Dn"].reshape(1024, 2048).float()).abs().max()),
            "analytic_logits_vjp_matches_eager": bool(torch.equal(analytic_reference_g, reference["Gz"])),
            "analytic_logits_vjp_max_abs": float((analytic_reference_g.float() - reference["Gz"].float()).abs().max()),
            "candidate_minus_eager_projection": projection(tile_c.float() - eager_tile.float(), direction, torch),
            "logits_vjp_stage_removal_projection": projection(logits_stage, direction, torch),
            "local_logits_vjp_removal_projection": projection(local_logits_stage, direction, torch),
            "upstream_logits_removal_projection": projection(upstream_logits_stage, direction, torch),
            "logits_substage_closure_max_abs": float((local_logits_stage + upstream_logits_stage - logits_stage).abs().max()),
            "lm_head_mm_stage_removal_projection": projection(mm_stage, direction, torch),
            "final_norm_stage_removal_projection": projection(norm_stage, direction, torch),
            "terminal_total_removal_projection": projection(terminal_total, direction, torch),
            "terminal_stage_closure_max_abs": float((logits_stage + mm_stage + norm_stage - terminal_total).abs().max()),
            "reference_g_residual_projection": projection(tile_g.float() - eager_tile.float(), direction, torch),
            "reference_dn_residual_projection": projection(tile_dn.float() - eager_tile.float(), direction, torch),
            "reference_t_residual_projection": projection(tile_t.float() - eager_tile.float(), direction, torch),
            "candidate_restoration_sham_max_abs": float((tile_sham.float() - tile_c.float()).abs().max()),
            "candidate_g_sham_max_abs": float((obs_sham["Gz"].float() - obs_c["Gz"].float()).abs().max()),
            "candidate_dn_sham_max_abs": float((obs_sham["Dn"].float() - obs_c["Dn"].float()).abs().max()),
            "t_ref_matches_reference": bool(torch.equal(obs_t["T"], reference["T"])),
        }
        rows.append(row)
        print(json.dumps({"state": state_index, "total": row["candidate_minus_eager_projection"], "G": row["logits_vjp_stage_removal_projection"], "MM": row["lm_head_mm_stage_removal_projection"], "Norm": row["final_norm_stage_removal_projection"]}, sort_keys=True), flush=True)

    result = {
        "schema": "kernel-analyzer-l23-terminal-decomposition-v1",
        "status": "COMPLETE",
        "checkpoint_step": args.step,
        "state_indices": state_indices,
        "parameter": PARAMETER,
        "tile": {"rows": [ROWS.start, ROWS.stop], "columns": [COLUMNS.start, COLUMNS.stop]},
        "equations": ["Gz = dNLL(log_softmax(logits))/dlogits", "Dn = Gz @ W_lm_head", "T = J_RMSNorm(H)^T Dn"],
        "binding": {"terminal_logits_vjp_kernel": nll_symbol, "nll_same_symbol_zero_based_ordinal": nll_ordinal, "lm_head_input_vjp": "mm_198", "mm_zero_based_ordinal": mm_ordinal, "final_norm_vjp_kernel": norm_symbol, "norm_same_symbol_zero_based_ordinal": norm_ordinal, "generated_source_sha256": source_sha256, "candidate_values_used_to_select_boundary": False},
        "nested_repairs": ["reference Gz at actual mm_198", "reference Dn at mm_198 output", "reference T at actual final-RMSNorm VJP output"],
        "rows": rows,
        "tensor_values_saved": False,
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"output": str(output_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
