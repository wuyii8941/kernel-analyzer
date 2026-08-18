#!/usr/bin/env python3
"""Split the layer-23 attention-output cotangent into residual and MLP paths.

For Qwen3 layer 23 the cotangent consumed by the o_proj input VJP is

    Go = R + M,

where R is the direct residual-stream cotangent and M is the cotangent returned
through post_attention_layernorm + MLP.  Inductor fuses the final RMSNorm VJP
and the residual addition into one mutating Triton kernel.  This script binds
that exact current-compile call, crosses eager/candidate R and M paths, and
lets the actual mm_262 -> attention backward -> q_proj weight VJP continue.
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

from scripts.long_horizon_trigger import (
    atomic_json,
    build_model,
    load_eval_states,
    load_milestone,
    under_root,
)


PARAMETER = "model.layers.23.self_attn.q_proj.weight"
ROWS = slice(1152, 1280)
COLUMNS = slice(1664, 1792)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--direction", type=Path, default=Path("results/final/l23_qproj_tile_direction.pt"))
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--layer", type=int, default=23, choices=range(23, 28))
    parser.add_argument("--state-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/final/l23_go_path_decomposition.json"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def projection(value, direction, torch) -> float:
    return float(
        torch.dot(value.reshape(-1).float(), direction.reshape(-1).float())
        / direction.float().norm()
    )


def main() -> None:
    args = parse_args()
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
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/tile_causal_compile"
    )

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from transformers import AutoTokenizer

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
    all_states, evaluation = load_eval_states(
        tokenizer, 1024, max(state_indices) + 1, device
    )
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)
    parameter = dict(model.named_parameters())[PARAMETER]
    direction_payload = torch.load(direction_path, map_location="cpu", weights_only=False)
    if direction_payload.get("schema") != "kernel-analyzer-frozen-tile-direction-v1":
        raise ValueError("direction schema mismatch")
    direction = direction_payload["direction"].float().to(device)

    class LossStep(torch.nn.Module):
        def __init__(self, subject):
            super().__init__()
            self.subject = subject

        def forward(self, input_ids, labels):
            return self.subject(
                input_ids=input_ids, labels=labels, use_cache=False, return_dict=False
            )[0]

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend=lookup_backend("inductor"), fullgraph=True, dynamic=False
    )
    model.zero_grad(set_to_none=True)
    warm_loss = candidate(*all_states[state_indices[0]])
    warm_loss.backward()
    torch.cuda.synchronize(device)

    consumer_mm = 206 + (27 - args.layer) * 14
    mm_marker = f"mm_{consumer_mm}]"
    previous_mm_marker = f"mm_{consumer_mm - 2}]"
    matches = []
    for module in list(PyCodeCache.modules[module_start:]):
        source_path = Path(module.__file__)
        source = source_path.read_text()
        if mm_marker in source and "bmm_76]" in source:
            matches.append((module, source_path, source))
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact layer-23 backward source, got {len(matches)}")
    source_module, source_path, source = matches[0]
    mm_position = source.index(mm_marker)
    call_start = source.rfind("def call(", 0, mm_position)
    previous_position = source.rfind(previous_mm_marker, call_start, mm_position)
    if previous_position < 0:
        raise RuntimeError("could not locate the layer-23 MLP VJP predecessor")
    region = source[previous_position:mm_position]
    calls = list(re.finditer(r"([A-Za-z0-9_]+)\.run\(", region))
    if not calls:
        raise RuntimeError("could not locate the mutating post-attention RMSNorm VJP")
    target_symbol = calls[-1].group(1)
    target_position = previous_position + calls[-1].start()
    if "add_div_expand_mul_pow_sum" not in target_symbol:
        raise RuntimeError(f"unexpected target kernel: {target_symbol}")
    target_ordinal = source[call_start:target_position].count(f"{target_symbol}.run(")
    target_kernel = getattr(source_module, target_symbol)
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    eager_capture: dict[str, torch.Tensor] = {}
    layer = model.model.layers[args.layer]

    def capture_eager(inputs):
        eager_capture.clear()

        def o_proj_hook(_module, _values, output):
            def grad_hook(gradient):
                eager_capture["Go"] = gradient.detach().clone()
                return gradient

            output.register_hook(grad_hook)

        def mlp_hook(_module, _values, output):
            def grad_hook(gradient):
                eager_capture["R"] = gradient.detach().clone()
                return gradient

            output.register_hook(grad_hook)

        def post_norm_backward(_module, grad_input, _grad_output):
            if len(grad_input) != 1 or grad_input[0] is None:
                raise RuntimeError("post-attention RMSNorm grad_input is incomplete")
            eager_capture["M"] = grad_input[0].detach().clone()

        handles = [
            layer.self_attn.o_proj.register_forward_hook(o_proj_hook),
            layer.mlp.register_forward_hook(mlp_hook),
            layer.post_attention_layernorm.register_full_backward_hook(post_norm_backward),
        ]
        model.zero_grad(set_to_none=True)
        try:
            loss = model(
                input_ids=inputs[0], labels=inputs[1], use_cache=False, return_dict=False
            )[0]
            loss.backward()
            torch.cuda.synchronize(device)
        finally:
            for handle in handles:
                handle.remove()
        if set(eager_capture) != {"Go", "R", "M"}:
            raise RuntimeError(f"eager path capture incomplete: {sorted(eager_capture)}")
        expected = (1, 1024, 2048)
        if any(tuple(value.shape) != expected for value in eager_capture.values()):
            raise RuntimeError("eager Go path has an unexpected shape")
        return float(loss.detach().float().cpu()), parameter.grad.detach()[ROWS, COLUMNS].clone()

    original_run = target_kernel.run

    def run_candidate(inputs, arm: str, reference=None):
        if arm not in {"CC", "RC", "CR", "RR", "SHAM"}:
            raise ValueError(f"unknown arm: {arm}")
        model.zero_grad(set_to_none=True)
        loss = candidate(*inputs)
        counter = {"call": 0}
        observed = {}

        def wrapped_run(*values, **kwargs):
            ordinal = counter["call"]
            counter["call"] += 1
            if ordinal != target_ordinal:
                return original_run(*values, **kwargs)
            if not values or tuple(values[0].shape) != (1, 1024, 2048):
                raise RuntimeError("exact layer-23 residual cotangent has unexpected shape")
            direct = values[0]
            observed["target"] = True
            observed["R"] = direct.detach().clone()
            if arm in {"CC", "SHAM"}:
                if arm == "SHAM":
                    direct.copy_(direct.clone())
                result = original_run(*values, **kwargs)
            elif arm == "RC":
                direct.copy_(reference["R"])
                result = original_run(*values, **kwargs)
            elif arm == "CR":
                # The reference M is an explicitly materialized eager F+B
                # endpoint.  Its BF16 add is part of the reference path.
                torch.add(direct, reference["M"], out=direct)
                result = None
            else:
                direct.copy_(reference["Go"])
                result = None
            observed["Go"] = direct.detach().clone()
            return result

        target_kernel.run = wrapped_run
        try:
            loss.backward()
            torch.cuda.synchronize(device)
        finally:
            target_kernel.run = original_run
        if not observed.get("target"):
            raise RuntimeError("exact layer-23 Go construction kernel was not observed")
        return (
            float(loss.detach().float().cpu()),
            parameter.grad.detach()[ROWS, COLUMNS].clone(),
            observed,
        )

    rows = []
    for state_index in state_indices:
        inputs = all_states[state_index]
        eager_loss, eager_tile = capture_eager(inputs)
        reference = {name: value for name, value in eager_capture.items()}
        replay_go = torch.add(reference["R"], reference["M"])

        candidate_loss, tile_cc, obs_cc = run_candidate(inputs, "CC")
        _, tile_rc, obs_rc = run_candidate(inputs, "RC", reference)
        _, tile_cr, obs_cr = run_candidate(inputs, "CR", reference)
        _, tile_rr, obs_rr = run_candidate(inputs, "RR", reference)
        _, tile_sham, obs_sham = run_candidate(inputs, "SHAM")

        delta_cc = tile_cc.float() - eager_tile.float()
        delta_rc = tile_rc.float() - eager_tile.float()
        delta_cr = tile_cr.float() - eager_tile.float()
        delta_rr = tile_rr.float() - eager_tile.float()
        r_shapley = 0.5 * (
            (tile_cc.float() - tile_rc.float()) + (tile_cr.float() - tile_rr.float())
        )
        m_shapley = 0.5 * (
            (tile_cc.float() - tile_cr.float()) + (tile_rc.float() - tile_rr.float())
        )
        total_removal = tile_cc.float() - tile_rr.float()
        rows.append(
            {
                "state_index": state_index,
                "offset": evaluation["offsets"][state_index],
                "token_sha256": evaluation["token_sha256"][state_index],
                "eager_loss": eager_loss,
                "candidate_loss": candidate_loss,
                "reference_go_replay_matches": bool(torch.equal(replay_go, reference["Go"])),
                "reference_go_replay_max_abs": float(
                    (replay_go.float() - reference["Go"].float()).abs().max()
                ),
                "candidate_minus_eager_projection": projection(delta_cc, direction, torch),
                "reference_r_candidate_m_residual_projection": projection(delta_rc, direction, torch),
                "candidate_r_reference_m_residual_projection": projection(delta_cr, direction, torch),
                "reference_r_reference_m_residual_projection": projection(delta_rr, direction, torch),
                "r_shapley_removal_projection": projection(r_shapley, direction, torch),
                "m_shapley_removal_projection": projection(m_shapley, direction, torch),
                "total_go_path_removal_projection": projection(total_removal, direction, torch),
                "shapley_closure_max_abs": float(
                    (r_shapley + m_shapley - total_removal).abs().max()
                ),
                "candidate_restoration_sham_max_abs": float(
                    (tile_sham.float() - tile_cc.float()).abs().max()
                ),
                "candidate_go_sham_max_abs": float(
                    (obs_sham["Go"].float() - obs_cc["Go"].float()).abs().max()
                ),
                "candidate_r_l2": float(obs_cc["R"].float().norm()),
                "reference_r_l2": float(reference["R"].float().norm()),
                "candidate_go_l2": float(obs_cc["Go"].float().norm()),
                "reference_go_l2": float(reference["Go"].float().norm()),
                "rc_go_l2": float(obs_rc["Go"].float().norm()),
                "cr_go_l2": float(obs_cr["Go"].float().norm()),
                "rr_go_matches_reference": bool(torch.equal(obs_rr["Go"], reference["Go"])),
            }
        )
        print(
            json.dumps(
                {
                    "state": state_index,
                    "total": rows[-1]["candidate_minus_eager_projection"],
                    "R": rows[-1]["r_shapley_removal_projection"],
                    "M": rows[-1]["m_shapley_removal_projection"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    result = {
        "schema": "kernel-analyzer-l23-go-path-decomposition-v1",
        "status": "COMPLETE",
        "target_layer": args.layer,
        "parameter": PARAMETER,
        "tile": {
            "rows": [ROWS.start, ROWS.stop],
            "columns": [COLUMNS.start, COLUMNS.stop],
        },
        "checkpoint_step": args.step,
        "state_indices": state_indices,
        "equation": f"Go_{args.layer} = R_{args.layer} + M_{args.layer}, where R is direct residual cotangent and M is post_attention_layernorm+MLP VJP cotangent",
        "binding": {
            "consumer": f"mm_{consumer_mm}: D_{args.layer} = Go_{args.layer} @ Wo_{args.layer}",
            "actual_go_construction_kernel": target_symbol,
            "same_symbol_zero_based_ordinal": target_ordinal,
            "generated_source_sha256": source_sha256,
            "selection": f"current-compile source interval mm_{consumer_mm - 2} -> mm_{consumer_mm} plus mutating RMSNorm-VJP residual-add signature",
            "candidate_values_used_to_select_boundary": False,
        },
        "hybrids": {
            "CC": "candidate direct residual R and candidate fused M path",
            "RC": "reference direct residual R and candidate fused M path",
            "CR": "candidate direct residual R and materialized reference M path",
            "RR": "reference Go from exact eager F+B",
        },
        "rows": rows,
        "tensor_values_saved": False,
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"output": str(output_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
