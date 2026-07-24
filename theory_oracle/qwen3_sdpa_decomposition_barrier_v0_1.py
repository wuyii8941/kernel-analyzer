#!/usr/bin/env python
"""Fail-closed decomposition of Qwen3 math-SDPA into qk/softmax/pv targets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import types
from pathlib import Path
from typing import Any

from qwen3_lm_head_operator_pilot_v0_1 import CANDIDATE_ANCHOR, EAGER_ANCHOR, metrics, move_tree, tensor_sha256


LAYERS = (0, 14, 27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from accelerate import Accelerator
    from torch import nn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM
    from transformers.integrations.sdpa_attention import repeat_kv
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    from trl.trainer.grpo_trainer import selective_log_softmax

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 128

    metadata = json.loads((snapshot_dir / "forkcert_transition_snapshot.json").read_text())
    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot_dir / "compiler_history" / target_path.name
    inputs = move_tree(torch.load(target_path, map_location="cpu", weights_only=False), "cuda")
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    accelerator = Accelerator(mixed_precision="fp16")
    wrapped = accelerator.prepare_model(model)
    raw = accelerator.unwrap_model(wrapped)

    def packed(value: dict[str, Any]) -> tuple[Any, Any, Any, int]:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        return input_ids, attention_mask, completion_ids, completion_ids.size(1) + 1

    def score(value: dict[str, Any], body: Any, head: Any) -> Any:
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
            outputs = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = head(outputs.last_hidden_state[:, -keep:, :])
        logits = logits.float()[:, :-1, :]
        return selective_log_softmax(logits[:, -completion_ids.size(1):, :], completion_ids)

    original_values = []
    for _ in range(2):
        value = score(inputs, raw.model, raw.lm_head)
        original_values.append(value.detach().float().cpu())
        del value
        gc.collect()
    original_hashes = [tensor_sha256(value) for value in original_values]

    compile_audit: dict[str, dict[str, Any]] = {}

    def tracked_compile(module: Any, name: str) -> Any:
        from torch._dynamo.backends.registry import lookup_backend
        inductor = lookup_backend("inductor")
        audit = {"backend_compiles": 0, "runtime_invocations": 0, "graph_hashes": [], "graph_nodes": []}
        compile_audit[name] = audit

        def backend(graph_module: Any, example_inputs: list[Any]):
            audit["backend_compiles"] += 1
            audit["graph_hashes"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
            audit["graph_nodes"].append(sum(1 for _ in graph_module.graph.nodes))
            compiled = inductor(graph_module, example_inputs)
            def counted(*values: Any):
                audit["runtime_invocations"] += 1
                return compiled(*values)
            return counted
        return torch.compile(module, backend=backend)

    class QKBMM(nn.Module):
        def __init__(self, scaling: float):
            super().__init__()
            self.sqrt_scale = math.sqrt(scaling)
        def forward(self, query: Any, key: Any) -> Any:
            q = query.float() * self.sqrt_scale
            k = key.float().transpose(-2, -1) * self.sqrt_scale
            return torch.matmul(q, k)

    class SafeSoftmax(nn.Module):
        def forward(self, scores: Any, mask: Any) -> Any:
            if mask is None:
                query_length, key_length = scores.shape[-2:]
                causal = torch.ones(
                    (query_length, key_length),
                    dtype=torch.bool,
                    device=scores.device,
                ).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            else:
                scores = scores + mask
            return torch.ops.aten._safe_softmax.default(scores, -1, None)

    class PVBMM(nn.Module):
        def forward(self, probabilities: Any, value: Any) -> Any:
            return torch.matmul(probabilities, value.float()).to(value.dtype)

    class FixedBoundary(nn.Module):
        def __init__(self, target_id: str, module: Any, compiled: Any):
            super().__init__()
            self.target_id = target_id
            self.module = module
            object.__setattr__(self, "_compiled_callable", compiled)
            self.mode = "eager"
            self.calls = {"eager": 0, "compiled": 0}
        @torch.compiler.disable
        def forward(self, *values: Any) -> Any:
            self.calls[self.mode] += 1
            return self._compiled_callable(*values) if self.mode == "compiled" else self.module(*values)

    boundaries: dict[str, FixedBoundary] = {}
    def make_boundary(target_id: str, module: Any) -> FixedBoundary:
        item = FixedBoundary(target_id, module, tracked_compile(module, f"target.{target_id}"))
        boundaries[target_id] = item
        return item

    def instrument_attention(attention: Any, layer_index: int) -> None:
        attention.add_module("_forkcert_qk", make_boundary(f"attention.qk_bmm.layer{layer_index}", QKBMM(attention.scaling)))
        attention.add_module("_forkcert_softmax", make_boundary(f"attention.softmax.layer{layer_index}", SafeSoftmax()))
        attention.add_module("_forkcert_pv", make_boundary(f"attention.pv_bmm.layer{layer_index}", PVBMM()))
        def forward(self: Any, hidden_states: Any, position_embeddings: Any, attention_mask: Any, past_key_values: Any = None, **kwargs: Any) -> tuple[Any, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            if past_key_values is not None:
                key, value = past_key_values.update(key, value, self.layer_idx)
            if self.num_key_value_groups > 1:
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)
            scores = self._forkcert_qk(query, key)
            probabilities = self._forkcert_softmax(scores, attention_mask)
            output = self._forkcert_pv(probabilities, value).transpose(1, 2).contiguous()
            output = output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(output), None
        attention.forward = types.MethodType(forward, attention)

    for layer_index in LAYERS:
        instrument_attention(raw.model.layers[layer_index].self_attn, layer_index)
    if len(boundaries) != 9:
        raise RuntimeError(f"expected 9 boundaries, found {len(boundaries)}")

    def set_modes(default: str, exception: str | None = None, exception_mode: str | None = None) -> None:
        for target_id, item in boundaries.items():
            item.mode = exception_mode if target_id == exception else default

    def observe(body: Any, head: Any) -> tuple[Any, dict[str, Any]]:
        repeats, hashes = [], []
        for _ in range(2):
            value = score(inputs, body, head)
            detached = value.detach().float().cpu()
            repeats.append(detached)
            hashes.append(tensor_sha256(detached))
            del value
            gc.collect()
        return repeats[0], {"sha256": hashes, "repeat_exact": hashes[0] == hashes[1], "repeat_max_abs": float((repeats[1] - repeats[0]).abs().max().item())}

    set_modes("eager")
    reconstructed, reconstructed_record = observe(raw.model, raw.lm_head)
    reconstruction_exact = reconstructed_record["sha256"][0] == original_hashes[0] == EAGER_ANCHOR
    if not reconstruction_exact:
        payload = {
            "schema_version": "forkcert.qwen3-sdpa-decomposition-barrier.v0.1",
            "status": "VALID_INVALIDATION_REFERENCE_RECONSTRUCTION_CHANGED",
            "state": "heldout-transport-B-step29",
            "original_eager": {"sha256": original_hashes, "repeat_exact": original_hashes[0] == original_hashes[1]},
            "reconstructed_eager": reconstructed_record,
            "reconstruction_delta": metrics(torch, original_values[0], reconstructed),
            "coverage_credit": 0,
            "targets_attempted": sorted(boundaries),
            "claim": "no qk-bmm, softmax or pv-bmm attribution",
        }
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    compiled_body = tracked_compile(raw.model, "outer_body_with_9_fixed_boundaries")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")
    set_modes("compiled")
    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = score(history, compiled_body, compiled_head)
        del value, history
        gc.collect()
    candidate, candidate_record = observe(compiled_body, compiled_head)
    compile_before = compile_audit["outer_body_with_9_fixed_boundaries"]["backend_compiles"]
    targets = {}
    all_exact, modes_exact = reconstructed_record["repeat_exact"] and candidate_record["repeat_exact"], True
    for target_id, item in boundaries.items():
        set_modes("eager", target_id, "compiled")
        before = dict(item.calls)
        injected, injection_record = observe(raw.model, raw.lm_head)
        injection_calls = {key: item.calls[key] - before[key] for key in before}
        set_modes("compiled", target_id, "eager")
        before = dict(item.calls)
        repaired, repair_record = observe(compiled_body, compiled_head)
        repair_calls = {key: item.calls[key] - before[key] for key in before}
        modes_exact = modes_exact and injection_calls == {"eager": 0, "compiled": 2} and repair_calls == {"eager": 2, "compiled": 0}
        all_exact = all_exact and injection_record["repeat_exact"] and repair_record["repeat_exact"]
        targets[target_id] = {
            "status": "PENDING_BATCH_GATES",
            "injection_effect": metrics(torch, reconstructed, injected),
            "repair_effect": metrics(torch, candidate, repaired),
            "injection_arm": injection_record,
            "repair_arm": repair_record,
            "injection_calls": injection_calls,
            "repair_calls": repair_calls,
        }
        del injected, repaired
        gc.collect()
    compile_after = compile_audit["outer_body_with_9_fixed_boundaries"]["backend_compiles"]
    gates = {
        "reference_reconstruction_exact": reconstruction_exact,
        "target_count_9": len(targets) == 9,
        "all_repeats_exact": all_exact,
        "all_target_modes_executed_exactly": modes_exact,
        "no_outer_recompile_across_target_modes": compile_before == compile_after,
    }
    valid = all(gates.values())
    for row in targets.values():
        row["status"] = "BARRIER_CONDITIONED" if valid else "INVALID_TREATMENT"
    candidate_anchor_exact = candidate_record["sha256"][0] == CANDIDATE_ANCHOR
    payload = {
        "schema_version": "forkcert.qwen3-sdpa-decomposition-barrier.v0.1",
        "status": "VALID_BARRIER_CONDITIONED_BATCH" if valid else "INVALID_TREATMENT_BATCH",
        "transport_to_original_candidate": "VALID" if valid and candidate_anchor_exact else "NOT_ESTABLISHED",
        "state": "heldout-transport-B-step29",
        "gates": gates,
        "candidate_anchor_exact": candidate_anchor_exact,
        "original_eager": {"sha256": original_hashes, "repeat_exact": original_hashes[0] == original_hashes[1]},
        "reconstructed_eager": reconstructed_record,
        "barrier_candidate": candidate_record,
        "targets": targets,
        "summary": {
            "targets": len(targets),
            "nonzero_injection_effects": sum(row["injection_effect"]["l2"] > 0 for row in targets.values()),
            "nonzero_repair_effects": sum(row["repair_effect"]["l2"] > 0 for row in targets.values()),
        },
        "compile_audit": compile_audit,
        "claim_limits": ["barrier-conditioned decomposed SDPA operations", "selected state only", "no correctness claim"],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "transport": payload["transport_to_original_candidate"], "gates": gates, "summary": payload["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
