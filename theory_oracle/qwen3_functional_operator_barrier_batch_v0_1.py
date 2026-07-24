#!/usr/bin/env python
"""Barrier batch for Qwen3 functional operators at layers 0, 14 and 27."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

from qwen3_lm_head_operator_pilot_v0_1 import (
    CANDIDATE_ANCHOR,
    EAGER_ANCHOR,
    metrics,
    move_tree,
    tensor_sha256,
)


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
    import torch.nn.functional as F
    from accelerate import Accelerator
    from torch import nn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM
    from transformers.integrations.sdpa_attention import repeat_kv, use_gqa_in_sdpa
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
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    accelerator = Accelerator(mixed_precision="fp16")
    wrapped = accelerator.prepare_model(model)
    raw = accelerator.unwrap_model(wrapped)

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

    class AddOp(nn.Module):
        def forward(self, left: Any, right: Any) -> Any:
            return left + right

    class MulOp(nn.Module):
        def forward(self, left: Any, right: Any) -> Any:
            return left * right

    class RotaryOp(nn.Module):
        def forward(self, query: Any, key: Any, cos: Any, sin: Any) -> Any:
            return apply_rotary_pos_emb(query, key, cos, sin)

    class SDPAOp(nn.Module):
        def __init__(self, attention: Any):
            super().__init__()
            self.num_key_value_groups = attention.num_key_value_groups
            self.scaling = attention.scaling
            self.dropout = attention.attention_dropout
            self.is_causal = attention.is_causal

        def forward(self, query: Any, key: Any, value: Any, attention_mask: Any) -> Any:
            sdpa_kwargs = {}
            if self.num_key_value_groups > 1:
                if not use_gqa_in_sdpa(attention_mask, key, value):
                    key = repeat_kv(key, self.num_key_value_groups)
                    value = repeat_kv(value, self.num_key_value_groups)
                else:
                    sdpa_kwargs = {"enable_gqa": True}
            is_causal = query.shape[2] > 1 and attention_mask is None and self.is_causal
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scaling,
                is_causal=is_causal,
                **sdpa_kwargs,
            )
            return output.transpose(1, 2).contiguous()

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
            if self.mode == "compiled":
                return self._compiled_callable(*values)
            if self.mode == "eager":
                return self.module(*values)
            raise RuntimeError(f"unknown mode {self.mode}")

    boundaries: dict[str, FixedBoundary] = {}

    def boundary(target_id: str, module: Any) -> FixedBoundary:
        item = FixedBoundary(target_id, module, tracked_compile(module, f"target.{target_id}"))
        boundaries[target_id] = item
        return item

    def instrument_mlp(mlp: Any, layer_index: int) -> None:
        mlp.add_module("_forkcert_silu", boundary(f"mlp.silu.layer{layer_index}", mlp.act_fn))
        mlp.add_module("_forkcert_gate_mul", boundary(f"mlp.gate_multiply.layer{layer_index}", MulOp()))

        def forward(self: Any, x: Any) -> Any:
            gate = self.gate_proj(x)
            activated = self._forkcert_silu(gate)
            up = self.up_proj(x)
            return self.down_proj(self._forkcert_gate_mul(activated, up))
        mlp.forward = types.MethodType(forward, mlp)

    def instrument_attention(attention: Any, layer_index: int) -> None:
        attention.add_module("_forkcert_rotary", boundary(f"attention.rotary.layer{layer_index}", RotaryOp()))
        attention.add_module("_forkcert_sdpa", boundary(f"attention.sdpa.layer{layer_index}", SDPAOp(attention)))

        def forward(
            self: Any,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any,
            past_key_values: Any = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = self._forkcert_rotary(query_states, key_states, cos, sin)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            attn_output = self._forkcert_sdpa(query_states, key_states, value_states, attention_mask)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output), None
        attention.forward = types.MethodType(forward, attention)

    def instrument_layer(layer: Any, layer_index: int) -> None:
        layer.add_module("_forkcert_attention_add", boundary(f"residual.attention.layer{layer_index}", AddOp()))
        layer.add_module("_forkcert_mlp_add", boundary(f"residual.mlp.layer{layer_index}", AddOp()))
        instrument_attention(layer.self_attn, layer_index)
        instrument_mlp(layer.mlp, layer_index)

        def forward(
            self: Any,
            hidden_states: Any,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            use_cache: Any = False,
            position_embeddings: Any = None,
            **kwargs: Any,
        ) -> Any:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = self._forkcert_attention_add(residual, hidden_states)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            return self._forkcert_mlp_add(residual, hidden_states)
        layer.forward = types.MethodType(forward, layer)

    for layer_index in LAYERS:
        instrument_layer(raw.model.layers[layer_index], layer_index)
    if len(boundaries) != 18:
        raise RuntimeError(f"expected 18 boundaries, found {len(boundaries)}")

    compiled_body = tracked_compile(raw.model, "outer_body_with_18_fixed_boundaries")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")

    def set_modes(default: str, exception: str | None = None, exception_mode: str | None = None) -> None:
        for target_id, item in boundaries.items():
            item.mode = exception_mode if target_id == exception else default

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

    set_modes("compiled")
    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = score(history, compiled_body, compiled_head)
        del value, history
        gc.collect()

    set_modes("eager")
    eager, eager_record = observe(raw.model, raw.lm_head)
    set_modes("compiled")
    candidate, candidate_record = observe(compiled_body, compiled_head)
    compile_before = compile_audit["outer_body_with_18_fixed_boundaries"]["backend_compiles"]
    targets = {}
    all_exact = eager_record["repeat_exact"] and candidate_record["repeat_exact"]
    modes_exact = True
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
            "injection_arm": injection_record,
            "repair_arm": repair_record,
            "injection_calls": injection_calls,
            "repair_calls": repair_calls,
            "injection_effect": metrics(torch, eager, injected),
            "repair_effect": metrics(torch, candidate, repaired),
            "compiled_context_target_effect": metrics(torch, repaired, candidate),
        }
        del injected, repaired
        gc.collect()
    compile_after = compile_audit["outer_body_with_18_fixed_boundaries"]["backend_compiles"]
    gates = {
        "target_count_18": len(targets) == 18,
        "eager_baseline_anchor_exact": eager_record["sha256"][0] == EAGER_ANCHOR,
        "all_repeats_exact": all_exact,
        "all_target_modes_executed_exactly": modes_exact,
        "no_outer_recompile_across_target_modes": compile_before == compile_after,
    }
    valid = all(gates.values())
    for row in targets.values():
        row["status"] = "BARRIER_CONDITIONED" if valid else "INVALID_TREATMENT"
    candidate_anchor_exact = candidate_record["sha256"][0] == CANDIDATE_ANCHOR
    payload = {
        "schema_version": "forkcert.qwen3-functional-operator-barrier-batch.v0.1",
        "status": "VALID_BARRIER_CONDITIONED_BATCH" if valid else "INVALID_TREATMENT_BATCH",
        "transport_to_original_candidate": "VALID" if valid and candidate_anchor_exact else "NOT_ESTABLISHED",
        "state": "heldout-transport-B-step29",
        "gates": gates,
        "anchors": {"eager": EAGER_ANCHOR, "candidate": CANDIDATE_ANCHOR},
        "candidate_anchor_exact": candidate_anchor_exact,
        "baseline_arms": {"reference": eager_record, "barrier_candidate": candidate_record},
        "barrier_total_effect": metrics(torch, eager, candidate),
        "targets": targets,
        "summary": {
            "targets": len(targets),
            "nonzero_injection_effects": sum(row["injection_effect"]["l2"] > 0 for row in targets.values()),
            "nonzero_repair_effects": sum(row["repair_effect"]["l2"] > 0 for row in targets.values()),
        },
        "compile_count_before_contrasts": compile_before,
        "compile_count_after_contrasts": compile_after,
        "compile_audit": compile_audit,
        "claim_limits": [
            "barrier-conditioned selected-invocation effects",
            "SDPA is composite and does not isolate its decomposed bmm/softmax/bmm primitives",
            "selected state and forward selected-token observable only",
            "implementation-relative, not correctness",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "transport": payload["transport_to_original_candidate"], "gates": gates, "summary": payload["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
