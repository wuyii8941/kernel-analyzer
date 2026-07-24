#!/usr/bin/env python
"""Fail-closed causal-mask repair/injection for the frozen Qwen3 forward."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import types
from pathlib import Path
from typing import Any

from qwen3_lm_head_operator_pilot_v0_1 import CANDIDATE_ANCHOR, EAGER_ANCHOR, metrics, move_tree, tensor_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
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
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast
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

    class MaskOp(nn.Module):
        def __init__(self, config: Any):
            super().__init__()
            object.__setattr__(self, "config", config)
        def forward(self, embeds: Any, attention_mask: Any, position_ids: Any) -> Any:
            return create_causal_mask(
                config=self.config,
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )

    mask_op = MaskOp(raw.model.config)
    compiled_mask = tracked_compile(mask_op, "target.causal_mask")

    class FixedBoundary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = mask_op
            object.__setattr__(self, "_compiled_callable", compiled_mask)
            self.mode = "eager"
            self.calls = {"eager": 0, "compiled": 0}
            self.output_types = {"eager": [], "compiled": []}
        @torch.compiler.disable
        def forward(self, *values: Any) -> Any:
            self.calls[self.mode] += 1
            result = self._compiled_callable(*values) if self.mode == "compiled" else self.module(*values)
            self.output_types[self.mode].append("None" if result is None else type(result).__name__)
            return result

    boundary = FixedBoundary()
    raw.model.add_module("_forkcert_causal_mask", boundary)

    def model_forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: Any = None,
        **kwargs: Any,
    ) -> Any:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = self._forkcert_causal_mask(inputs_embeds, attention_mask, position_ids)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)

    raw.model.forward = types.MethodType(model_forward, raw.model)
    compiled_body = tracked_compile(raw.model, "outer_body_with_mask_boundary")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")

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

    boundary.mode = "compiled"
    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = score(history, compiled_body, compiled_head)
        del value, history
        gc.collect()

    boundary.mode = "eager"
    eager, eager_record = observe(raw.model, raw.lm_head)
    boundary.mode = "compiled"
    candidate, candidate_record = observe(compiled_body, compiled_head)
    compile_before = compile_audit["outer_body_with_mask_boundary"]["backend_compiles"]

    boundary.mode = "compiled"
    before = dict(boundary.calls)
    injected, injection_record = observe(raw.model, raw.lm_head)
    injection_calls = {key: boundary.calls[key] - before[key] for key in before}
    boundary.mode = "eager"
    before = dict(boundary.calls)
    repaired, repair_record = observe(compiled_body, compiled_head)
    repair_calls = {key: boundary.calls[key] - before[key] for key in before}
    compile_after = compile_audit["outer_body_with_mask_boundary"]["backend_compiles"]

    gates = {
        "instrumented_eager_anchor_exact": eager_record["sha256"][0] == original_hashes[0] == EAGER_ANCHOR,
        "all_repeats_exact": all(row["repeat_exact"] for row in (eager_record, candidate_record, injection_record, repair_record)),
        "target_modes_executed_exactly": injection_calls == {"eager": 0, "compiled": 2} and repair_calls == {"eager": 2, "compiled": 0},
        "no_outer_recompile_across_mask_modes": compile_before == compile_after,
    }
    valid = all(gates.values())
    candidate_anchor_exact = candidate_record["sha256"][0] == CANDIDATE_ANCHOR
    payload = {
        "schema_version": "forkcert.qwen3-causal-mask-barrier.v0.1",
        "status": "VALID_BARRIER_CONDITIONED_OPERATOR_EFFECT" if valid else "INVALID_TREATMENT",
        "transport_to_original_candidate": "VALID" if valid and candidate_anchor_exact else "NOT_ESTABLISHED",
        "state": "heldout-transport-B-step29",
        "gates": gates,
        "candidate_anchor_exact": candidate_anchor_exact,
        "original_eager_hashes": original_hashes,
        "arms": {"reference": eager_record, "candidate": candidate_record, "injection": injection_record, "repair": repair_record},
        "target_call_deltas": {"injection": injection_calls, "repair": repair_calls},
        "target_output_types": boundary.output_types,
        "contrasts": {
            "injection": metrics(torch, eager, injected),
            "repair": metrics(torch, candidate, repaired),
            "barrier_total": metrics(torch, eager, candidate),
        },
        "compile_counts": {"before_mode_contrasts": compile_before, "after_mode_contrasts": compile_after},
        "compile_audit": compile_audit,
        "coverage_credit": "BARRIER_CONDITIONED" if valid else "INVALID_TREATMENT",
        "claim_limits": ["high-level causal mask invocation only", "selected state only", "no primitive or correctness attribution"],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "transport": payload["transport_to_original_candidate"], "gates": gates, "output_types": boundary.output_types, "contrasts": payload["contrasts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
