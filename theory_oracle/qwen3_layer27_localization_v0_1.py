#!/usr/bin/env python
"""Fail-closed localization of the Qwen3 scorer discrepancy at decoder layer 27."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from qwen3_lm_head_operator_pilot_v0_1 import (
    CANDIDATE_ANCHOR,
    EAGER_ANCHOR,
    metrics,
    move_tree,
    tensor_sha256,
)


TARGET_LAYER = 27


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
    from transformers.masking_utils import create_causal_mask
    from trl.trainer.grpo_trainer import selective_log_softmax

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 64

    metadata = json.loads((snapshot_dir / "forkcert_transition_snapshot.json").read_text())
    target = Path(metadata["target_minibatch_path"])
    if not target.is_file():
        target = snapshot_dir / "compiler_history" / target.name
    inputs = move_tree(torch.load(target, map_location="cpu", weights_only=False), "cuda")
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
    if not (hasattr(wrapped, "_original_forward") and hasattr(wrapped.forward, "__wrapped__")):
        raise RuntimeError("Accelerate FP16 forward wrapper was not reproduced")
    raw = accelerator.unwrap_model(wrapped)
    if len(raw.model.layers) != 28:
        raise RuntimeError(f"expected 28 decoder layers, found {len(raw.model.layers)}")

    class Prefix(nn.Module):
        def __init__(self, body: Any):
            super().__init__()
            self.embed_tokens = body.embed_tokens
            self.layers = body.layers[:TARGET_LAYER]
            self.rotary_emb = body.rotary_emb
            self.config = body.config

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any, Any, Any, Any]:
            hidden = self.embed_tokens(input_ids)
            position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            cos, sin = self.rotary_emb(hidden, position_ids)
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_embeddings=(cos, sin),
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
            return hidden, mask, cos, sin, position_ids

    class TargetLayer(nn.Module):
        def __init__(self, layer: Any):
            super().__init__()
            self.layer = layer

        def forward(self, hidden: Any, mask: Any, cos: Any, sin: Any, position_ids: Any) -> Any:
            return self.layer(
                hidden,
                attention_mask=mask,
                position_embeddings=(cos, sin),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )

    class Tail(nn.Module):
        def __init__(self, root: Any):
            super().__init__()
            self.norm = root.model.norm
            self.head = root.lm_head

        def forward(self, hidden: Any, keep: int) -> Any:
            hidden = self.norm(hidden)
            return self.head(hidden[:, -keep:, :])

    prefix = Prefix(raw.model)
    target_layer = TargetLayer(raw.model.layers[TARGET_LAYER])
    tail = Tail(raw)
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

    compiled_prefix = tracked_compile(prefix, "layers_0_26_prefix")
    compiled_target = tracked_compile(target_layer, "layer_27")
    compiled_tail = tracked_compile(tail, "final_norm_and_lm_head_tail")

    def packed(value: dict[str, Any]) -> tuple[Any, Any, Any, int]:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        return input_ids, attention_mask, completion_ids, completion_ids.size(1) + 1

    def whole_score(value: dict[str, Any]) -> Any:
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH):
            outputs = wrapped(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -completion_ids.size(1) :, :]
            return selective_log_softmax(logits, completion_ids)

    def split_score(value: dict[str, Any], pfx: Any, layer: Any, post: Any) -> Any:
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
            hidden, mask, cos, sin, position_ids = pfx(input_ids, attention_mask)
            hidden = layer(hidden, mask, cos, sin, position_ids)
            logits = post(hidden, keep)
        logits = logits.float()[:, :-1, :]
        logits = logits[:, -completion_ids.size(1) :, :]
        return selective_log_softmax(logits, completion_ids)

    arms: dict[str, Callable[[dict[str, Any]], Any]] = {
        "whole_eager": whole_score,
        "split_EEE": lambda value: split_score(value, prefix, target_layer, tail),
        "split_CCC": lambda value: split_score(value, compiled_prefix, compiled_target, compiled_tail),
        "repair_CEC": lambda value: split_score(value, compiled_prefix, target_layer, compiled_tail),
        "injection_ECE": lambda value: split_score(value, prefix, compiled_target, tail),
    }
    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = arms["split_CCC"](history)
        del value, history
        gc.collect()

    values: dict[str, Any] = {}
    arm_records: dict[str, Any] = {}
    for name, function in arms.items():
        repeats = []
        hashes = []
        for _ in range(2):
            value = function(inputs)
            detached = value.detach().float().cpu()
            repeats.append(detached)
            hashes.append(tensor_sha256(detached))
            del value
            gc.collect()
        values[name] = repeats[0]
        arm_records[name] = {
            "sha256": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "repeat_max_abs": float((repeats[1] - repeats[0]).abs().max().item()),
        }
    gates = {
        "whole_eager_anchor_exact": arm_records["whole_eager"]["sha256"][0] == EAGER_ANCHOR,
        "split_reference_preserved": arm_records["split_EEE"]["sha256"][0] == arm_records["whole_eager"]["sha256"][0],
        "split_candidate_anchor_exact": arm_records["split_CCC"]["sha256"][0] == CANDIDATE_ANCHOR,
        "all_repeats_exact": all(record["repeat_exact"] for record in arm_records.values()),
    }
    contrasts = {
        "total_EEE_to_CCC": metrics(torch, values["split_EEE"], values["split_CCC"]),
        "layer27_repair_CCC_to_CEC": metrics(torch, values["split_CCC"], values["repair_CEC"]),
        "layer27_repair_residual_EEE_to_CEC": metrics(torch, values["split_EEE"], values["repair_CEC"]),
        "layer27_injection_EEE_to_ECE": metrics(torch, values["split_EEE"], values["injection_ECE"]),
        "layer27_effect_on_eager_prefix_EEE_to_ECE": metrics(torch, values["split_EEE"], values["injection_ECE"]),
        "layer27_effect_on_compiled_prefix_CEC_to_CCC": metrics(torch, values["repair_CEC"], values["split_CCC"]),
    }
    valid = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-layer27-localization.v0.1",
        "status": "VALID_SELECTED_STATE_LAYER_LOCALIZATION" if valid else "INVALID_FOR_ORIGINAL_CANDIDATE_LOCALIZATION",
        "subject": "Qwen3-0.6B decoder layer 27 composite module",
        "subject_is_operator": False,
        "state": "heldout-transport-B-step29",
        "anchors": {"eager": EAGER_ANCHOR, "candidate": CANDIDATE_ANCHOR},
        "gates": gates,
        "arms": arm_records,
        "contrasts": contrasts,
        "compile_audit": compile_audit,
        "claim_limits": [
            "composite-layer localization, not operator attribution",
            "selected matched state only",
            "implementation-relative, not correctness",
            "invalid for original-candidate localization if any gate fails",
            "forward selected-token observable only",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
