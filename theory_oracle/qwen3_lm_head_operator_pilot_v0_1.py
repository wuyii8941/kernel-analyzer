#!/usr/bin/env python
"""Fail-closed Qwen3 final-linear operator feasibility pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


EAGER_ANCHOR = "742468b7b182ea8e70fec4733f702dbcc71ebb64fa3f4aec5e9fbc2450a29806"
CANDIDATE_ANCHOR = "1107b4ac9c2662b34572cee3b4b4e1bf454a4b6d0a6def0c427d84f9944a09f2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def tensor_sha256(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def move_tree(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    return value


def metrics(torch: Any, left: Any, right: Any) -> dict[str, float]:
    delta = right.detach().float() - left.detach().float()
    return {
        "mean_signed": float(delta.mean().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
    }


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
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM
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

    compiled_body = tracked_compile(raw.model, "body")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")

    def packed(value: dict[str, Any]) -> tuple[Any, Any, Any, int]:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        return input_ids, attention_mask, completion_ids, completion_ids.size(1) + 1

    def whole_score(value: dict[str, Any]) -> Any:
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH):
            outputs = wrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=keep,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -completion_ids.size(1) :, :]
            return selective_log_softmax(logits, completion_ids)

    def split_score(value: dict[str, Any], body: Any, head: Any) -> Any:
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
            outputs = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = outputs.last_hidden_state[:, -keep:, :]
            logits = head(hidden)
        # Accelerate's model-level mixed-precision wrapper converts model outputs
        # to fp32 after the autocast region. Reproduce that boundary explicitly.
        logits = logits.float()
        logits = logits[:, :-1, :]
        logits = logits[:, -completion_ids.size(1) :, :]
        return selective_log_softmax(logits, completion_ids)

    arms: dict[str, Callable[[dict[str, Any]], Any]] = {
        "whole_eager": whole_score,
        "split_EE": lambda value: split_score(value, raw.model, raw.lm_head),
        "split_CC": lambda value: split_score(value, compiled_body, compiled_head),
        "repair_CE": lambda value: split_score(value, compiled_body, raw.lm_head),
        "injection_EC": lambda value: split_score(value, raw.model, compiled_head),
    }

    # Realize compiler specialization using the frozen ordered input history.
    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = arms["split_CC"](history)
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
        "split_reference_preserved": arm_records["split_EE"]["sha256"][0] == arm_records["whole_eager"]["sha256"][0],
        "split_candidate_anchor_exact": arm_records["split_CC"]["sha256"][0] == CANDIDATE_ANCHOR,
        "all_repeats_exact": all(record["repeat_exact"] for record in arm_records.values()),
    }
    attribution_valid = all(gates.values())
    contrasts = {
        "whole_eager_to_split_EE": metrics(torch, values["whole_eager"], values["split_EE"]),
        "total_split_EE_to_CC": metrics(torch, values["split_EE"], values["split_CC"]),
        "repair_CC_to_CE": metrics(torch, values["split_CC"], values["repair_CE"]),
        "repair_residual_EE_to_CE": metrics(torch, values["split_EE"], values["repair_CE"]),
        "injection_EE_to_EC": metrics(torch, values["split_EE"], values["injection_EC"]),
        "head_effect_on_eager_body_EE_to_EC": metrics(torch, values["split_EE"], values["injection_EC"]),
        "head_effect_on_compiled_body_CE_to_CC": metrics(torch, values["repair_CE"], values["split_CC"]),
    }
    payload = {
        "schema_version": "forkcert.qwen3-lm-head-operator-pilot.v0.1",
        "status": "VALID_SELECTED_STATE_OPERATOR_ATTRIBUTION" if attribution_valid else "INVALID_FOR_ORIGINAL_CANDIDATE_ATTRIBUTION",
        "subject": "Qwen3-0.6B final lm_head Linear invocation",
        "state": "heldout-transport-B-step29",
        "anchors": {"eager": EAGER_ANCHOR, "candidate": CANDIDATE_ANCHOR},
        "gates": gates,
        "arms": arm_records,
        "contrasts": contrasts,
        "compile_audit": compile_audit,
        "claim_limits": [
            "selected matched state only",
            "implementation-relative, not correctness",
            "invalid for original-candidate operator attribution if any gate fails",
            "forward selected-token observable only; update propagation not measured",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
