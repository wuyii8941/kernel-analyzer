#!/usr/bin/env python
"""Batch barrier-controlled repair/injection for named Qwen3 operators."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from qwen3_lm_head_operator_pilot_v0_1 import (
    CANDIDATE_ANCHOR,
    EAGER_ANCHOR,
    metrics,
    move_tree,
    tensor_sha256,
)


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

    class FixedBoundary(nn.Module):
        def __init__(self, target_id: str, module: Any, compiled: Any):
            super().__init__()
            self.target_id = target_id
            self.module = module
            object.__setattr__(self, "_compiled_callable", compiled)
            self.mode = "eager"
            self.calls = {"eager": 0, "compiled": 0}

        @torch.compiler.disable
        def forward(self, *values: Any, **kwargs: Any) -> Any:
            self.calls[self.mode] += 1
            if self.mode == "compiled":
                return self._compiled_callable(*values, **kwargs)
            if self.mode == "eager":
                return self.module(*values, **kwargs)
            raise RuntimeError(f"unknown target mode {self.mode}")

    boundaries: dict[str, FixedBoundary] = {}

    def install(parent: Any, attribute: str, target_id: str) -> None:
        module = getattr(parent, attribute)
        compiled = tracked_compile(module, f"target.{target_id}")
        boundary = FixedBoundary(target_id, module, compiled)
        setattr(parent, attribute, boundary)
        boundaries[target_id] = boundary

    for layer_index in (0, 14, 27):
        layer = raw.model.layers[layer_index]
        for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
            install(layer.self_attn, role, f"linear.{role}.layer{layer_index}")
        for role in ("gate_proj", "up_proj", "down_proj"):
            install(layer.mlp, role, f"linear.{role}.layer{layer_index}")
        install(layer, "post_attention_layernorm", f"norm.post_attention.layer{layer_index}")
        install(layer.self_attn, "q_norm", f"norm.q_norm.layer{layer_index}")
        install(layer.self_attn, "k_norm", f"norm.k_norm.layer{layer_index}")
    install(raw.model.layers[0], "input_layernorm", "norm.input.layer0")
    for layer_index in (1, 14, 27):
        install(raw.model.layers[layer_index], "input_layernorm", f"norm.input.layer{layer_index}")
    install(raw.model, "embed_tokens", "embedding.token")

    if len(boundaries) != 35:
        raise RuntimeError(f"expected 35 target boundaries, found {len(boundaries)}")

    compiled_body = tracked_compile(raw.model, "outer_body_with_35_fixed_boundaries")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")

    def set_modes(default: str, exception: str | None = None, exception_mode: str | None = None) -> None:
        for target_id, boundary in boundaries.items():
            boundary.mode = exception_mode if target_id == exception else default

    def packed(value: dict[str, Any]) -> tuple[Any, Any, Any, int]:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        return input_ids, attention_mask, completion_ids, completion_ids.size(1) + 1

    def score(value: dict[str, Any], body: Any, head: Any) -> Any:
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
            outputs = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = outputs.last_hidden_state[:, -keep:, :]
            logits = head(hidden)
        logits = logits.float()[:, :-1, :]
        logits = logits[:, -completion_ids.size(1) :, :]
        return selective_log_softmax(logits, completion_ids)

    def observe(body: Any, head: Any) -> tuple[Any, dict[str, Any]]:
        repeats = []
        hashes = []
        for _ in range(2):
            value = score(inputs, body, head)
            detached = value.detach().float().cpu()
            repeats.append(detached)
            hashes.append(tensor_sha256(detached))
            del value
            gc.collect()
        return repeats[0], {
            "sha256": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "repeat_max_abs": float((repeats[1] - repeats[0]).abs().max().item()),
        }

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
    eager_baseline, eager_record = observe(raw.model, raw.lm_head)
    set_modes("compiled")
    candidate_baseline, candidate_record = observe(compiled_body, compiled_head)
    compile_count_before_contrasts = compile_audit["outer_body_with_35_fixed_boundaries"]["backend_compiles"]

    targets: dict[str, Any] = {}
    all_repeat_exact = eager_record["repeat_exact"] and candidate_record["repeat_exact"]
    target_mode_integrity = True
    for target_id, boundary in boundaries.items():
        set_modes("eager", target_id, "compiled")
        before = dict(boundary.calls)
        injected, injection_record = observe(raw.model, raw.lm_head)
        after = dict(boundary.calls)
        injection_calls = {key: after[key] - before[key] for key in before}

        set_modes("compiled", target_id, "eager")
        before = dict(boundary.calls)
        repaired, repair_record = observe(compiled_body, compiled_head)
        after = dict(boundary.calls)
        repair_calls = {key: after[key] - before[key] for key in before}
        target_mode_integrity = target_mode_integrity and injection_calls == {"eager": 0, "compiled": 2}
        target_mode_integrity = target_mode_integrity and repair_calls == {"eager": 2, "compiled": 0}
        all_repeat_exact = all_repeat_exact and injection_record["repeat_exact"] and repair_record["repeat_exact"]
        targets[target_id] = {
            "status": "BARRIER_CONDITIONED",
            "injection_arm": injection_record,
            "repair_arm": repair_record,
            "injection_calls": injection_calls,
            "repair_calls": repair_calls,
            "injection_effect": metrics(torch, eager_baseline, injected),
            "repair_effect": metrics(torch, candidate_baseline, repaired),
            "compiled_context_target_effect": metrics(torch, repaired, candidate_baseline),
        }
        del injected, repaired
        gc.collect()

    compile_count_after_contrasts = compile_audit["outer_body_with_35_fixed_boundaries"]["backend_compiles"]
    gates = {
        "target_count_35": len(targets) == 35,
        "eager_baseline_anchor_exact": eager_record["sha256"][0] == EAGER_ANCHOR,
        "all_repeats_exact": all_repeat_exact,
        "all_target_modes_executed_exactly": target_mode_integrity,
        "no_outer_recompile_across_target_modes": compile_count_before_contrasts == compile_count_after_contrasts,
    }
    valid = all(gates.values())
    candidate_anchor_exact = candidate_record["sha256"][0] == CANDIDATE_ANCHOR
    for row in targets.values():
        row["status"] = "BARRIER_CONDITIONED" if valid else "INVALID_TREATMENT"
    nonzero_injection = sum(row["injection_effect"]["l2"] > 0 for row in targets.values())
    nonzero_repair = sum(row["repair_effect"]["l2"] > 0 for row in targets.values())
    payload = {
        "schema_version": "forkcert.qwen3-named-operator-barrier-batch.v0.1",
        "status": "VALID_BARRIER_CONDITIONED_BATCH" if valid else "INVALID_TREATMENT_BATCH",
        "transport_to_original_candidate": "VALID" if valid and candidate_anchor_exact else "NOT_ESTABLISHED",
        "state": "heldout-transport-B-step29",
        "anchors": {"eager": EAGER_ANCHOR, "candidate": CANDIDATE_ANCHOR},
        "gates": gates,
        "baseline_arms": {"reference": eager_record, "barrier_candidate": candidate_record},
        "candidate_anchor_exact": candidate_anchor_exact,
        "barrier_total_effect": metrics(torch, eager_baseline, candidate_baseline),
        "targets": targets,
        "summary": {
            "targets": len(targets),
            "nonzero_injection_effects": nonzero_injection,
            "nonzero_repair_effects": nonzero_repair,
        },
        "compile_count_before_contrasts": compile_count_before_contrasts,
        "compile_count_after_contrasts": compile_count_after_contrasts,
        "compile_audit": compile_audit,
        "claim_limits": [
            "barrier-conditioned selected-invocation effects",
            "no original-candidate root-cause transport unless candidate anchor is exact",
            "selected matched state only",
            "named module invocation level, not constituent primitive level",
            "forward selected-token observable only",
            "implementation-relative, not correctness",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "transport": payload["transport_to_original_candidate"], "gates": gates, "summary": payload["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
