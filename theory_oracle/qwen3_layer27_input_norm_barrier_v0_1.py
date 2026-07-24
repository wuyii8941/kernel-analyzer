#!/usr/bin/env python
"""Barrier-controlled repair/injection for Qwen3 layer-27 input RMSNorm."""

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

    original_norm = raw.model.layers[TARGET_LAYER].input_layernorm
    compiled_norm = tracked_compile(original_norm, "target_input_rmsnorm")

    class FixedBoundary(nn.Module):
        def __init__(self, norm: Any, compiled: Any):
            super().__init__()
            self.norm = norm
            object.__setattr__(self, "_compiled_callable", compiled)
            self.mode = "eager"
            self.calls = {"eager": 0, "compiled": 0}

        @torch.compiler.disable
        def forward(self, hidden: Any) -> Any:
            self.calls[self.mode] += 1
            if self.mode == "compiled":
                return self._compiled_callable(hidden)
            if self.mode == "eager":
                return self.norm(hidden)
            raise RuntimeError(f"unknown boundary mode {self.mode}")

    boundary = FixedBoundary(original_norm, compiled_norm)
    raw.model.layers[TARGET_LAYER].input_layernorm = boundary
    compiled_body = tracked_compile(raw.model, "outer_body_with_fixed_boundary")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")

    def packed(value: dict[str, Any]) -> tuple[Any, Any, Any, int]:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        return input_ids, attention_mask, completion_ids, completion_ids.size(1) + 1

    def score(value: dict[str, Any], body: Any, head: Any, mode: str) -> Any:
        boundary.mode = mode
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
            outputs = body(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = outputs.last_hidden_state[:, -keep:, :]
            logits = head(hidden)
        logits = logits.float()[:, :-1, :]
        logits = logits[:, -completion_ids.size(1) :, :]
        return selective_log_softmax(logits, completion_ids)

    arms = {
        "barrier_reference_Eb": (raw.model, raw.lm_head, "eager"),
        "barrier_injection_EbI": (raw.model, raw.lm_head, "compiled"),
        "barrier_candidate_Cb": (compiled_body, compiled_head, "compiled"),
        "barrier_repair_CbR": (compiled_body, compiled_head, "eager"),
    }

    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = score(history, *arms["barrier_candidate_Cb"])
        del value, history
        gc.collect()

    values: dict[str, Any] = {}
    arm_records: dict[str, Any] = {}
    call_deltas: dict[str, dict[str, int]] = {}
    compile_counts_before_arms = {name: audit["backend_compiles"] for name, audit in compile_audit.items()}
    for name, arguments in arms.items():
        before = dict(boundary.calls)
        repeats = []
        hashes = []
        for _ in range(2):
            value = score(inputs, *arguments)
            detached = value.detach().float().cpu()
            repeats.append(detached)
            hashes.append(tensor_sha256(detached))
            del value
            gc.collect()
        after = dict(boundary.calls)
        values[name] = repeats[0]
        arm_records[name] = {
            "sha256": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "repeat_max_abs": float((repeats[1] - repeats[0]).abs().max().item()),
        }
        call_deltas[name] = {key: after[key] - before[key] for key in before}
    compile_counts_after_arms = {name: audit["backend_compiles"] for name, audit in compile_audit.items()}

    gates = {
        "barrier_reference_reproduces_eager_anchor": arm_records["barrier_reference_Eb"]["sha256"][0] == EAGER_ANCHOR,
        "all_repeats_exact": all(row["repeat_exact"] for row in arm_records.values()),
        "compiled_target_invoked_in_injection": call_deltas["barrier_injection_EbI"]["compiled"] == 2,
        "compiled_target_invoked_in_candidate": call_deltas["barrier_candidate_Cb"]["compiled"] == 2,
        "eager_target_invoked_in_reference": call_deltas["barrier_reference_Eb"]["eager"] == 2,
        "eager_target_invoked_in_repair": call_deltas["barrier_repair_CbR"]["eager"] == 2,
        "no_outer_recompile_between_target_modes": compile_counts_after_arms["outer_body_with_fixed_boundary"] == compile_counts_before_arms["outer_body_with_fixed_boundary"],
    }
    within_barrier_valid = all(gates.values())
    candidate_transport = arm_records["barrier_candidate_Cb"]["sha256"][0] == CANDIDATE_ANCHOR
    contrasts = {
        "injection_Eb_to_EbI": metrics(torch, values["barrier_reference_Eb"], values["barrier_injection_EbI"]),
        "repair_Cb_to_CbR": metrics(torch, values["barrier_candidate_Cb"], values["barrier_repair_CbR"]),
        "target_effect_on_compiled_context_CbR_to_Cb": metrics(torch, values["barrier_repair_CbR"], values["barrier_candidate_Cb"]),
        "barrier_total_Eb_to_Cb": metrics(torch, values["barrier_reference_Eb"], values["barrier_candidate_Cb"]),
    }
    payload = {
        "schema_version": "forkcert.qwen3-layer27-input-norm-barrier.v0.1",
        "status": "VALID_BARRIER_CONDITIONED_OPERATOR_EFFECT" if within_barrier_valid else "INVALID_TREATMENT",
        "transport_to_original_candidate": "VALID" if within_barrier_valid and candidate_transport else "NOT_ESTABLISHED",
        "subject": "Qwen3-0.6B decoder layer 27 input_layernorm invocation",
        "state": "heldout-transport-B-step29",
        "anchors": {"eager": EAGER_ANCHOR, "candidate": CANDIDATE_ANCHOR},
        "gates": gates,
        "candidate_anchor_exact": candidate_transport,
        "arms": arm_records,
        "target_call_deltas": call_deltas,
        "contrasts": contrasts,
        "compile_audit": compile_audit,
        "compile_counts_before_arms": compile_counts_before_arms,
        "compile_counts_after_arms": compile_counts_after_arms,
        "claim_limits": [
            "barrier-conditioned invocation effect unless candidate anchor is exact",
            "selected matched state only",
            "RMSNorm invocation, not constituent primitive attribution",
            "forward selected-token observable only",
            "implementation-relative, not correctness",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
