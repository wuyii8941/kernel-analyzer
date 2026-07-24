#!/usr/bin/env python
"""Build the forward/provenance half of the step236 operator evidence slice."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from forkcert.operator_evidence import (
    EvidenceGates,
    allowed_claim_level,
    canonical_json_sha256,
    compare_non_target_context,
    fingerprint_tree,
    production_mediation_interpretation,
    sha256_file,
    tensor_fingerprint,
)


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


def tensor_sha256(value: Any) -> str:
    return tensor_fingerprint(value)["content_sha256"]


def metrics(torch: Any, reference: Any, candidate: Any) -> dict[str, Any]:
    delta = candidate.double() - reference.double()
    return {
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_signed": float(delta.mean().item()),
        "nonzero": int(torch.count_nonzero(delta).item()),
    }


def clip_profile(torch: Any, logps: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    advantages = inputs["advantages"].unsqueeze(1)
    old = inputs["old_per_token_logps"]
    ratio = torch.exp(logps - old)
    signed_margin = torch.where(advantages > 0, ratio - 1.2, 0.8 - ratio)
    eligible = advantages != 0
    decisions = ((ratio < 0.8) & (advantages < 0)) | ((ratio > 1.2) & (advantages > 0))
    return {
        "ratio": ratio,
        "signed_margin": signed_margin,
        "eligible": eligible.expand_as(ratio),
        "decisions": decisions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    snapshot_dir = Path(manifest["snapshot_dir"]).resolve()
    contract_path = Path(manifest["realization_contract"]).resolve()
    contract = json.loads(contract_path.read_text())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = out_dir / "raw" / "inductor_trace"
    graph_dir = out_dir / "raw" / "dynamo_graphs"
    graph_dir.mkdir(parents=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("INDUCTOR_PROVENANCE", "1")

    import torch
    from accelerate import Accelerator
    from torch import nn
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from trl.trainer.grpo_trainer import selective_log_softmax

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 64
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_dir = str(trace_dir)
    torch._inductor.config.trace.save_real_tensors = False
    torch._inductor.config.trace.fx_graph = True
    torch._inductor.config.trace.fx_graph_transformed = True
    torch._inductor.config.trace.ir_pre_fusion = True
    torch._inductor.config.trace.ir_post_fusion = True
    torch._inductor.config.trace.output_code = True

    metadata_path = snapshot_dir / "forkcert_transition_snapshot.json"
    metadata = json.loads(metadata_path.read_text())
    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot_dir / "compiler_history" / target_path.name
    inputs = move_tree(torch.load(target_path, map_location="cpu", weights_only=False), "cuda")

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    accelerator = Accelerator(mixed_precision="fp16")
    wrapped = accelerator.prepare_model(model)
    raw = accelerator.unwrap_model(wrapped)

    class PreNormDecoder(nn.Module):
        def __init__(self, body: Any):
            super().__init__()
            self.embed_tokens = body.embed_tokens
            self.layers = body.layers
            self.rotary_emb = body.rotary_emb
            self.config = body.config
            self.has_sliding_layers = body.has_sliding_layers

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden_states = self.embed_tokens(input_ids)
            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            mask_kwargs = dict(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=masks[self.config.layer_types[index]],
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
            return hidden_states

    compile_audit: dict[str, dict[str, Any]] = {}

    def tracked_compile(module: Any, region_id: str) -> Any:
        from torch._dynamo.backends.registry import lookup_backend

        inductor = lookup_backend("inductor")
        audit = {"backend_compiles": 0, "runtime_invocations": 0, "graphs": []}
        compile_audit[region_id] = audit

        def backend(graph_module: Any, example_inputs: list[Any]):
            code = graph_module.code
            graph_hash = hashlib.sha256(code.encode()).hexdigest()
            nodes = []
            for node in graph_module.graph.nodes:
                nodes.append(
                    {
                        "name": node.name,
                        "op": node.op,
                        "target": str(node.target),
                        "nn_module_stack": str(node.meta.get("nn_module_stack")),
                        "source_fn_stack": str(node.meta.get("source_fn_stack")),
                        "stack_trace": node.meta.get("stack_trace"),
                    }
                )
            index = audit["backend_compiles"]
            graph_file = graph_dir / f"{region_id}_{index:02d}_{graph_hash}.json"
            graph_file.write_text(json.dumps(nodes, indent=2, sort_keys=True) + "\n")
            audit["backend_compiles"] += 1
            audit["graphs"].append(
                {
                    "region_id": region_id,
                    "sha256": graph_hash,
                    "node_count": len(nodes),
                    "node_manifest": str(graph_file.relative_to(out_dir)),
                    "node_manifest_sha256": sha256_file(graph_file),
                }
            )
            compiled = inductor(graph_module, example_inputs)

            def counted(*values: Any):
                audit["runtime_invocations"] += 1
                return compiled(*values)

            return counted

        return torch.compile(module, backend=backend)

    prefix = PreNormDecoder(raw.model)
    compiled_prefix = tracked_compile(prefix, "decoder_prefix")
    compiled_norm = tracked_compile(raw.model.norm, "final_rmsnorm")
    compiled_head = tracked_compile(raw.lm_head, "lm_head")

    def packed(value: dict[str, Any]) -> tuple[Any, Any, Any, int]:
        completion = value["completion_ids"]
        return (
            torch.cat([value["prompt_ids"], completion], dim=1),
            torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1),
            completion,
            completion.size(1) + 1,
        )

    def split_score(value: dict[str, Any], pre: Any, norm: Any, head: Any, capture: dict[str, Any] | None = None) -> Any:
        input_ids, attention_mask, completion_ids, keep = packed(value)
        with sdpa_kernel(SDPBackend.MATH), accelerator.autocast():
            hidden = pre(input_ids, attention_mask)
            if capture is not None:
                capture["norm_input"] = hidden.detach()
            hidden = norm(hidden)
            if capture is not None:
                capture["norm_output"] = hidden.detach()
            logits = head(hidden[:, -keep:, :])
        logits = logits.float()[:, :-1, :]
        return selective_log_softmax(logits[:, -completion_ids.size(1) :, :], completion_ids)

    # Warm the exact ordered history using the partitioned candidate.
    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = split_score(history, compiled_prefix, compiled_norm, compiled_head)
        del history, value
        gc.collect()

    arms: dict[str, Callable[[], Any]] = {
        "reference": lambda: split_score(inputs, prefix, raw.model.norm, raw.lm_head),
        "candidate": lambda: split_score(inputs, compiled_prefix, compiled_norm, compiled_head),
        "repair": lambda: split_score(inputs, compiled_prefix, raw.model.norm, compiled_head),
        "injection": lambda: split_score(inputs, prefix, compiled_norm, raw.lm_head),
    }
    arm_values: dict[str, Any] = {}
    arm_records: dict[str, Any] = {}
    compile_audit_after_candidate: dict[str, Any] | None = None
    compile_audit_after_repair: dict[str, Any] | None = None
    for name, function in arms.items():
        values = [function().detach().float().cpu() for _ in range(2)]
        hashes = [tensor_sha256(value) for value in values]
        arm_values[name] = values[0]
        arm_records[name] = {
            "sha256": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "repeat_max_abs": float((values[1] - values[0]).abs().max().item()),
        }
        if name == "candidate":
            compile_audit_after_candidate = copy.deepcopy(compile_audit)
        elif name == "repair":
            compile_audit_after_repair = copy.deepcopy(compile_audit)
        del values
        gc.collect()

    # Same-input local replay uses one reference-produced boundary tensor.
    capture: dict[str, Any] = {}
    _ = split_score(inputs, prefix, raw.model.norm, raw.lm_head, capture)
    boundary = capture["norm_input"]
    boundary_before = fingerprint_tree((boundary,))
    local_eager = [raw.model.norm(boundary).detach().float().cpu() for _ in range(2)]
    boundary_middle = fingerprint_tree((boundary,))
    local_compiled = [compiled_norm(boundary).detach().float().cpu() for _ in range(2)]
    boundary_after = fingerprint_tree((boundary,))
    local_record = {
        "region_id": "final_rmsnorm",
        "input_fingerprint": boundary_before,
        "input_exact_before_between_after": boundary_before == boundary_middle == boundary_after,
        "reference_output_sha256": [tensor_sha256(value) for value in local_eager],
        "candidate_output_sha256": [tensor_sha256(value) for value in local_compiled],
        "reference_repeat_exact": tensor_sha256(local_eager[0]) == tensor_sha256(local_eager[1]),
        "candidate_repeat_exact": tensor_sha256(local_compiled[0]) == tensor_sha256(local_compiled[1]),
        "discrepancy": metrics(torch, local_eager[0], local_compiled[0]),
    }

    reference_clip = clip_profile(torch, arm_values["reference"], move_tree(inputs, "cpu"))
    oracle: dict[str, Any] = {"arms": {}, "contrasts": {}}
    for name, value in arm_values.items():
        profile = clip_profile(torch, value, move_tree(inputs, "cpu"))
        oracle["arms"][name] = {
            "scorer_sha256": tensor_sha256(value),
            "clip_count": int(profile["decisions"].sum().item()),
        }
        delta = profile["signed_margin"] - reference_clip["signed_margin"]
        upward = (~reference_clip["decisions"]) & profile["decisions"]
        downward = reference_clip["decisions"] & (~profile["decisions"])
        denominator = int(reference_clip["decisions"].numel())
        oracle["contrasts"][f"reference_to_{name}"] = {
            "mean_signed_margin_shift": float(delta[reference_clip["eligible"]].double().mean().item()),
            "off_to_on": int(upward.sum().item()),
            "on_to_off": int(downward.sum().item()),
            "signed_semantic_effect": float((upward.sum() - downward.sum()).item() / denominator),
            "semantic_disagreement": float((upward.sum() + downward.sum()).item() / denominator),
        }
    candidate_clip = clip_profile(torch, arm_values["candidate"], move_tree(inputs, "cpu"))
    repair_clip = clip_profile(torch, arm_values["repair"], move_tree(inputs, "cpu"))
    upward = (~candidate_clip["decisions"]) & repair_clip["decisions"]
    downward = candidate_clip["decisions"] & (~repair_clip["decisions"])
    delta = repair_clip["signed_margin"] - candidate_clip["signed_margin"]
    oracle["contrasts"]["candidate_to_repair"] = {
        "mean_signed_margin_shift": float(delta[candidate_clip["eligible"]].double().mean().item()),
        "off_to_on": int(upward.sum().item()),
        "on_to_off": int(downward.sum().item()),
        "signed_semantic_effect": float((upward.sum() - downward.sum()).item() / denominator),
        "semantic_disagreement": float((upward.sum() + downward.sum()).item() / denominator),
    }
    production_observed = local_record["discrepancy"]["nonzero"] > 0
    forward_mediation = {
        "fixed_suffix": "eager_lm_head",
        "boundary_values": "eager_vs_compiled final_rmsnorm outputs from the same eager-prefix input",
        "continuous_margin": production_mediation_interpretation(
            production_observed,
            arm_records["injection"]["sha256"][0] != arm_records["reference"]["sha256"][0],
        ),
        "clipping_decision": production_mediation_interpretation(
            production_observed,
            oracle["contrasts"]["reference_to_injection"]["semantic_disagreement"] > 0,
        ),
        "gradient_update": production_mediation_interpretation(production_observed, None),
    }

    trace_files = []
    for path in sorted(item for item in trace_dir.rglob("*") if item.is_file()):
        trace_files.append(
            {
                "path": str(path.relative_to(out_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    provenance_mapping_files = [
        row for row in trace_files if "provenance" in row["path"].lower() and row["path"].endswith(".json")
    ]
    provenance = {
        "status": "COMPLETE_WITHIN_PARTITIONED_PIPELINE" if provenance_mapping_files else "PARTIAL_NO_OFFICIAL_MAPPING_ARTIFACT",
        "scope": "partitioned candidate only; correspondence to the original whole-model compiled realization is unverified",
        "dynamo_graphs": compile_audit,
        "trace_files": trace_files,
        "official_mapping_artifacts": provenance_mapping_files,
        "claim": "lineage/observability only; no equivalence or causality",
    }

    def make_context(audit: dict[str, Any]) -> dict[str, Any]:
        return {
        "compiler_config_digest": contract["compiler_config_digest"],
        "graph_count": sum(row["backend_compiles"] for row in audit.values()),
        "graphs": [graph for row in audit.values() for graph in row["graphs"]],
        "artifacts": [
            {"target_id": graph["region_id"], "sha256": graph["sha256"], "node_count": graph["node_count"]}
            for row in audit.values()
            for graph in row["graphs"]
        ],
        "shape_layout_contracts": boundary_before,
        "autotuning": {"status": "UNOBSERVED"},
        }
    if compile_audit_after_candidate is None or compile_audit_after_repair is None:
        raise RuntimeError("candidate/repair compile-audit snapshots were not captured")
    context_check = compare_non_target_context(
        make_context(compile_audit_after_candidate),
        make_context(compile_audit_after_repair),
        ["final_rmsnorm"],
    )
    candidate_anchor_exact = (
        arm_records["candidate"]["sha256"][0] == contract["candidate_scorer_sha256"]
    )
    reference_anchor_exact = (
        arm_records["reference"]["sha256"][0] == contract["reference_scorer_sha256"]
    )
    all_repeats_exact = all(row["repeat_exact"] for row in arm_records.values())
    gates = EvidenceGates(
        complete_witness=reference_anchor_exact and candidate_anchor_exact,
        same_input_local_replay=local_record["input_exact_before_between_after"],
        local_discrepancy_reproducible=(
            local_record["discrepancy"]["nonzero"] > 0
            and local_record["reference_repeat_exact"]
            and local_record["candidate_repeat_exact"]
        ),
        provenance_complete=False,
        candidate_realization_preserved=False,
        intervention_executed=True,
        oracle_recomputed=True,
        non_target_context_invariant=context_check["exact"],
        null_controls_valid=reference_anchor_exact and all_repeats_exact,
    )
    region_inventory = []
    for region in manifest["regions"]:
        region_inventory.append(
            {
                **region,
                "replayable": region["region_id"] in {"decoder_prefix", "final_rmsnorm", "lm_head"},
                "mutation_alias_status": "INPUT_FINGERPRINTED_NO_MUTATION_DETECTED" if region["region_id"] == "final_rmsnorm" else "UNOBSERVED",
            }
        )
    report = {
        "schema_version": "forkcert.operator-evidence.v0.1",
        "case_identity": {
            "case_id": manifest["case_id"],
            "state_id": manifest["state_id"],
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "snapshot_metadata_sha256": sha256_file(metadata_path),
            "target_minibatch_sha256": sha256_file(target_path),
            "realization_contract_sha256": contract["contract_sha256"],
        },
        "region_inventory": region_inventory,
        "local_replay": local_record,
        "provenance": provenance,
        "intervention": {
            "target": "final_rmsnorm",
            "type": "repair_and_injection_forward_prototype",
            "arms": arm_records,
            "candidate_anchor_exact": candidate_anchor_exact,
            "candidate_realization_preserved": False,
            "candidate_realization_reason": "output identity is not graph/kernel/autograd realization identity",
            "reference_anchor_exact": reference_anchor_exact,
            "non_target_context": context_check,
            "one_step_transition_status": "PENDING_SEPARATE_FRESH_PROCESS_ARMS",
        },
        "oracle": oracle,
        "production_mediation": forward_mediation,
        "gates": gates.__dict__,
        "allowed_claim_level": allowed_claim_level(gates),
        "claim_scope": {
            "licensed": "final_rmsnorm has a reproducible same-input local discrepancy and an auditable isolated intervention",
            "not_licensed": "final_rmsnorm explains the observed clipping disagreement or is a root cause",
        },
        "substantive_findings": {
            "candidate_semantic_disagreement_observed": oracle["contrasts"]["reference_to_candidate"]["semantic_disagreement"] > 0,
            "repair_changed_candidate_output": arm_records["repair"]["sha256"][0] != arm_records["candidate"]["sha256"][0],
            "repair_changed_candidate_semantics": oracle["contrasts"]["candidate_to_repair"]["semantic_disagreement"] > 0,
            "injection_changed_reference_output": arm_records["injection"]["sha256"][0] != arm_records["reference"]["sha256"][0],
            "injection_changed_reference_semantics": oracle["contrasts"]["reference_to_injection"]["semantic_disagreement"] > 0,
        },
        "limitations": [
            "selected post-hoc pipeline-feasibility state, not population evidence",
            "eager is a baseline, not mathematical truth",
            "independently compiled region equality to a whole-model kernel requires provenance evidence",
            "forward repair is not yet a one-step update intervention",
            "no lower-level IR replay or first-bad-pass isolation",
        ],
        "content_sha256": None,
    }
    report["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    (out_dir / "evidence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "allowed_claim_level": report["allowed_claim_level"],
                "candidate_anchor_exact": candidate_anchor_exact,
                "reference_anchor_exact": reference_anchor_exact,
                "local_discrepancy": local_record["discrepancy"],
                "provenance_status": provenance["status"],
                "oracle_contrasts": oracle["contrasts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
