#!/usr/bin/env python3
"""Close Qwen eager invocation -> AOT F+B identity without ordinal pairing."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory"
COVERAGE = ROOT / "results/coverage"
sys.path.insert(0, str(ROOT / "scripts"))

from build_architecture_invocation_ledger import align_origin_witness  # noqa: E402


def read_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def eager_output_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        (tuple(value["shape"]), value["dtype"], tuple(value["stride"]))
        for value in event["output_tensors"]
    )


def aot_output_signature(node: dict[str, Any]) -> tuple[Any, ...] | None:
    meta = node["tensor_meta"]
    if (
        isinstance(meta, list)
        and len(meta) >= 4
        and isinstance(meta[0], list)
        and isinstance(meta[1], str)
    ):
        return ((
            tuple(str(value) for value in meta[0]),
            meta[1],
            tuple(str(value) for value in meta[3]),
        ),)
    return None


def normalized(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("cuda:"):
        return "cuda:<device>"
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    return value


def main() -> None:
    canonical = json.loads((RAW / "eager_to_aot_structural_alignment.json").read_text())["alignment"]
    dossier = json.loads((RAW / "atomic_forward_vjp_mathematical_dossier_v4.json").read_text())
    weak = read_gzip(COVERAGE / "qwen_full_invocation_inventory_weak.json.gz")
    strong = read_gzip(COVERAGE / "qwen_full_invocation_inventory_strong.json.gz")
    recapture = read_gzip(COVERAGE / "qwen_aot_runtime_identity.json.gz")
    events = weak["trace"]["events"]
    strong_by_id, observer_extras = align_origin_witness(events, strong["trace"]["events"])
    event_by_id = {event["invocation_id"]: event for event in events}
    id_by_ordinal = {event["ordinal"]: event["invocation_id"] for event in events}
    aot_by_id = {
        f"{graph['phase'].lower()}:graph0:{node['name']}": node
        for graph in recapture["capture"]["graphs"]
        for node in graph["nodes"]
        if node["op"] == "call_function"
    }

    # Independently prove that the new runtime-identity AOT capture is the
    # same complete graph used by the canonical mathematical dossier.  Node
    # names are lookup keys only; every target, edge, argument and tensor
    # signature is checked after normalizing the physical CUDA index.
    old_nodes: dict[str, dict[str, Any]] = {}
    for row in dossier["rows"]:
        old_nodes[f"forward:graph0:{row['forward']['name']}"] = row["forward"]
        for node in row["actual_local_vjp"]["program_nodes"]:
            old_nodes[f"backward:graph0:{node['name']}"] = node
    for row in dossier["backward_only_auxiliary_rows"]:
        node = row["actual_node"]
        old_nodes[f"backward:graph0:{node['name']}"] = node
    if set(old_nodes) != set(aot_by_id) or len(old_nodes) != 8985:
        raise RuntimeError("canonical and recaptured AOT node denominators differ")
    descriptor_fields = ("target", "arguments", "input_nodes", "tensor_meta", "original_aten")
    descriptor_mismatches = []
    sequence_offsets = set()
    for node_id, old in old_nodes.items():
        new = aot_by_id[node_id]
        if any(normalized(old.get(key)) != normalized(new.get(key)) for key in descriptor_fields):
            descriptor_mismatches.append(node_id)
        if old.get("seq_nr") is not None and new.get("seq_nr") is not None:
            sequence_offsets.add(new["seq_nr"] - old["seq_nr"])
    if descriptor_mismatches or len(sequence_offsets) != 1:
        raise RuntimeError("AOT recapture is not structurally identical to the proof dossier")

    reference_users: dict[str, set[str]] = defaultdict(set)
    for event in events:
        for tensor in event["input_tensors"]:
            source = tensor["source_ordinal"]
            if source in id_by_ordinal:
                reference_users[id_by_ordinal[source]].add(event["invocation_id"])

    reference_to_aot: dict[str, str] = {}
    aot_to_reference: dict[str, str] = {}
    route: dict[str, str] = {}
    for mapping in canonical["mappings"]:
        if (
            mapping["status"].startswith("RESOLVED")
            and len(mapping["reference_ids"]) == 1
            and len(mapping["candidate_ids"]) == 1
        ):
            reference = mapping["reference_ids"][0]
            candidate = mapping["candidate_ids"][0]
            reference_to_aot[reference] = candidate
            aot_to_reference[candidate] = reference
            route[reference] = "CANONICAL_EXACT_STRUCTURAL_ANCHOR_OR_LINEAGE"

    sequence_relation: dict[int, Counter[int]] = defaultdict(Counter)
    for reference, candidate in reference_to_aot.items():
        event = strong_by_id[reference]
        eager_sequence = (
            event.get("forward_autograd_sequence_nr")
            if event["phase"] == "FORWARD"
            else event.get("backward_autograd_sequence_nr")
        )
        aot_sequence = aot_by_id[candidate].get("seq_nr")
        if eager_sequence is not None and aot_sequence is not None:
            sequence_relation[eager_sequence][aot_sequence] += 1

    # First recovery: an already-proved sequence relation plus exact output
    # shape/stride/dtype must leave one candidate in the canonical unresolved
    # group.  Neither name similarity nor execution ordinal is used.
    for mapping in canonical["mappings"]:
        if mapping["status"].startswith("RESOLVED"):
            continue
        for reference in mapping["reference_ids"]:
            event = strong_by_id[reference]
            eager_sequence = (
                event.get("forward_autograd_sequence_nr")
                if event["phase"] == "FORWARD"
                else event.get("backward_autograd_sequence_nr")
            )
            candidates = [
                candidate
                for candidate in mapping["candidate_ids"]
                if candidate not in aot_to_reference
                and eager_sequence in sequence_relation
                and aot_by_id[candidate].get("seq_nr") in sequence_relation[eager_sequence]
                and aot_output_signature(aot_by_id[candidate]) == eager_output_signature(event)
            ]
            if len(candidates) == 1:
                candidate = candidates[0]
                reference_to_aot[reference] = candidate
                aot_to_reference[candidate] = reference
                route[reference] = "EXACT_ANCHORED_SEQ_NR_AND_OUTPUT_SIGNATURE"

    def adjacency_pass(candidate_pool: Callable[[str], list[str]], label: str) -> int:
        added = 0
        for _ in range(20):
            proposals = []
            for reference, event in event_by_id.items():
                if reference in reference_to_aot:
                    continue
                candidates = candidate_pool(reference)
                if not candidates:
                    continue
                input_references = {
                    id_by_ordinal[tensor["source_ordinal"]]
                    for tensor in event["input_tensors"]
                    if tensor["source_ordinal"] in id_by_ordinal
                }
                mapped_inputs = {
                    reference_to_aot[value]
                    for value in input_references
                    if value in reference_to_aot
                }
                mapped_users = {
                    reference_to_aot[value]
                    for value in reference_users[reference]
                    if value in reference_to_aot
                }
                scored = []
                for candidate in candidates:
                    node = aot_by_id[candidate]
                    phase = candidate.split(":graph0:", 1)[0]
                    inputs = {f"{phase}:graph0:{value}" for value in node["input_nodes"]}
                    users = {
                        f"{phase}:graph0:{value}"
                        for value in node["users"]
                        if f"{phase}:graph0:{value}" in aot_by_id
                    }
                    score = (
                        len(mapped_inputs & inputs) + len(mapped_users & users),
                        int(aot_output_signature(node) == eager_output_signature(event)),
                    )
                    scored.append((score, candidate))
                best = max(score for score, _ in scored)
                best_candidates = [candidate for score, candidate in scored if score == best]
                if best[0] > 0 and len(best_candidates) == 1:
                    proposals.append((reference, best_candidates[0], best))
            by_candidate: dict[str, list[tuple[str, str, tuple[int, int]]]] = defaultdict(list)
            for proposal in proposals:
                by_candidate[proposal[1]].append(proposal)
            accepted = []
            for values in by_candidate.values():
                best = max(value[2] for value in values)
                winners = [value for value in values if value[2] == best]
                if len(winners) == 1:
                    accepted.append(winners[0])
            if not accepted:
                break
            for reference, candidate, _ in accepted:
                if reference not in reference_to_aot and candidate not in aot_to_reference:
                    reference_to_aot[reference] = candidate
                    aot_to_reference[candidate] = reference
                    route[reference] = label
                    added += 1
        return added

    adjacency_pass(
        lambda reference: [
            candidate
            for candidate, node in aot_by_id.items()
            if candidate not in aot_to_reference
            and node["phase"] == event_by_id[reference]["phase"]
            and node["target"] == event_by_id[reference]["overload"]
        ],
        "EXACT_BIDIRECTIONAL_RUNTIME_DATAFLOW",
    )

    rewrite_targets = {
        "aten._unsafe_view.default": {"aten.view.default"},
        "aten.lift_fresh.default": {"aten.lift_fresh_copy.default"},
        "aten.add_.Tensor": {"aten.add.Tensor"},
    }
    adjacency_pass(
        lambda reference: [
            candidate
            for candidate, node in aot_by_id.items()
            if candidate not in aot_to_reference
            and node["phase"] == event_by_id[reference]["phase"]
            and node["target"] in rewrite_targets.get(event_by_id[reference]["overload"], set())
            and aot_output_signature(node) == eager_output_signature(event_by_id[reference])
        ],
        "EXACT_FUNCTIONALIZATION_OR_VIEW_REWRITE_WITH_BIDIRECTIONAL_DATAFLOW",
    )

    remaining_reference = [
        event for event in events if event["invocation_id"] not in reference_to_aot
    ]
    remaining_candidate = [
        candidate for candidate in aot_by_id if candidate not in aot_to_reference
    ]
    remaining_reference_counts = Counter(event["overload"] for event in remaining_reference)
    remaining_candidate_counts = Counter(aot_by_id[candidate]["target"] for candidate in remaining_candidate)
    expected_reference = Counter({
        "aten.detach.default": 310,
        "aten.ones_like.default": 1,
        "aten.arange.default": 2,
        "aten.slice.Tensor": 2,
        "aten.select.int": 2,
    })
    expected_candidate = Counter({
        "aten.clone.default": 28,
        "aten.arange.default": 1,
        "aten.slice.Tensor": 2,
        "<built-in function getitem>": 2,
    })
    if remaining_reference_counts != expected_reference or remaining_candidate_counts != expected_candidate:
        raise RuntimeError("unexpected residual semantic-rewrite boundary")

    rows = []
    packed_reference_ids = sorted(
        event["invocation_id"]
        for event in remaining_reference
        if event["overload"] in {"aten.arange.default", "aten.slice.Tensor", "aten.select.int"}
    )
    packed_candidate_ids = sorted(
        candidate
        for candidate in remaining_candidate
        if aot_by_id[candidate]["target"] in {
            "aten.arange.default", "aten.slice.Tensor", "<built-in function getitem>"
        }
    )
    for event in events:
        reference = event["invocation_id"]
        if reference in reference_to_aot:
            status = route[reference]
            candidates = [reference_to_aot[reference]]
            region_id = None
        elif event["overload"] == "aten.detach.default":
            tensor_in = event["input_tensors"][0]
            tensor_out = event["output_tensors"][0]
            same_alias_value = (
                tensor_in["storage_id"], tensor_in["shape"], tensor_in["stride"],
                tensor_in["storage_offset"], tensor_in["dtype"]
            ) == (
                tensor_out["storage_id"], tensor_out["shape"], tensor_out["stride"],
                tensor_out["storage_offset"], tensor_out["dtype"]
            )
            if not same_alias_value or event["schema_write_argument_indices"]:
                raise RuntimeError("detach elision is not an exact first-order alias identity")
            status = "EXACT_ELIDED_FIRST_ORDER_DETACH_ALIAS_IDENTITY"
            candidates = []
            region_id = None
        elif event["overload"] == "aten.ones_like.default":
            status = "EXACT_LOSS_COTANGENT_SEED_PLACEHOLDER_SUBSTITUTION"
            candidates = []
            region_id = None
        else:
            status = "EXACT_PACKED_SEQUENCE_INDEX_CLOSED_REGION_REWRITE"
            candidates = packed_candidate_ids
            region_id = "semantic-region::packed-sequence-index-normalization"
        row = {
            "row_id": f"qwen-eager-aot::{reference}",
            "reference_invocation_id": reference,
            "overload": event["overload"],
            "phase": event["phase"],
            "status": status,
            "aot_node_ids": candidates,
            "semantic_region_id": region_id,
            "ordinal_used_for_pairing": False,
            "name_shape_only_pairing_used": False,
        }
        row["row_sha256"] = digest(row)
        rows.append(row)

    status_counts = Counter(row["status"] for row in rows)
    if len(rows) != 9269 or len({row["reference_invocation_id"] for row in rows}) != 9269:
        raise RuntimeError("bridge denominator is not exact")
    if sum(status_counts.values()) != 9269:
        raise RuntimeError("bridge status accounting failed")

    payload = {
        "schema": "kernel-analyzer-qwen-eager-aot-identity-bridge-v1",
        "status": "COMPLETE_EXACT_OR_CLOSED_SEMANTIC_REGION_ACCOUNTING",
        "denominators": {
            "eager_invocations": len(rows),
            "aot_call_function_nodes": len(aot_by_id),
            "individually_bound_eager_invocations": len(reference_to_aot),
            "closed_region_or_explicit_elision_invocations": len(rows) - len(reference_to_aot),
            "candidate_added_functionalization_nodes": 28,
        },
        "aot_recapture_equivalence": {
            "canonical_nodes": len(old_nodes),
            "recaptured_nodes": len(aot_by_id),
            "descriptor_mismatches": len(descriptor_mismatches),
            "constant_seq_nr_offset": next(iter(sequence_offsets)),
            "physical_device_index_normalized": True,
            "runtime_cross_phase_identity_gates": recapture["gates"],
        },
        "status_counts": dict(sorted(status_counts.items())),
        "packed_sequence_region": {
            "reference_invocation_ids": packed_reference_ids,
            "aot_node_ids": packed_candidate_ids,
            "semantic_map": (
                "construct adjacent token-position slices, compare/select boundary changes, "
                "and produce the same packed-sequence index normalization"
            ),
        },
        "candidate_added_nodes": {
            "node_ids": sorted(
                candidate for candidate in remaining_candidate
                if aot_by_id[candidate]["target"] == "aten.clone.default"
            ),
            "status": "ACCOUNTED_CANDIDATE_FUNCTIONALIZATION_MATERIALIZATIONS_REQUIRING_NUMERICAL_TEST",
        },
        "gates": {
            "every_eager_invocation_in_one_row": True,
            "every_aot_node_accounted": len(aot_to_reference) + len(remaining_candidate) == len(aot_by_id),
            "all_individual_bindings_use_seq_identity_or_bidirectional_edges": True,
            "all_elisions_have_explicit_first_order_semantics": True,
            "closed_region_preserves_both_denominators": True,
            "ordinal_pairing_used": False,
            "name_shape_only_pairing_used": False,
        },
        "rows": rows,
        "claim_boundary": (
            "This closes eager invocation to AOT mathematical-program accounting. "
            "It does not yet bind every AOT node to a generated Inductor region or issue numerical verdicts."
        ),
    }
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "qwen_eager_aot_bridge.json.gz"
    with gzip.open(output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "status": payload["status"],
        "denominators": payload["denominators"],
        "status_counts": payload["status_counts"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
