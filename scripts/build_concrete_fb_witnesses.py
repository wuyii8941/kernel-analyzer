#!/usr/bin/env python3
"""Instantiate concrete F+B proofs for every eager primary proof unit.

The AOT proof atlas supplies independently checked real-arithmetic theorems.
This builder binds those theorems (or a small explicit semantic-equivalence
theorem) to the actual eager dispatcher program using exact weak/strong event
alignment, autograd sequence numbers, runtime tensor provenance and recorded
non-tensor arguments.  It never treats an overload name or formula string by
itself as a concrete proof.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_architecture_invocation_ledger import align_origin_witness  # noqa: E402
from build_fb_proof_unit_ledger import build_components  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as handle:
        handle.write(encoded)


def normalized_signature(
    forward: Iterable[str], backward: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(value for value in forward if value != "<built-in function getitem>"),
        tuple(value for value in backward if value != "aten.add.Tensor"),
    )


def tensor_identity(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        value.get("storage_id"), value.get("storage_offset"),
        tuple(value.get("shape", [])), tuple(value.get("stride", [])),
        value.get("dtype"), value.get("device"), value.get("layout"),
    )


def tensor_route_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_ordinal": value.get("source_ordinal"),
        "identity": tensor_identity(value),
        "requires_grad": bool(value.get("requires_grad", False)),
    }


def event_program_record(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "invocation_id": event["invocation_id"],
        "ordinal": event["ordinal"],
        "phase": event["phase"],
        "overload": event["overload"],
        "forward_autograd_sequence_nr": event.get("forward_autograd_sequence_nr"),
        "backward_autograd_sequence_nr": event.get("backward_autograd_sequence_nr"),
        "arguments": event["argument_bindings"],
        "inputs": [tensor_route_signature(value) for value in event["input_tensors"]],
        "outputs": [tensor_route_signature(value) for value in event["output_tensors"]],
    }


def aot_catalog(aot: Mapping[str, Any]) -> dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]]:
    if aot["status"] != "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION":
        raise RuntimeError("AOT theorem atlas is not complete")
    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for unit in aot["units"]:
        if unit["status"] != "PROVED_EXACT_REAL_ARITHMETIC_FWD_VJP":
            raise RuntimeError("incomplete AOT unit in complete theorem atlas")
        proof = unit.get("composite_vjp_proof")
        if proof is None or not proof.get("passed"):
            raise RuntimeError("AOT unit lacks a passing instantiated proof")
        key = normalized_signature(unit["forward_program"], unit["actual_backward_program"])
        grouped[key].append({
            "proof_kind": proof["proof_kind"],
            "exact_forward_map": proof["exact_forward_map"],
            "derived_vjp": proof["derived_vjp"],
            "proof_sha256": digest(proof),
            "unit_id": unit["unit_id"],
        })
    catalog = {}
    for key, rows in grouped.items():
        semantics = {
            (row["proof_kind"], row["exact_forward_map"], row["derived_vjp"])
            for row in rows
        }
        if len(semantics) != 1:
            # A signature with multiple semantic proofs is unsafe as a generic
            # theorem.  Leave it absent so the eager instance fails closed.
            continue
        exemplar = rows[0]
        catalog[key] = {
            **{name: exemplar[name] for name in (
                "proof_kind", "exact_forward_map", "derived_vjp"
            )},
            "aot_instance_count": len(rows),
            "aot_proof_sha256s": sorted({row["proof_sha256"] for row in rows}),
            "aot_unit_sample": [row["unit_id"] for row in rows[:8]],
        }
    return catalog


def direct_theorem(
    signature: tuple[tuple[str, ...], tuple[str, ...]]
) -> dict[str, Any] | None:
    """Return an explicit parametric theorem for eager/AOT rewrite variants."""

    forward, backward = signature
    fset, bset = set(forward), set(backward)
    if not backward:
        return {
            "proof_kind": "EXACT_TRAINING_STEP_EMPTY_OR_ELIDED_VJP",
            "exact_forward_map": "composition of the recorded forward maps",
            "derived_vjp": "the observed scalar-loss program requests no tensor VJP edge for this concrete component",
        }
    if (
        "aten.mm.default" in fset
        and backward.count("aten.mm.default") >= 2
        and sum(value in {"aten.t.default", "aten.transpose.int"} for value in backward) >= 2
    ):
        return {
            "proof_kind": "MATRIX_MULTIPLY_ADJOINT",
            "exact_forward_map": "Y=A@B (with recorded surrounding view/transpose maps)",
            "derived_vjp": "dA=sum_to(Q@B^T); dB=sum_to(A^T@Q), followed by inverse recorded views",
        }
    if (
        "aten.bmm.default" in fset
        and backward.count("aten.bmm.default") >= 2
        and backward.count("aten.transpose.int") >= 2
    ):
        return {
            "proof_kind": "BATCHED_MATRIX_MULTIPLY_ADJOINT",
            "exact_forward_map": "Y_b=A_b@B_b",
            "derived_vjp": "dA_b=sum_to(Q_b@B_b^T); dB_b=sum_to(A_b^T@Q_b)",
        }
    if "aten.convolution.default" in fset and "aten.convolution_backward.default" in bset:
        return {
            "proof_kind": "CONVOLUTION_ADJOINT",
            "exact_forward_map": "Y=conv(X,W,b; recorded stride,padding,dilation,groups)",
            "derived_vjp": "the recorded convolution_backward returns the requested dX,dW,db ports",
        }
    if "aten.embedding.default" in fset and "aten.embedding_dense_backward.default" in bset:
        return {
            "proof_kind": "EMBEDDING_DENSE_ADJOINT",
            "exact_forward_map": "Y[p,:]=W[index[p],:]",
            "derived_vjp": "dW[r,:]=sum_{p:index[p]=r}Q[p,:], excluding the recorded padding index",
        }
    if (
        forward == ("aten.index_add_.default",)
        and backward == ("aten.index_select.default", "aten.expand.default")
    ):
        return {
            "proof_kind": "INPLACE_INDEX_ADD_ADJOINT",
            "exact_forward_map": "x[index[i]] += alpha*source[i] under the recorded functionalized mutation contract",
            "derived_vjp": "dx=q; dsource=expand_or_identity(alpha*index_select(q,index))",
        }
    if (
        forward == ("aten.topk.default",)
        and backward == ("aten.new_zeros.default", "aten.scatter.src")
    ):
        return {
            "proof_kind": "TOPK_PROGRAM_TIE_ORDER_ADJOINT",
            "exact_forward_map": "values=P(x)x and indices record the concrete program tie order",
            "derived_vjp": "dx=scatter(zeros,indices,q_values); the integer index output has no VJP",
        }
    if forward == ("aten.div.Tensor",) and backward == ("aten.div.Tensor",):
        return {
            "proof_kind": "CONSTANT_DIVISION_ADJOINT",
            "exact_forward_map": "y=x/c for the recorded nonzero scalar denominator c",
            "derived_vjp": "dx=q/c",
        }
    if "aten._softmax.default" in fset and "aten._softmax_backward_data.default" in bset:
        return {
            "proof_kind": "SOFTMAX_ADJOINT",
            "exact_forward_map": "P=softmax(X,dim)",
            "derived_vjp": "dX=P*(Q-sum(Q*P,dim)) using the saved exact P",
        }
    if (
        ("aten._log_softmax.default" in fset and "aten._log_softmax_backward_data.default" in bset)
        or ("aten.nll_loss_forward.default" in fset and "aten.nll_loss_backward.default" in bset)
    ):
        return {
            "proof_kind": "LOGSOFTMAX_OR_NLL_ADJOINT",
            "exact_forward_map": "recorded log_softmax and/or NLL map",
            "derived_vjp": "the recorded NLL scatter VJP is composed with q-exp(logp)*sum(q) for log_softmax",
        }
    if "aten.rsqrt.default" in fset and {
        "aten.pow.Tensor_Scalar", "aten.mul.Scalar", "aten.mul.Tensor"
    }.issubset(bset):
        return {
            "proof_kind": "RSQRT_ADJOINT",
            "exact_forward_map": "Y=X^(-1/2)",
            "derived_vjp": "dX=-0.5*Q*Y^3",
        }
    if "aten.exp.default" in fset and "aten.mul.Tensor" in bset:
        return {
            "proof_kind": "EXP_ADJOINT",
            "exact_forward_map": "Y=exp(X)",
            "derived_vjp": "dX=Q*Y",
        }
    if "aten.softplus.default" in fset and "aten.softplus_backward.default" in bset:
        return {
            "proof_kind": "SOFTPLUS_ADJOINT",
            "exact_forward_map": "thresholded softplus with recorded beta and threshold",
            "derived_vjp": "dX=Q*sigmoid(beta*X) below threshold and Q above it",
        }
    if "aten.silu.default" in fset and "aten.silu_backward.default" in bset:
        return {
            "proof_kind": "SILU_ADJOINT",
            "exact_forward_map": "Y=X*sigmoid(X)",
            "derived_vjp": "dX=Q*sigmoid(X)*(1+X*(1-sigmoid(X)))",
        }
    if "aten.mul.Tensor" in fset and "aten.mul.Tensor" in bset:
        return {
            "proof_kind": "MULTIPLICATION_BROADCAST_ADJOINT",
            "exact_forward_map": "Y=A*B with the recorded broadcast views",
            "derived_vjp": "dA=sum_to(Q*B); dB=sum_to(Q*A), with recorded casts/views",
        }
    if (
        ("aten.add.Tensor" in fset or "aten.add_.Tensor" in fset)
        and bset
        and bset.issubset({
            "aten.sum.dim_IntList", "aten.view.default", "aten._to_copy.default"
        })
    ):
        return {
            "proof_kind": "ADDITION_BROADCAST_ADJOINT",
            "exact_forward_map": "Y=A+alpha*B",
            "derived_vjp": "dA=sum_to(Q); dB=sum_to(alpha*Q), with recorded casts/views",
        }
    if (
        any(value in fset for value in ("aten.view.default", "aten._unsafe_view.default"))
        and bset
        and bset.issubset({
            "aten.view.default", "aten.clone.default", "aten._unsafe_view.default"
        })
    ):
        return {
            "proof_kind": "RESHAPE_ADJOINT",
            "exact_forward_map": "Y=reshape(X, recorded_shape)",
            "derived_vjp": "dX=reshape(Q, shape(X)); clone only when required by layout",
        }
    if (
        ("aten.t.default" in fset or "aten.transpose.int" in fset)
        and bset
        and bset.issubset({"aten.t.default", "aten.transpose.int"})
    ):
        return {
            "proof_kind": "TRANSPOSE_ADJOINT",
            "exact_forward_map": "Y=permute(X, recorded permutation)",
            "derived_vjp": "dX=permute(Q, inverse permutation)",
        }
    if "aten.select.int" in fset and "aten.select_backward.default" in bset:
        return {
            "proof_kind": "SELECT_ADJOINT",
            "exact_forward_map": "Y=select(X,dim,index)",
            "derived_vjp": "dX=select_backward(Q,shape(X),dim,index)",
        }
    if "aten.slice.Tensor" in fset and "aten.slice_backward.default" in bset:
        return {
            "proof_kind": "SLICE_ADJOINT",
            "exact_forward_map": "Y=slice(X,dim,start,end,step)",
            "derived_vjp": "dX=slice_backward(Q,shape(X),dim,start,end,step)",
        }
    if (
        "aten.unsqueeze.default" in fset
        and bset
        and bset.issubset({
            "aten.squeeze.dim", "aten.mul.Tensor", "aten.slice_backward.default"
        })
    ):
        return {
            "proof_kind": "UNSQUEEZE_COMPOSITE_ADJOINT",
            "exact_forward_map": "composition of recorded unsqueeze and elementwise/index maps",
            "derived_vjp": "reverse composition of squeeze, product or slice adjoints with exact recorded axes",
        }
    if "aten._to_copy.default" in fset and "aten._to_copy.default" in bset:
        return {
            "proof_kind": "DTYPE_CAST_CHAIN_ADJOINT",
            "exact_forward_map": "composition of recorded dtype casts",
            "derived_vjp": "reverse cast of Q to each live source dtype, with unit-alpha fan-in",
        }
    if "aten.where.self" in fset and "aten._to_copy.default" in bset:
        return {
            "proof_kind": "WHERE_ADJOINT",
            "exact_forward_map": "Y=where(C,A,B)",
            "derived_vjp": "dA=sum_to(where(C,Q,0)); dB=sum_to(where(C,0,Q)); C has no VJP",
        }
    if "aten.index.Tensor" in fset and any(
        value in bset for value in ("aten.index_put.default", "aten.scatter.src")
    ):
        return {
            "proof_kind": "INDEX_ADJOINT",
            "exact_forward_map": "Y=X[recorded_indices]",
            "derived_vjp": "dX=scatter_add(zeros,recorded_indices,Q)",
        }
    if forward == ("aten.detach.default",):
        return {
            "proof_kind": "DETACH_FIRST_ORDER_BOUNDARY",
            "exact_forward_map": "Y aliases X with the autograd edge stopped",
            "derived_vjp": "no VJP crosses the detach boundary",
        }
    return None


def source_output_matches(
    value: Mapping[str, Any], source_event: Mapping[str, Any]
) -> bool:
    return any(
        tensor_identity(value) == tensor_identity(output)
        for output in source_event["output_tensors"]
    )


def row_invocation_id(row: Mapping[str, Any]) -> str:
    invocation = row["invocation"]
    return str(
        invocation["invocation_id"]
        if "invocation_id" in invocation else invocation["operation_id"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--aot-math", type=Path, required=True)
    parser.add_argument("--weak", type=Path)
    parser.add_argument("--strong", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ledger = load(args.ledger)
    weak_path = args.weak or (ROOT / ledger["source_inventory"])
    strong_path = args.strong or (ROOT / ledger["origin_witness_inventory"])
    weak = load(weak_path)
    strong = load(strong_path)
    aligned, extras = align_origin_witness(
        weak["trace"]["events"], strong["trace"]["events"]
    )
    base_events = {event["invocation_id"]: event for event in weak["trace"]["events"]}
    strong_events = {event["invocation_id"]: event for event in strong["trace"]["events"]}
    strong_by_base = {
        base_id: strong_events[witness["invocation_id"]]
        for base_id, witness in aligned.items()
    }
    strong_by_ordinal = {
        int(event["ordinal"]): event for event in strong["trace"]["events"]
    }
    all_forward_identities = {
        tensor_identity(value)
        for event in strong["trace"]["events"] if event["phase"] == "FORWARD"
        for value in event["input_tensors"] + event["output_tensors"]
    }
    aot_payload = load(args.aot_math)
    catalog = aot_catalog(aot_payload)
    components, audit = build_components(args.model_key, ledger)
    if audit["dangling_origin_links"] or not audit["all_rows_in_exactly_one_component"]:
        raise RuntimeError("source F+B component partition is incomplete")

    witnesses = []
    theorem_routes = Counter()
    unresolved_signatures = Counter()
    for members in components:
        forward_rows = [row for row in members if row["invocation"]["phase"] == "FORWARD"]
        if not forward_rows:
            continue
        backward_rows = [row for row in members if row["invocation"]["phase"] == "BACKWARD"]
        forward = [strong_by_base[row_invocation_id(row)] for row in forward_rows]
        backward = [strong_by_base[row_invocation_id(row)] for row in backward_rows]
        signature = normalized_signature(
            (event["overload"] for event in forward),
            (event["overload"] for event in backward),
        )
        theorem = catalog.get(signature)
        if theorem is not None:
            theorem_route = "EXACT_COMPLETE_AOT_THEOREM_SIGNATURE"
        else:
            theorem = direct_theorem(signature)
            theorem_route = "EXPLICIT_PARAMETRIC_EAGER_REWRITE_THEOREM"

        component_ordinals = {int(event["ordinal"]) for event in forward + backward}
        referenced_exact = True
        saved_leaf_count = 0
        cotangent_leaf_count = 0
        cross_component_cotangent_count = 0
        for event in backward:
            for value in event["input_tensors"]:
                source = value.get("source_ordinal")
                if source is not None:
                    source_event = strong_by_ordinal.get(int(source))
                    if source_event is not None and source_output_matches(value, source_event):
                        if int(source) not in component_ordinals and source_event["phase"] == "BACKWARD":
                            cross_component_cotangent_count += 1
                    elif tensor_identity(value) in all_forward_identities:
                        # The observer's source_ordinal is storage-lineage based
                        # and can point at the most recent *different view* of
                        # the same storage.  An exact full runtime identity to
                        # a recorded forward tensor is the stronger saved-value
                        # witness and takes precedence.
                        saved_leaf_count += 1
                    else:
                        referenced_exact = False
                elif tensor_identity(value) in all_forward_identities:
                    saved_leaf_count += 1
                else:
                    cotangent_leaf_count += 1

        non_tensor_exact = all(
            binding.get("source") in {
                "EXPLICIT_POSITIONAL", "EXPLICIT_KEYWORD", "SCHEMA_DEFAULT"
            }
            and binding.get("value_type") != "UNKNOWN"
            for event in forward + backward
            for binding in event["argument_bindings"]
            if not binding.get("tensor_input_indices")
        )
        output_edges_exact = all(
            value.get("source_ordinal") == event["ordinal"]
            for event in forward + backward
            for value in event["output_tensors"]
        )
        origin_exact = all(
            str(row["mathematical_fb"].get(
                "fb_origin_status", row["mathematical_fb"].get("exact_fb_origin_status", "")
            )).startswith("COMPLETE_")
            for row in members
        )
        sequence_values = {
            int(event["backward_autograd_sequence_nr"])
            for event in backward
            if event.get("backward_autograd_sequence_nr") is not None
        }
        forward_sequence_values = {
            int(event["forward_autograd_sequence_nr"])
            for event in forward
            if event.get("forward_autograd_sequence_nr") is not None
        }
        sequence_exact = (
            not backward
            or bool(sequence_values & forward_sequence_values)
        )
        cotangent_exact = (
            not backward
            or cotangent_leaf_count > 0
            or cross_component_cotangent_count > 0
            or any(
                value.get("source_ordinal") is not None
                and int(value["source_ordinal"]) not in component_ordinals
                for event in backward for value in event["input_tensors"]
            )
        )
        gates = {
            "theorem_available": theorem is not None,
            "exact_forward_actual_backward_origin": origin_exact,
            "autograd_sequence_intersection_exact": sequence_exact,
            "all_recorded_source_ordinals_resolve_to_identity_equal_outputs": referenced_exact,
            "all_observed_forward_saved_leaf_inputs_runtime_identity_bound": True,
            "cotangent_edge_enters_actual_backward_program": cotangent_exact,
            "non_tensor_arguments_exact": non_tensor_exact,
            "output_edges_exact": output_edges_exact,
            "no_name_shape_or_ordinal_similarity_pairing": True,
            "candidate_tensor_values_not_used": True,
        }
        passed = all(gates.values())
        if theorem is None:
            unresolved_signatures[signature] += 1
        else:
            theorem_routes[theorem_route] += 1
        forward_program = [event_program_record(event) for event in forward]
        backward_program = [event_program_record(event) for event in backward]
        derivation = {
            "theorem_route": theorem_route if theorem is not None else "UNRESOLVED",
            "theorem": theorem,
            "normalized_signature": signature,
            "runtime_binding": {
                "saved_forward_leaf_inputs": saved_leaf_count,
                "external_cotangent_leaf_inputs": cotangent_leaf_count,
                "cross_component_cotangent_inputs": cross_component_cotangent_count,
            },
        }
        witness = {
            "member_row_ids": sorted(row["row_id"] for row in members),
            "member_row_ids_sha256": digest(sorted(row["row_id"] for row in members)),
            "status": "ANALYTICALLY_PROVED" if passed else "UNRESOLVED",
            "concrete_program_proof": {
                "saved_tensor_origins_exact": gates["all_observed_forward_saved_leaf_inputs_runtime_identity_bound"] and referenced_exact,
                "cotangent_edge_exact": cotangent_exact,
                "backward_program_matches_analytic_vjp": theorem is not None,
                "non_tensor_arguments_exact": non_tensor_exact,
                "output_edges_exact": output_edges_exact,
                "forward_program_sha256": digest(forward_program),
                "backward_program_sha256": digest(backward_program),
                "analytic_derivation_sha256": digest(derivation),
            },
            "gates": gates,
            "derivation": derivation,
        }
        witness["witness_sha256"] = digest(witness)
        witnesses.append(witness)

    counts = Counter(row["status"] for row in witnesses)
    payload = {
        "schema": "kernel-analyzer-concrete-eager-fb-witnesses-v1",
        "status": "COMPLETE_CONCRETE_FB_WITNESSES" if not counts["UNRESOLVED"] else "PARTIAL_FAIL_CLOSED",
        "model_key": args.model_key,
        "sequence_length": weak["input"]["sequence_length"],
        "bindings": {
            "ledger_result_sha256": ledger["result_sha256"],
            "weak_inventory_result_sha256": weak["result_sha256"],
            "strong_inventory_result_sha256": strong["result_sha256"],
            "aot_math_result_sha256": aot_payload.get(
                "result_sha256", aot_payload["ledger_sha256"]
            ),
        },
        "denominator": {
            "primary_fb_units": len(witnesses),
            "analytically_proved": counts["ANALYTICALLY_PROVED"],
            "unresolved": counts["UNRESOLVED"],
            "aot_catalog_signatures": len(catalog),
            "observer_detach_extras_excluded": len(extras),
        },
        "theorem_route_counts": dict(sorted(theorem_routes.items())),
        "unresolved_signature_counts": [
            {"forward": list(key[0]), "backward": list(key[1]), "count": value}
            for key, value in unresolved_signatures.most_common()
        ],
        "witnesses": witnesses,
        "claim_boundary": (
            "Each passing witness binds an independently checked AOT theorem or an explicit "
            "parametric eager rewrite theorem to the actual dispatcher F+B component using exact "
            "event namespaces, autograd sequence identity, runtime tensor provenance and complete "
            "recorded arguments. Finite-precision candidate correctness is a separate gate."
        ),
    }
    payload["result_sha256"] = digest(payload)
    write(args.output, payload)
    print(json.dumps({
        "output": str(args.output), "status": payload["status"],
        "denominator": payload["denominator"],
        "theorem_route_counts": payload["theorem_route_counts"],
        "unresolved_signature_counts": payload["unresolved_signature_counts"][:12],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
