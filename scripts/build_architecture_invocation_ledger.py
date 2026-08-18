#!/usr/bin/env python3
"""Build fail-closed per-invocation ledgers for supported full LM F+B runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from round2_vl_math import FORMULAS as COMMON_FORMULAS


ROOT = Path(__file__).resolve().parents[1]

EXTRA_FORMULAS: dict[str, dict[str, str]] = {
    "aten._local_scalar_dense.default": {
        "map": "return the sole tensor element as a host scalar (requires numel(x)=1)",
        "adjoint": "no tensor output and therefore no tensor VJP edge in the recorded graph",
    },
    "aten.add_.Tensor": {
        "map": "a is mutated to a + alpha*b",
        "adjoint": "functional value map: da=sum_to(q); db=sum_to(alpha*q)",
    },
    "aten.amax.default": {
        "map": "y=max_D(x) over the declared dimensions",
        "adjoint": "dx=expand(q)*1{x=expand(y)}/tie_count on each reduced fiber",
    },
    "aten.cumsum.default": {
        "map": "y[i]=sum_{j<=i} x[j] along dim",
        "adjoint": "dx[j]=sum_{i>=j} q[i], implemented as reverse-cumsum-reverse",
    },
    "aten.div.Tensor": {
        "map": "y=a/b",
        "adjoint": "da=sum_to(q/b); db=sum_to(-q*a/b^2)",
    },
    "aten.div.Tensor_mode": {
        "map": "y=rounding_mode_divide(a,b)",
        "adjoint": "discrete integer/rounded map; no classical floating VJP",
    },
    "aten.exp.default": {"map": "y=exp(x)", "adjoint": "dx=q*y"},
    "aten.eq.Tensor": {
        "map": "y=(a==b)",
        "adjoint": "Boolean comparison has no differentiable output",
    },
    "aten.full.default": {
        "map": "allocate a tensor filled with scalar c",
        "adjoint": "no differentiable tensor source",
    },
    "aten.gt.Tensor": {
        "map": "y=(a>b)",
        "adjoint": "Boolean comparison has no differentiable output",
    },
    "aten.gt.Scalar": {
        "map": "y=(x>c) elementwise for scalar c",
        "adjoint": "Boolean comparison has no differentiable output",
    },
    "aten.index_add.default": {
        "map": "y=x; y[index[i]] += alpha*source[i]",
        "adjoint": "dx=q; dsource=alpha*index_select(q,index)",
    },
    "aten.index_select.default": {
        "map": "y[i]=x[index[i]] along dim",
        "adjoint": "dx=index_add(zeros,index,q)",
    },
    "aten.gather.default": {
        "map": "y[i]=x[index[i]] along the declared dimension",
        "adjoint": "dx=scatter_add(zeros,index,q) along the same dimension",
    },
    "aten.lift_fresh.default": {
        "map": "lift a fresh tensor into the functional graph without changing values",
        "adjoint": "identity on a floating differentiable input; otherwise no VJP edge",
    },
    "aten.mul_.Tensor": {
        "map": "a is mutated to a*b",
        "adjoint": "functional value map: da=sum_to(q*b); db=sum_to(q*a_saved)",
    },
    "aten.max.default": {
        "map": "m=max_i x_i over all elements",
        "adjoint": "dx_i=q*1{x_i=m}/sum_j 1{x_j=m}; ties share the gradient equally",
    },
    "aten.log.default": {"map": "y=log(x)", "adjoint": "dx=q/x"},
    "aten.reciprocal.default": {
        "map": "y=1/x",
        "adjoint": "dx=-q*y^2",
    },
    "aten.native_dropout.default": {
        "map": "y=mask*x/(1-p), returning y and mask",
        "adjoint": "dx=mask*q/(1-p)",
    },
    "aten.native_dropout_backward.default": {
        "map": "dx=mask*q/scale_inverse",
        "adjoint": "actual dropout VJP program",
    },
    "aten.ne.Scalar": {
        "map": "y=(x!=c)",
        "adjoint": "Boolean comparison has no differentiable output",
    },
    "aten.ones_like.default": {
        "map": "allocate ones with x metadata",
        "adjoint": "output is value-independent of x; dx=0",
    },
    "aten.scatter.src": {
        "map": "y is x with source written at index coordinates",
        "adjoint": "dx=q outside overwritten coordinates; dsource=gather(q,index) under the declared duplicate-index program semantics",
    },
    "aten.scatter.value": {
        "map": "y is x with scalar value written at index coordinates",
        "adjoint": "dx=q outside overwritten coordinates; scalar value is not a tensor edge",
    },
    "aten.silu_backward.default": {
        "map": "dx=q*sigmoid(x)*(1+x*(1-sigmoid(x)))",
        "adjoint": "actual SiLU VJP program",
    },
    "aten.softplus.default": {
        "map": "y=log1p(exp(beta*x))/beta below threshold, else x",
        "adjoint": "dx=q*sigmoid(beta*x) below threshold, else q",
    },
    "aten.softplus_backward.default": {
        "map": "dx is the thresholded softplus derivative times q",
        "adjoint": "actual softplus VJP program",
    },
    "aten.sort.default": {
        "map": "values=P(x)x and indices encode stable program tie ordering",
        "adjoint": "dx=P(x)^T q_values; index output has no VJP",
    },
    "aten.split.Tensor": {
        "map": "return consecutive slices of x with declared split size",
        "adjoint": "dx=cat(q_k,dim)",
    },
    "aten.split_with_sizes.default": {
        "map": "return consecutive slices of x with declared sizes",
        "adjoint": "dx=cat(q_k,dim)",
    },
    "aten.topk.default": {
        "map": "return selected top-k values and program tie-ordered indices",
        "adjoint": "dx=scatter(zeros,selected_indices,q_values); indices have no VJP",
    },
    "aten.sum.default": {
        "map": "y=sum_i x_i over all elements",
        "adjoint": "dx=expand(q,shape(x))",
    },
    "aten.triu.default": {
        "map": "y=upper_triangular_mask(x,diagonal)",
        "adjoint": "dx=upper_triangular_mask(q,diagonal)",
    },
    "prims.convert_element_type.default": {
        "map": "y=cast_dtype(x)",
        "adjoint": "dx=cast_input_dtype(q) for a floating differentiable source",
    },
    "prims.fma.default": {
        "map": "y=a*b+c with one fused-rounding primitive",
        "adjoint": "da=sum_to(q*b); db=sum_to(q*a); dc=sum_to(q)",
    },
    "prims.iota.default": {
        "map": "y is a value-independent arithmetic coordinate sequence",
        "adjoint": "no differentiable tensor source",
    },
}

FORMULAS = {**COMMON_FORMULAS, **EXTRA_FORMULAS}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def compact_ids(values: list[str], limit: int = 8) -> dict[str, Any]:
    ordered = sorted(set(values))
    return {
        "count": len(ordered),
        "sha256": digest(ordered),
        "ids": ordered if len(ordered) <= limit else None,
        "sample": ordered[:limit] if len(ordered) > limit else [],
    }


def tensor_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": row["shape"],
        "stride": row["stride"],
        "dtype": row["dtype"],
        "layout": row["layout"],
        "requires_grad": row["requires_grad"],
    }


def argument_signature(binding: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": binding["name"],
        "schema_type": binding["schema_type"],
        "source": binding["source"],
        "value_type": binding["value_type"],
        "value": binding["value"],
        "tensor_input_indices": binding["tensor_input_indices"],
    }


def alignment_signature(event: dict[str, Any]) -> Any:
    return (
        event["phase"],
        event["overload"],
        tuple(event["module_context"]),
        tuple(
            (value["shape"], value["dtype"], value["requires_grad"])
            for value in event["input_tensors"]
        ),
        tuple(
            (value["shape"], value["dtype"], value["requires_grad"])
            for value in event["output_tensors"]
        ),
    )


def align_origin_witness(
    events: list[dict[str, Any]], witness_events: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Align a strong-origin witness while rejecting non-detach observer drift."""

    aligned: dict[str, dict[str, Any]] = {}
    extras: list[dict[str, Any]] = []
    left = 0
    right = 0
    while left < len(events) and right < len(witness_events):
        event = events[left]
        witness = witness_events[right]
        if alignment_signature(event) == alignment_signature(witness):
            aligned[event["invocation_id"]] = witness
            left += 1
            right += 1
            continue
        if witness["phase"] == "FORWARD" and witness["overload"] == "aten.detach.default":
            extras.append(witness)
            right += 1
            continue
        raise RuntimeError(
            "origin witness diverges beyond an observer-induced forward detach: "
            f"base={event['invocation_id']}:{event['overload']} "
            f"witness={witness['invocation_id']}:{witness['overload']}"
        )
    while right < len(witness_events):
        witness = witness_events[right]
        if witness["phase"] != "FORWARD" or witness["overload"] != "aten.detach.default":
            raise RuntimeError("non-detach trailing origin-witness event")
        extras.append(witness)
        right += 1
    if left != len(events) or len(aligned) != len(events):
        raise RuntimeError("origin witness does not cover every base invocation")
    return aligned, extras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("qwen", "mamba", "moe", "phi", "deepseek8"), required=True)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--origin-inventory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--screen", type=Path)
    args = parser.parse_args()

    inventory = read_gzip(args.inventory)
    events = inventory["trace"]["events"]
    origin_inventory = (
        read_gzip(args.origin_inventory) if args.origin_inventory else inventory
    )
    origin_by_invocation, observer_extra_events = align_origin_witness(
        events, origin_inventory["trace"]["events"]
    )
    base_id_by_origin_id = {
        witness["invocation_id"]: invocation_id
        for invocation_id, witness in origin_by_invocation.items()
    }
    overloads = {event["overload"] for event in events}
    missing_formulas = sorted(overloads - FORMULAS.keys())
    if missing_formulas:
        raise RuntimeError(f"missing mathematical formulas: {missing_formulas}")

    exact_forward_by_sequence: dict[int, list[str]] = defaultdict(list)
    for event in events:
        witness = origin_by_invocation[event["invocation_id"]]
        sequence = witness.get("forward_autograd_sequence_nr")
        if event["phase"] == "FORWARD" and sequence is not None:
            exact_forward_by_sequence[int(sequence)].append(event["invocation_id"])

    backward_by_sequence: dict[int, list[str]] = defaultdict(list)
    for event in events:
        witness = origin_by_invocation[event["invocation_id"]]
        sequence = witness.get("backward_autograd_sequence_nr")
        if event["phase"] == "BACKWARD" and sequence is not None:
            backward_by_sequence[int(sequence)].append(event["invocation_id"])

    condition_units: dict[str, dict[str, Any]] = {}
    rows = []
    unresolved_fb = []
    for event in events:
        witness = origin_by_invocation[event["invocation_id"]]
        formula = FORMULAS[event["overload"]]
        phase = event["phase"]
        forward_sequence = witness.get("forward_autograd_sequence_nr")
        backward_sequence = witness.get("backward_autograd_sequence_nr")
        if (
            phase == "FORWARD"
            and forward_sequence is not None
            and backward_by_sequence.get(int(forward_sequence))
        ):
            fb_status = "COMPLETE_EXACT_FORWARD_OUTPUT_GRAD_FN_AND_ACTUAL_BACKWARD_PROGRAM"
            peer_ids = backward_by_sequence.get(int(forward_sequence), [])
            origin_ids: list[str] = []
        elif phase == "FORWARD" and forward_sequence is not None:
            # The exact output grad_fn exists, but the complete backward trace
            # contains no dispatcher invocation carrying its sequence number.
            # This covers identity/elided VJPs and graph branches not reached by
            # the scalar loss.  Do not invent a backward kernel for it.
            fb_status = "COMPLETE_EXACT_EMPTY_ELIDED_OR_UNREACHED_ACTUAL_VJP"
            peer_ids = []
            origin_ids = []
        elif phase == "FORWARD" and witness["sequence_binding_status"] == "EXACT_NESTED_FORWARD_NO_INDEPENDENT_AUTOGRAD_BOUNDARY":
            fb_status = "COMPLETE_EXACT_ELIDED_NESTED_DISPATCH_VJP"
            peer_ids = []
            enclosing = witness.get("enclosing_dispatch_invocation_id")
            origin_ids = [base_id_by_origin_id[enclosing]] if enclosing in base_id_by_origin_id else []
        elif phase == "FORWARD" and not any(
            output["requires_grad"] for output in event["output_tensors"]
        ):
            fb_status = "COMPLETE_EXACT_EMPTY_OR_STOP_GRAD_VJP"
            peer_ids = []
            origin_ids = []
        elif phase == "BACKWARD" and backward_sequence is not None and int(backward_sequence) in exact_forward_by_sequence:
            fb_status = "COMPLETE_EXACT_ACTUAL_BACKWARD_PROGRAM_TO_FORWARD_ORIGIN"
            peer_ids = []
            origin_ids = exact_forward_by_sequence[int(backward_sequence)]
        elif phase == "BACKWARD" and witness.get("autograd_node") == "torch::autograd::AccumulateGrad":
            fb_status = "COMPLETE_EXACT_GRADIENT_ACCUMULATION_AUXILIARY"
            peer_ids = []
            origin_ids = []
        elif phase == "BACKWARD" and event["overload"] == "aten.ones_like.default":
            fb_status = "COMPLETE_EXACT_LOSS_COTANGENT_SEED"
            peer_ids = []
            origin_ids = []
        else:
            fb_status = "UNRESOLVED_FORWARD_BACKWARD_ORIGIN"
            peer_ids = []
            origin_ids = []
            unresolved_fb.append(event["invocation_id"])

        condition_payload = {
            "architecture": args.architecture,
            "implementation": inventory["implementation"],
            "dtype": "torch.bfloat16",
            "sequence_length": inventory["input"]["sequence_length"],
            "phase": phase,
            "overload": event["overload"],
            "inputs": [tensor_signature(value) for value in event["input_tensors"]],
            "outputs": [tensor_signature(value) for value in event["output_tensors"]],
            "arguments": [argument_signature(value) for value in event["argument_bindings"]],
        }
        condition_id = f"condition::{digest(condition_payload)}"
        condition_units.setdefault(
            condition_id, {"condition_id": condition_id, **condition_payload}
        )
        row = {
            "row_id": f"{args.architecture}::{event['invocation_id']}",
            "invocation": {
                "invocation_id": event["invocation_id"],
                "origin_witness_invocation_id": witness["invocation_id"],
                "ordinal": event["ordinal"],
                "phase": phase,
                "overload": event["overload"],
                "module_context": event["module_context"],
            },
            "mathematical_fb": {
                "status": "FORMULA_REGISTERED_AND_ARGUMENTS_RECORDED",
                "map": formula["map"],
                "adjoint": formula["adjoint"],
                "formula_sha256": digest(formula),
                "analytic_proof_status": (
                    "UNRESOLVED_NO_CONCRETE_SAVED_TENSOR_COTANGENT_PROGRAM_PROOF"
                ),
                "fb_origin_status": fb_status,
                "actual_backward_program_invocations": compact_ids(peer_ids),
                "actual_backward_program_origin_witness_invocations": compact_ids([
                    origin_by_invocation[peer_id]["invocation_id"] for peer_id in peer_ids
                ]),
                "forward_origin_invocations": compact_ids(origin_ids),
                "forward_origin_witness_invocations": compact_ids([
                    origin_by_invocation[origin_id]["invocation_id"] for origin_id in origin_ids
                ]),
                "autograd_sequence_nr": forward_sequence if phase == "FORWARD" else backward_sequence,
            },
            "eager_aot_binding": {
                "status": (
                    "EXACT_EAGER_AUTOGRAD_FB_IDENTITY_AOT_NODE_PENDING"
                    if args.architecture == "qwen" and not fb_status.startswith("UNRESOLVED")
                    else "UNRESOLVED_EAGER_AUTOGRAD_FB_IDENTITY"
                    if args.architecture == "qwen"
                    else "NOT_CAPTURED_EAGER_ONLY_ARCHITECTURE_ATLAS"
                ),
                "aot_node_ids": compact_ids([]),
            },
            "candidate_region_binding": {
                "status": "UNRESOLVED_NO_FULL_MODEL_INVOCATION_TO_CANDIDATE_REGION_BRIDGE",
                "candidate_region_ids": compact_ids([]),
            },
            "numerical_measurement": {
                "status": "FULL_MODEL_SCREEN_EXISTS_WITHOUT_PER_INVOCATION_BINDING" if args.screen else "UNMEASURED",
                "evidence": repo_path(args.screen) if args.screen else None,
            },
            "condition_unit_id": condition_id,
            "bias_verdict": {
                "status": "UNRESOLVED_NO_INVOCATION_LEVEL_CANDIDATE_MEASUREMENT",
                "candidate_correctness_certified": False,
                "directional_bias_certified": False,
            },
        }
        row["row_sha256"] = digest(row)
        rows.append(row)

    if len(rows) != len(events) or len({row["row_id"] for row in rows}) != len(events):
        raise RuntimeError("invocation denominator is not one-row-per-event")
    summary = {
        "actual_invocations": len(rows),
        "forward_invocations": sum(row["invocation"]["phase"] == "FORWARD" for row in rows),
        "backward_invocations": sum(row["invocation"]["phase"] == "BACKWARD" for row in rows),
        "unique_overloads": len(overloads),
        "formula_registered_and_arguments_recorded": len(rows),
        "concrete_invocation_analytic_proof_complete": 0,
        "fb_origin_or_explicit_auxiliary_complete": len(rows) - len(unresolved_fb),
        "unresolved_fb_origin": len(unresolved_fb),
        "exact_candidate_region_binding": 0,
        "invocation_level_numerical_measurement": 0,
        "candidate_correctness_certified": 0,
        "directional_bias_certified": 0,
    }
    payload = {
        "schema": "kernel-analyzer-full-architecture-fail-closed-invocation-ledger-v1",
        "status": "PARTIAL_FAIL_CLOSED",
        "architecture": args.architecture,
        "subject": inventory["coverage"]["subject"],
        "source_inventory": repo_path(args.inventory),
        "source_inventory_sha256": inventory["result_sha256"],
        "origin_witness_inventory": (
            repo_path(args.origin_inventory) if args.origin_inventory else repo_path(args.inventory)
        ),
        "origin_witness_inventory_sha256": origin_inventory["result_sha256"],
        "instrumentation_audit": {
            "baseline_and_weak_observer_invocations": len(events),
            "strong_origin_witness_invocations": len(origin_inventory["trace"]["events"]),
            "observer_induced_extra_invocations_excluded": len(observer_extra_events),
            "observer_induced_extra_overloads": dict(sorted(Counter(
                event["overload"] for event in observer_extra_events
            ).items())),
            "all_nonextra_invocations_exactly_aligned": len(origin_by_invocation) == len(events),
            "base_to_origin_witness_id_map_sha256": digest(sorted(
                (base_id, witness["invocation_id"])
                for base_id, witness in origin_by_invocation.items()
            )),
            "base_and_origin_witness_ids_kept_in_separate_namespaces": True,
            "baseline_vs_weak_loss_and_gradient_exact": all(
                inventory["observation_stability"].get(key, False)
                for key in ("loss_exact", "all_parameter_gradient_digest_exact")
            ),
            "baseline_vs_strong_loss_and_gradient_exact": all(
                origin_inventory["observation_stability"].get(key, False)
                for key in ("loss_exact", "all_parameter_gradient_digest_exact")
            ),
        },
        "screen_evidence": repo_path(args.screen) if args.screen else None,
        "summary": summary,
        "gates": {
            "every_actual_invocation_in_exactly_one_row": True,
            "all_overload_formulas_registered": not missing_formulas,
            "all_concrete_invocations_analytically_proved": False,
            "all_fb_origins_or_explicit_auxiliaries_complete": not unresolved_fb,
            "all_candidate_region_bindings_exact": False,
            "all_invocations_numerically_measured": False,
            "all_invocations_have_bias_verdict": False,
        },
        "condition_units": [condition_units[key] for key in sorted(condition_units)],
        "rows": rows,
        "claim_boundary": (
            "This closes the eager execution denominator, overload-formula registry and actual "
            "forward/backward origin accounting. It does not prove that each concrete invocation's "
            "saved tensors, cotangent and backward arithmetic instantiate the formula, and it does "
            "not infer candidate binding, numerical correctness, or bias safety."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(json.dumps({
        "output": str(args.output),
        "summary": summary,
        "gates": payload["gates"],
        "condition_units": len(condition_units),
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
