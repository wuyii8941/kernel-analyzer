#!/usr/bin/env python3
"""Close one concrete attention softmax+score forward/backward proof."""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"
CID = "qwen_seq128_layer27_attention_softmax_fb"
SEQ = 14308
ALPHA = 0.08838834764831845
FORWARD_LINE = 4154
BACKWARD_LINE = 2511
FORWARD_LINE_SHA = "2967373ee293b3dfbc79aa01411dd71afe1615f4fc3267c1d4cf8f1a9d286179"
BACKWARD_LINE_SHA = "536265df506aecc767de55c88596a5fd8dff95629bed91d4b97aabd3457fc0e8"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def shortest_output_path(
    nodes: dict[str, dict], start: str, outputs: set[str],
) -> list[str] | None:
    queue = collections.deque([(start, [start])])
    seen = {start}
    while queue:
        name, path = queue.popleft()
        if name in outputs:
            return path
        for user in nodes[name]["users"]:
            if user != "output" and user not in seen:
                seen.add(user)
                queue.append((user, path + [user]))
    return None


def ancestors(nodes: dict[str, dict], start: str) -> set[str]:
    result: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name in result:
            continue
        result.add(name)
        stack.extend(nodes[name]["input_nodes"])
    return result


def mean_summary(formal: dict, source: str, metric: str) -> float:
    values = [row["repeats"][0]["summary"][source + "_" + metric]
              for row in formal["states"]]
    return float(sum(values) / len(values))


def max_summary(formal: dict, source: str, metric: str) -> float:
    return float(max(row["repeats"][0]["summary"][source + "_" + metric]
                     for row in formal["states"]))


def main() -> None:
    aot_path = COVERAGE / "qwen_seq128_aot.json.gz"
    formal_path = COVERAGE / "cases/qwen128_softmax_fb_formal.json"
    release = COVERAGE / "runtime_releases/qwen_seq128_r1"
    with gzip.open(aot_path, "rt") as handle:
        capture = json.load(handle)
    formal = json.loads(formal_path.read_text())
    release_capture = json.loads((release / "capture.json").read_text())

    assert formal["candidate_id"] == CID
    assert formal["status"] == "COMPLETE_TYPED_SOFTMAX_FB_FORMAL"
    assert len(formal["states"]) == 32 and all(formal["gates"].values())
    assert set(formal["direction"]) == {
        "kernel", "output_rounding", "saved_state_reconstruction",
        "forward_probability_kernel", "forward_probability_rounding", "semantic_total",
    }
    for source in formal["direction"].values():
        assert source["states"] == 32 and source["coordinates"] == 262144
        assert source["complete_coordinates"] and source["streamed_complete_gram"]

    modules = {row["phase"]: row for row in release_capture["modules"]}
    forward_source_path = Path(modules["FORWARD"]["captured_source"])
    backward_source_path = Path(modules["BACKWARD"]["captured_source"])
    forward_source = forward_source_path.read_text()
    backward_source = backward_source_path.read_text()
    assert hashlib.sha256(forward_source.encode()).hexdigest() == modules["FORWARD"]["sha256"]
    assert hashlib.sha256(backward_source.encode()).hexdigest() == modules["BACKWARD"]["sha256"]
    forward_call = forward_source.splitlines()[FORWARD_LINE - 1].strip()
    backward_call = backward_source.splitlines()[BACKWARD_LINE - 1].strip()
    assert hashlib.sha256(forward_call.encode()).hexdigest() == FORWARD_LINE_SHA
    assert hashlib.sha256(backward_call.encode()).hexdigest() == BACKWARD_LINE_SHA
    assert "softmax_27" in forward_source.splitlines()[FORWARD_LINE - 3]
    assert "softmax_27" in backward_source.splitlines()[BACKWARD_LINE - 3]
    for snippet in (
        "tmp2 = tmp0 * tmp1", "tmp14 = tl.where(tmp11, tmp12, tmp13)",
        "tmp22 = triton_helpers.max2", "tmp24 = libdevice.exp(tmp23)",
        "tmp28 = tl.sum(tmp27, 1)", "tmp31 = (tmp30 / tmp28)",
        "tl.store(out_ptr2", "tl.store(out_ptr0", "tl.store(out_ptr1",
    ):
        assert snippet in forward_source
    for snippet in (
        "tmp5 = tmp3 - tmp4", "tmp6 = libdevice.exp(tmp5)",
        "tmp8 = (tmp6 / tmp7)", "tmp9 = tmp1 * tmp8",
        "tmp13 = tl.sum(tmp12, 1)", "tmp15 = tl.fma(tmp14, tmp13, tmp9)",
        "tmp17 = tl.full([1, 1], 0.08838834764831845",
        "tl.store(in_out_ptr0",
    ):
        assert snippet in backward_source

    graphs = capture["capture"]["graphs"]
    forward = next(graph for graph in graphs if graph["phase"] == "FORWARD")
    backward = next(graph for graph in graphs if graph["phase"] == "BACKWARD")
    fw = {node["name"]: node for node in forward["nodes"]}
    bw = {node["name"]: node for node in backward["nodes"]}

    assert fw["bmm_55"]["input_nodes"] == ["view_475", "view_476"]
    assert fw["bmm_55"]["tensor_meta"][0] == [16, 128, 128]
    assert fw["mul_390"]["arguments"]["args"] == [{"node": "view_477"}, ALPHA]
    assert fw["add_249"]["input_nodes"] == ["alias_27", "mul_390"]
    softmax = fw["_softmax_27"]
    assert softmax["seq_nr"] == SEQ
    assert softmax["target"] == "aten._softmax.default"
    assert softmax["arguments"] == {
        "args": [{"node": "_to_copy_279"}, -1, False], "kwargs": {},
    }
    assert fw["_to_copy_279"]["input_nodes"] == ["add_249"]
    assert fw["detach_138"]["input_nodes"] == ["_softmax_27"]
    assert "detach_138" in fw["output"]["input_nodes"]

    bridge = capture["capture"]["cross_phase_runtime_bridge"]
    assert all(bridge["gates"].values())
    bridge_row = next(row for row in bridge["runs"][0]["backward_inputs"]
                      if row["placeholder"] == "detach_138")
    assert bridge_row["forward_output_matches"] == [{
        "identity_mode": "EXACT_PYTHON_OBJECT",
        "runtime_token": "forward-output:1103:detach_138",
    }]

    assert bw["detach_138"]["op"] == "placeholder"
    assert bw["detach_145"]["input_nodes"] == ["detach_138"]
    assert bw["_to_copy_291"]["input_nodes"] == ["view_505"]
    assert "tangents_1" in ancestors(bw, "_to_copy_291")
    vjp = bw["_softmax_backward_data"]
    assert vjp["target"] == "aten._softmax_backward_data.default"
    assert vjp["seq_nr"] == SEQ
    assert vjp["arguments"] == {
        "args": [{"node": "_to_copy_291"}, {"node": "detach_145"},
                 -1, "torch.float32"],
        "kwargs": {},
    }
    assert bw["_to_copy_292"]["input_nodes"] == ["_softmax_backward_data"]
    assert bw["_to_copy_292"]["arguments"]["kwargs"]["dtype"] == "torch.bfloat16"
    assert bw["mul_414"]["arguments"]["args"] == [{"node": "_to_copy_292"}, ALPHA]
    assert bw["view_506"]["input_nodes"] == ["mul_414"]
    assert set(bw["view_506"]["users"]) == {"bmm_59", "bmm_60"}
    assert all(bw[name]["seq_nr"] == 14303 for name in ("bmm_59", "bmm_60"))
    outputs = set(bw["output"]["input_nodes"])
    q_path = shortest_output_path(bw, "bmm_60", outputs)
    k_path = shortest_output_path(bw, "bmm_59", outputs)
    assert q_path == ["bmm_60", "view_508", "mul_418", "add_261",
                      "transpose_149", "mul_428", "sum_9", "view_516"]
    assert k_path == ["bmm_59", "view_507", "transpose_146", "view_510",
                      "sum_6", "squeeze_1", "mul_416", "add_259",
                      "transpose_148", "mul_420", "sum_7", "view_513"]

    derivation = {
        "symbols": {
            "Q,K": "[16,128,128] BF16 score operands",
            "M": "[1,1,128,128] causal/token mask",
            "G": "dL/dP [1,16,128,128]",
            "alpha": ALPHA,
        },
        "forward": [
            "S = Q K^T",
            "Z = alpha S + M",
            "P_ij = exp(Z_ij - m_i) / sum_k exp(Z_ik - m_i)",
            "m_i = max_j Z_ij",
        ],
        "backward": [
            "dZ_ij = P_ij (G_ij - sum_k G_ik P_ik)",
            "dS = alpha dZ",
            "dQ = dS K",
            "dK = dS^T Q",
        ],
        "aot_program": {
            "forward_score": "bmm_55 -> view_477 -> mul_390 -> add_249",
            "forward_softmax": "_to_copy_279 -> _softmax_27 -> detach_138",
            "saved_probability": "detach_138 -> exact-object bridge -> detach_145",
            "upstream": "tangents_1 -> ... -> view_505 -> _to_copy_291",
            "softmax_vjp": "_softmax_backward_data -> _to_copy_292 -> mul_414",
            "q_vjp_output_path": q_path,
            "k_vjp_output_path": k_path,
        },
        "lowered_saved_state_transform": (
            "Generated forward emits BF16 Z plus FP32 row max/sum; generated backward "
            "reconstructs P from those auxiliaries instead of reading the AOT-saved FP32 P."
        ),
        "generated_forward_call": forward_call,
        "generated_backward_call": backward_call,
    }

    numerical = {}
    for source in (
        "kernel", "output_rounding", "saved_state_reconstruction",
        "forward_probability_kernel", "forward_probability_rounding", "semantic_total",
    ):
        certificate = formal["direction"][source]
        numerical[source] = {
            "mean_l2": mean_summary(formal, source, "l2"),
            "max_abs": max_summary(formal, source, "max_abs"),
            "u_statistic": certificate["cross_state_inner_product_u"],
            "cluster_bootstrap_95": certificate["cluster_bootstrap_95"],
            "direction_status": certificate["status"],
        }
    coherent = formal["coherent_sources"]
    total_coherent = formal["direction"]["semantic_total"]["status"] == "PASS"
    verdict = "COHERENT_BIAS" if total_coherent else "NONCOHERENT_PRECISION_ERROR"
    payload = {
        "schema": "kernel-analyzer-concrete-softmax-fb-proof-v1",
        "status": "COMPLETE_CONCRETE_FB_TYPED_PRECISION_" + (
            "BIAS" if coherent else "NEGATIVE"
        ),
        "candidate_id": CID,
        "subject": "Qwen3-1.7B seq128 layer-27 attention score softmax F+B",
        "verdict": verdict,
        "coherent_sources": coherent,
        "semantic_total_coherent": total_coherent,
        "forward_backward_unit": derivation,
        "concrete_program_proof": {
            "forward_program_exact": True,
            "saved_tensor_origin_exact": True,
            "cross_phase_object_identity_exact": True,
            "cotangent_edge_exact": True,
            "analytic_vjp_matches_typed_aten": True,
            "q_and_k_output_edges_exact": True,
            "generated_forward_source_exact": True,
            "generated_backward_source_exact": True,
            "coordinatewise_decomposition_exact": True,
            "forward_program_sha256": forward["code_sha256"],
            "backward_program_sha256": backward["code_sha256"],
            "analytic_derivation_sha256": canonical(derivation),
        },
        "numerical": {
            "states": len(formal["states"]),
            "coordinates_per_state": 262144,
            "two_repeats_exact": formal["gates"]["two_repeats_exact"],
            "sources": numerical,
            "max_reconstructed_probability_row_sum_error": max(
                row["repeats"][0]["summary"]
                ["reconstructed_probability_row_sum_max_error"]
                for row in formal["states"]
            ),
            "max_reconstructed_vs_semantic_probability": max(
                row["repeats"][0]["summary"]
                ["reconstructed_vs_semantic_probability_max_abs"]
                for row in formal["states"]
            ),
            "max_forward_generated_vs_rounded_semantic_probability": max(
                row["repeats"][0]["summary"]["forward"]
                ["generated_probability_vs_rounded_semantic_max_abs"]
                for row in formal["states"]
            ),
            "max_analytic_formula_vs_typed_aten_vjp": max(
                row["repeats"][0]["summary"]["formula_vs_typed_aten_max_abs"]
                for row in formal["states"]
            ),
        },
        "bindings": {
            "aot_capture_sha256": capture["result_sha256"],
            "aot_file_sha256": hashlib.sha256(aot_path.read_bytes()).hexdigest(),
            "release_capture_sha256": release_capture["result_sha256"],
            "forward_generated_source_sha256": modules["FORWARD"]["sha256"],
            "backward_generated_source_sha256": modules["BACKWARD"]["sha256"],
            "forward_source_line": FORWARD_LINE,
            "backward_source_line": BACKWARD_LINE,
            "formal_result_sha256": formal["result_sha256"],
            "formal_file_sha256": hashlib.sha256(formal_path.read_bytes()).hexdigest(),
            "sequence_nr": SEQ,
        },
        "claim_boundary": (
            "Complete concrete local F+B mathematical and program proof for one layer-27 "
            "softmax score unit, including the lowered F-to-B saved-state transformation. "
            "The direction verdict concerns this exact invocation over 32 frozen natural "
            "states; it neither proves the whole softmax family safe nor establishes a "
            "downstream weight-accumulation mechanism."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    output = COVERAGE / "cases/qwen128_softmax_fb.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)),
                      "status": payload["status"], "verdict": verdict,
                      "coherent_sources": coherent}))


if __name__ == "__main__":
    main()
