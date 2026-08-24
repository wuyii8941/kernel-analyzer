#!/usr/bin/env python3
"""Sham-controlled FP32 accumulation repair for one frozen external MM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import linecache
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.streaming import StreamingGramAccumulator  # noqa: E402
from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.generated_nontriton_fp32_observer import fp32_external_reference  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402


CID = "qwen_seq128_forward_8_output"
TARGET_SHA = "1847d6184bdf781a1b57531571a298c898337332aa694e47a96de633d30ed2af"
CARRIER = "model.layers.0.self_attn.v_proj.weight"


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class VProjRepair:
    def __init__(
        self,
        modules: list[Any],
        mode: str,
        target_sha: str | None,
        expected_output_shape: tuple[int, ...] | None = None,
        expected_match_index: int | None = None,
    ) -> None:
        self.modules = modules
        self.mode = mode
        self.target_sha = target_sha
        self.expected_output_shape = expected_output_shape
        self.expected_match_index = expected_match_index
        self.restores: list[tuple[Any, Any]] = []
        self.calls = 0
        self.local: dict[str, Any] | None = None
        self.seen: list[dict[str, Any]] = []

    def __enter__(self) -> "VProjRepair":
        seen: set[int] = set()
        for module in self.modules:
            namespace = getattr(module, "extern_kernels", None)
            if namespace is None or id(namespace) in seen:
                continue
            seen.add(id(namespace))
            original = namespace.mm

            def wrapped(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                filename, line, digest = _source_identity()
                if self.target_sha is not None and digest != self.target_sha:
                    return _original(*args, **kwargs)
                result = _original(*args, **kwargs)
                candidate = kwargs.get("out", result)
                if not isinstance(candidate, torch.Tensor):
                    raise RuntimeError("target MM has no tensor output")
                if (
                    self.expected_output_shape is not None
                    and tuple(candidate.shape) != self.expected_output_shape
                ):
                    return result
                self.match_count = getattr(self, "match_count", 0) + 1
                if (
                    self.expected_match_index is not None
                    and self.match_count != self.expected_match_index
                ):
                    return result
                self.seen.append({
                    "source_sha": digest,
                    "source_line": line,
                    "source_text": linecache.getline(filename, line).strip(),
                    "shape": list(candidate.shape),
                })
                before = candidate.detach().clone()
                if self.mode == "SHAM":
                    reference = torch.mm(args[0], args[1])
                elif self.mode == "REPAIR_FP32_CAST_BF16":
                    reference = fp32_external_reference("mm", args, kwargs)
                else:
                    raise ValueError(self.mode)
                delivered = reference.to(candidate.dtype)
                candidate.copy_(delivered)
                delta = before.float() - delivered.float()
                candidate_error = before.float() - reference.float()
                delivered_error = delivered.float() - reference.float()
                candidate_sse = float(
                    candidate_error.double().square().sum().item()
                )
                delivered_sse = float(
                    delivered_error.double().square().sum().item()
                )
                self.local = {
                    "coordinates": delta.numel(),
                    "changed_coordinates": int(torch.count_nonzero(delta).item()),
                    "max_abs_intervention": float(delta.abs().max().item()),
                    "l2_intervention": float(torch.linalg.vector_norm(delta).item()),
                    "candidate_vs_fp32_l2": float(torch.linalg.vector_norm(candidate_error).item()),
                    "repair_vs_fp32_l2": float(torch.linalg.vector_norm(delivered_error).item()),
                    "candidate_vs_fp32_max_abs": float(candidate_error.abs().max().item()),
                    "repair_vs_fp32_max_abs": float(delivered_error.abs().max().item()),
                    "candidate_vs_fp32_sse": candidate_sse,
                    "repair_vs_fp32_sse": delivered_sse,
                    "fp32_sse_reduction": candidate_sse - delivered_sse,
                }
                self.calls += 1
                return result

            namespace.mm = wrapped
            self.restores.append((namespace, original))
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        for namespace, original in self.restores:
            namespace.mm = original
        if self.calls != 1:
            unique = []
            for item in self.seen:
                if item not in unique:
                    unique.append(item)
            raise RuntimeError(
                f"target repair executed {self.calls} times; "
                f"matching source/shape calls={unique[:16]}"
            )


def run(model: torch.nn.Module, candidate: Any, values: torch.Tensor, seed: int,
        modules: list[Any], mode: str | None, target_sha: str) -> tuple[dict[str, str], dict[str, Any] | None]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    observer = VProjRepair(modules, mode, target_sha) if mode else None
    if observer:
        with observer:
            loss = candidate(values)
            loss.backward()
    else:
        loss = candidate(values)
        loss.backward()
    torch.cuda.synchronize(values.device)
    identity = {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}
    return identity, observer.local if observer else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), default="qwen")
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, default=ROOT / "results/coverage/qwen_seq128_input_bank.json")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "results/coverage/runtime_releases/qwen_seq128_r1")
    parser.add_argument("--candidate-id", default=CID)
    parser.add_argument("--target-sha", default=TARGET_SHA)
    parser.add_argument("--carrier", default=CARRIER)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.states < 2:
        raise ValueError("carrier direction requires at least two states")
    queue = json.loads((ROOT / "results/coverage/bias_candidate_queue.json").read_text())
    bound = next(
        row for row in queue["candidates"] if row["candidate_id"] == args.candidate_id
    )
    if (
        bound["exact_generated_call"]["function"] != "extern_kernels.mm"
        or bound["exact_generated_call"]["source_line_sha256"] != args.target_sha
    ):
        raise RuntimeError("candidate ID does not bind the declared MM source identity")
    if args.output is None:
        args.output = ROOT / "results/coverage/cases/qwen128_vproj_repair_pilot.json"
    bank_path = args.input_bank
    bank = json.loads(bank_path.read_text())
    states = bank.get("states", bank.get("records"))[:args.states]
    release = args.release_dir
    capture = json.loads((release / "capture.json").read_text())
    if file_digest(bank_path) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank does not match frozen release")
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    parameters = dict(model.named_parameters())
    if args.carrier not in parameters:
        raise RuntimeError("declared carrier parameter is absent")
    start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not bool(capture.get("allow_graph_breaks", False)), dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)

    spool = Path("/data1/tzh/cache/kernel_analyzer_contrasts/mm_accumulation_repair") / args.candidate_id
    carrier = StreamingGramAccumulator(spool, args.carrier, chunk_elements=1_048_576)
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        state_id = str(state.get("sequence_id", state.get("state_id", index)))
        tokens = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 24000 + index
        standard, _ = run(model, candidate, values, seed, modules, None, args.target_sha)
        standard_carrier = parameters[args.carrier].grad.detach().float().cpu().clone()
        sham, sham_local = run(model, candidate, values, seed, modules, "SHAM", args.target_sha)
        if sham != standard or not sham_local or sham_local["changed_coordinates"] != 0:
            raise RuntimeError("same-dtype sham changed endpoints")
        repair, repair_local = run(
            model, candidate, values, seed, modules, "REPAIR_FP32_CAST_BF16", args.target_sha
        )
        if repair_local is None:
            raise RuntimeError("repair emitted no local summary")
        repair_carrier = parameters[args.carrier].grad.detach().float().cpu()
        carrier_delta = standard_carrier - repair_carrier
        carrier.add_array(state_id, carrier_delta.numpy())
        row = {
            "state_id": state_id,
            "standard_endpoint": standard,
            "sham_endpoint": sham,
            "repair_endpoint": repair,
            "sham_local": sham_local,
            "repair_local": repair_local,
            "repair_loss_changed": repair["loss"] != standard["loss"],
            "repair_parameter_gradients_changed": repair["gradients"] != standard["gradients"],
            "carrier_delta_l2": float(torch.linalg.vector_norm(carrier_delta).item()),
            "carrier_delta_max_abs": float(carrier_delta.abs().max().item()),
            "carrier_changed_coordinates": int(torch.count_nonzero(carrier_delta).item()),
        }
        rows.append(row)
        write(args.output, {
            "schema": "kernel-analyzer-external-mm-accumulation-repair-v2",
            "status": "RUNNING", "candidate_id": args.candidate_id, "states": rows,
        })
        del standard_carrier, repair_carrier, carrier_delta, values
        torch.cuda.empty_cache()
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id,
                          "local_changed": repair_local["changed_coordinates"],
                          "carrier_changed": row["carrier_changed_coordinates"]}), flush=True)

    certificate = carrier.finalize(bootstrap_draws=4000, seed=14031, cleanup=True)
    local_improves = all(
        row["repair_local"]["fp32_sse_reduction"] > 0.0 for row in rows
    )
    nonnull = all(row["repair_local"]["changed_coordinates"] > 0 for row in rows)
    sham_exact = all(row["standard_endpoint"] == row["sham_endpoint"] for row in rows)
    carrier_nonnull = all(row["carrier_changed_coordinates"] > 0 for row in rows)
    gates = {
        "restoration_sham_exact": sham_exact,
        "accumulation_intervention_nonnull_every_state": nonnull,
        "accumulation_intervention_reduces_fp32_sse_every_state": local_improves,
        "direct_weight_carrier_nonnull_every_state": carrier_nonnull,
        "direct_weight_carrier_coherent": certificate["status"] == "PASS",
    }
    accepted = all(gates.values())
    payload = {
        "schema": "kernel-analyzer-external-mm-accumulation-repair-v2",
        "status": (
            "COMPLETE_LOCAL_ACCUMULATION_CAUSAL_PILOT_ACCEPTED"
            if accepted else "COMPLETE_LOCAL_ACCUMULATION_CAUSAL_PILOT_REJECTED"
        ),
        "candidate_id": args.candidate_id,
        "architecture": args.architecture,
        "intervention": "Exact generated v_proj MM in FP32 followed by required BF16 ABI cast",
        "states": rows,
        "hypothesis_tested": (
            "The coherent precision error is caused by the generated MM's local "
            "accumulation, holding its BF16 operands and BF16 output ABI fixed."
        ),
        "gates": gates,
        "direct_weight_carrier": {"parameter": args.carrier, **certificate},
        "release_capture_sha256": capture["result_sha256"],
        "input_bank_sha256": file_digest(bank_path),
        "claim_boundary": (
            "This pilot isolates only local MM accumulation from output rounding and "
            "inherited operand error. Rejection does not reject the original T1 precision "
            "bias; it rejects this local-accumulation explanation. No T4 claim."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    write(args.output, payload)
    print(json.dumps({"event": "PILOT_COMPLETE", "status": payload["status"],
                      "gates": payload["gates"]}))


if __name__ == "__main__":
    main()
