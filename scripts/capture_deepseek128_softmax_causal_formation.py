#!/usr/bin/env python3
"""Causally localize DeepSeek seq128 attention-softmax bias formation.

The experiment uses independently recompiled FP32-pointer Triton programs and
never passes FP32 storage through a frozen BF16 ABI.  It compares four matched
arms under one complete F+B step: sham, forward saved-logit repair, backward
softmax-VJP repair, and their joint repair.  The nearest exact AOT-reachable
parameter carrier is layer-10 q_norm.weight.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
                str(ROOT / "archive/round1_code/src")]

from kernel_analyzer.bias_formation_v21 import (  # noqa: E402
    FormationPolicy,
    summarize_streamed_state_vector_files,
)
from scripts.generated_fp32_observer import (  # noqa: E402
    promoted_pointer_arguments,
    runtime_signature,
    validate_typed_triton_reference_abi,
)
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import load_model  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402
from scripts.typed_triton_reference import compile_fp32_pointer_kernels  # noqa: E402


MODEL = Path("/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
BANK = ROOT / "results/coverage/deepseek8b_seq128_input_bank.json"
RELEASE = ROOT / "results/coverage/runtime_releases/deepseek8b_seq128_r1"
CARRIER = "model.layers.10.self_attn.q_norm.weight"
TARGETS = {
    "FORWARD_SAVED_A": "forward:191",
    "BACKWARD_VJP": "backward:1529",
}
LR = 1.0e-4


def load_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def spool(root: Path, arm: str, layer: str, partition: str,
          state_id: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu().contiguous().numpy().reshape(-1)
    target = root / arm / layer / partition / f"{state_id}.f32"
    target.parent.mkdir(parents=True, exist_ok=True)
    value.tofile(target)
    return {
        "state_id": state_id, "path": str(target),
        "storage_dtype": "float32", "coordinate_count": int(value.size),
    }


class TypedEndpointController:
    def __init__(self, modules: list[Any], campaign_rows: list[dict[str, Any]]) -> None:
        self.modules = modules
        self.rows = {row["region_id"]: row for row in campaign_rows
                     if row["region_id"] in TARGETS.values()}
        if set(self.rows) != set(TARGETS.values()):
            raise RuntimeError("target campaign rows are incomplete")
        expected = {str(row["symbol"]): str(row["embedded_program_sha256"])
                    for row in self.rows.values()}
        self.references, self.reference_metadata = compile_fp32_pointer_kernels(
            modules, expected_program_sha256=expected, selected_symbols=set(expected))
        self.runtime = {}
        for module in modules:
            for symbol, value in vars(module).items():
                if symbol in expected and callable(getattr(value, "run", None)):
                    if symbol in self.runtime and self.runtime[symbol] is not value:
                        raise RuntimeError("target symbol has multiple runtime kernels")
                    self.runtime[symbol] = value
        if set(self.runtime) != set(expected):
            raise RuntimeError("target runtime kernels are incomplete")
        all_rows = defaultdict(list)
        for row in campaign_rows:
            all_rows[str(row["symbol"])].append(row)
        self.target_indices = {
            region: next(i for i, row in enumerate(all_rows[str(target["symbol"])])
                         if row["region_id"] == region)
            for region, target in self.rows.items()
        }

    @contextmanager
    def apply(self, repair_regions: set[str]) -> Iterator[dict[str, torch.Tensor]]:
        counts: dict[str, int] = defaultdict(int)
        records: dict[str, torch.Tensor] = {}
        restores = []
        region_by_symbol_index = {
            (str(row["symbol"]), self.target_indices[region]): region
            for region, row in self.rows.items()
        }
        for symbol, kernel in self.runtime.items():
            original = kernel.run

            def wrapped(*args: Any, _symbol: str = symbol,
                        _kernel: Any = kernel, _original: Any = original,
                        **kwargs: Any) -> Any:
                index = counts[_symbol]; counts[_symbol] += 1
                region = region_by_symbol_index.get((_symbol, index))
                if region is None:
                    return _original(*args, **kwargs)
                promoted_args, promoted_tensors = promoted_pointer_arguments(args)
                reference = self.references[_symbol]
                validate_typed_triton_reference_abi(reference, promoted_args)
                result = _original(*args, **kwargs)
                reference.run(*promoted_args, **kwargs)
                pointer_names = [name for name, annotation in runtime_signature(_kernel)
                                 if str(annotation).startswith("*")]
                candidate_tensors = [value for value in args if isinstance(value, torch.Tensor)]
                candidate = dict(zip(pointer_names, candidate_tensors))["in_out_ptr0"]
                reference_value = dict(zip(pointer_names, promoted_tensors))["in_out_ptr0"]
                cast_reference = reference_value.to(dtype=candidate.dtype)
                records[region] = candidate.detach().float().cpu() - cast_reference.detach().float().cpu()
                if region in repair_regions:
                    candidate.copy_(cast_reference)
                return result

            restores.append((kernel, original)); kernel.run = wrapped
        try:
            yield records
        finally:
            for kernel, original in reversed(restores):
                kernel.run = original
        if set(records) != set(TARGETS.values()):
            raise RuntimeError(f"target invocation mismatch: observed={sorted(records)}")


def branch(model: torch.nn.Module, candidate: Any, controller: TypedEndpointController,
           values: torch.Tensor, seed: int, repairs: set[str]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model.zero_grad(set_to_none=True)
    with controller.apply(repairs) as records:
        loss = candidate(values); loss.backward()
    torch.cuda.synchronize(values.device)
    gradient = dict(model.named_parameters())[CARRIER].grad.detach().float().cpu().clone()
    return gradient, {key: value.clone() for key, value in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spool-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.states not in (2, 32):
        raise ValueError("use two-state engineering or formal 32-state formation")
    bank = json.loads(BANK.read_text()); states = bank.get("states", bank.get("records"))
    if len(states) < args.states:
        raise RuntimeError("input bank is incomplete")
    capture = json.loads((RELEASE / "capture.json").read_text())
    campaign = load_gzip(RELEASE / "campaign.json.gz")
    device = torch.device(args.device)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    configure_candidate_runtime(24000)
    model = load_model("deepseek8", MODEL, device)
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm_tokens = states[0].get("input_ids", states[0].get("token_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    validate_release(wrapper_modules(modules), capture)
    controller = TypedEndpointController(modules, campaign["rows"])
    args.spool_dir.mkdir(parents=True, exist_ok=True)
    observations: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {layer: [] for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE")}
        for arm in ("FORWARD_REPAIR", "BACKWARD_REPAIR", "JOINT_REPAIR")
    }
    reach = []
    for index, state in enumerate(states[:args.states]):
        state_id = str(state.get("state_id", state.get("sequence_id", index)))
        partition = "calibration" if index < 16 else "confirmation"
        tokens = state.get("input_ids", state.get("token_ids"))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        seed = 52000 + index
        baseline, baseline_records = branch(model, candidate, controller, values, seed, set())
        sham, sham_records = branch(model, candidate, controller, values, seed, set())
        if not torch.equal(baseline, sham):
            raise RuntimeError("typed observation sham changed the declared carrier")
        arms = {
            "FORWARD_REPAIR": ({TARGETS["FORWARD_SAVED_A"]}, TARGETS["FORWARD_SAVED_A"]),
            "BACKWARD_REPAIR": ({TARGETS["BACKWARD_VJP"]}, TARGETS["BACKWARD_VJP"]),
            "JOINT_REPAIR": (set(TARGETS.values()), TARGETS["BACKWARD_VJP"]),
        }
        state_reach = {"state_id": state_id}
        for arm, (repair_regions, local_region) in arms.items():
            repaired, records = branch(model, candidate, controller, values, seed, repair_regions)
            gradient_delta = baseline - repaired
            state_reach[arm] = {
                "gradient_error_energy": float(torch.sum(gradient_delta.double() ** 2)),
                "carrier_reached": bool(torch.count_nonzero(gradient_delta)),
                "local_error_energy": float(torch.sum(records[local_region].double() ** 2)),
            }
            if args.states == 32:
                observations[arm]["LOCAL_ENDPOINT"].append(spool(
                    args.spool_dir, arm, "local", partition, state_id, records[local_region]))
                observations[arm]["PARAMETER_GRADIENT"].append(spool(
                    args.spool_dir, arm, "gradient", partition, state_id, gradient_delta))
                observations[arm]["EFFECTIVE_UPDATE"].append(spool(
                    args.spool_dir, arm, "update", partition, state_id, -LR * gradient_delta))
        reach.append(state_reach)
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)
        del values, baseline, sham, baseline_records, sham_records
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "kernel-analyzer-deepseek128-softmax-causal-formation-v1",
        "status": "ENGINEERING_REACH" if args.states == 2 else "COMPLETE",
        "subject": "DeepSeek seq128 layer10 attention softmax forward/backward",
        "targets": TARGETS, "carrier": CARRIER,
        "reference": "INDEPENDENT_TYPED_FP32_POINTER_TRITON_PROGRAM",
        "arms": reach,
        "sham_exact_on_carrier": True,
        "reference_metadata": controller.reference_metadata,
    }
    if args.states == 32:
        policy = FormationPolicy(min_states=16, bootstrap_samples=2000)
        populations = {}
        for arm in observations:
            populations[arm] = {}
            for partition in ("calibration", "confirmation"):
                populations[arm][partition] = {}
                for layer in ("LOCAL_ENDPOINT", "PARAMETER_GRADIENT", "EFFECTIVE_UPDATE"):
                    rows = [row for row in observations[arm][layer]
                            if partition in Path(row["path"]).parts]
                    populations[arm][partition][layer] = summarize_streamed_state_vector_files(
                        rows, layer=layer, partition=partition, policy=policy).as_dict()
        payload["populations"] = populations
    target = args.output_dir / ("engineering_reach.json" if args.states == 2 else "formation.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(args.spool_dir)
    print(json.dumps({"event": "COMPLETE", "output": str(target)}), flush=True)


if __name__ == "__main__":
    main()
