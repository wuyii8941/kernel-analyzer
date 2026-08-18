#!/usr/bin/env python3
"""Bind terminal generated outputs to actual parameter-gradient endpoints."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import transformers
import triton
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from scripts.aot_capture import AOTForwardBackwardCapture  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    gradient_digest, load_model, tensor_digest,
)
from scripts.same_dtype_semantic_observer import (  # noqa: E402
    SameDtypeSemanticCandidateObserver,
)


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-graph-breaks", action="store_true")
    args = parser.parse_args()

    plan = load(args.task_plan)
    inventory = load(args.inventory)
    campaign = load(args.campaign)
    pending = [
        row for row in plan["rows"]
        if row["status"].startswith("UNRESOLVED")
    ]
    if not pending:
        raise RuntimeError("task plan has no terminal unresolved output")

    frozen_artifact = load(args.campaign.parent / "environment.json")
    expected_environment = frozen_artifact["environment"]
    actual_environment = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "triton_version": triton.__version__,
    }
    if actual_environment != expected_environment:
        raise RuntimeError("parameter-gradient binding environment mismatch")

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    tokens = states[0].get("token_ids", states[0].get("input_ids"))
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not args.allow_graph_breaks, dynamic=False,
    )
    values = torch.tensor([tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(values).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])

    model.zero_grad(set_to_none=True)
    baseline_loss = candidate(values)
    baseline_loss.backward()
    torch.cuda.synchronize(device)
    baseline = {
        "loss": tensor_digest(baseline_loss),
        "gradients": gradient_digest(model),
    }

    aliases_by_parameter: dict[int, list[str]] = {}
    parameter_by_id: dict[int, torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases_by_parameter.setdefault(id(parameter), []).append(name)
        parameter_by_id[id(parameter)] = parameter

    runs = []
    for repeat in range(2):
        captured: dict[str, torch.Tensor] = {}
        incoming_gradients: dict[int, list[torch.Tensor]] = {
            parameter_id: [] for parameter_id in parameter_by_id
        }
        hooks = []
        for parameter_id, parameter in parameter_by_id.items():
            def retain_gradient(
                gradient: torch.Tensor, *, _parameter_id: int = parameter_id,
            ) -> torch.Tensor:
                incoming_gradients[_parameter_id].append(gradient)
                return gradient

            hooks.append(parameter.register_hook(retain_gradient))

        def sink(task_id: str, tensor: torch.Tensor, metadata: Any) -> None:
            del metadata
            if task_id in captured:
                raise RuntimeError(f"terminal candidate endpoint repeated: {task_id}")
            captured[task_id] = tensor

        torch.manual_seed(24000)
        torch.cuda.manual_seed_all(24000)
        model.zero_grad(set_to_none=True)
        observer = SameDtypeSemanticCandidateObserver(
            modules=modules,
            campaign_rows=campaign["rows"],
            inventory_rows=inventory["runtime_call_audit"]["rows"],
            task_rows=pending,
            include_unresolved_tasks=True,
            sink=sink,
        )
        try:
            with observer:
                loss = candidate(values)
                loss.backward()
            torch.cuda.synchronize(device)
        finally:
            # A failed candidate/observer run must not leave parameter hooks
            # attached to the long-lived model used by later retries.
            for hook in hooks:
                hook.remove()
        observer.validate()
        identity = {"loss": tensor_digest(loss), "gradients": gradient_digest(model)}
        if identity != baseline:
            raise RuntimeError("parameter-gradient binding observer perturbed candidate")
        rows = []
        for task in pending:
            task_id = str(task["task_id"])
            value = captured[task_id]
            matches = []
            for parameter_id, gradients in incoming_gradients.items():
                modes = sorted({
                    mode for gradient in gradients
                    if (
                        mode := AOTForwardBackwardCapture._runtime_identity_mode(
                            value, gradient
                        )
                    ) is not None
                })
                if modes:
                    matches.append({
                        "parameter_aliases": sorted(aliases_by_parameter[parameter_id]),
                        "identity_mode": modes[0],
                        "accumulate_grad_hook_inputs": len(gradients),
                    })
            if len(matches) != 1:
                raise RuntimeError(
                    f"terminal output does not bind one parameter gradient: {task_id}: {matches}"
                )
            rows.append({"task_id": task_id, **matches[0]})
        runs.append({"repeat": repeat, "rows": rows})

    if runs[0]["rows"] != runs[1]["rows"]:
        raise RuntimeError("parameter-gradient runtime binding changed across repeats")
    rows = []
    for row in runs[0]["rows"]:
        endpoint = "parameter_gradient:" + "|".join(row["parameter_aliases"])
        rows.append({
            **row,
            "semantic_endpoint_id": endpoint,
            "name_shape_ordinal_or_candidate_value_pairing_used": False,
        })
    payload = {
        "schema": "kernel-analyzer-parameter-gradient-closure-v1",
        "status": "COMPLETE_TERMINAL_OUTPUTS_BOUND_TO_PARAMETER_GRADIENTS",
        "bindings": {
            "task_plan_result_sha256": plan["result_sha256"],
            "proof_capture_result_sha256": plan["bindings"]["proof_capture_result_sha256"],
            "inventory_result_sha256": inventory["result_sha256"],
            "campaign_result_sha256": campaign["result_sha256"],
            "environment_result_sha256": frozen_artifact["result_sha256"],
        },
        "denominator": {"terminal_outputs": len(pending), "bound": len(rows), "unresolved": 0},
        "runs": runs,
        "rows": rows,
        "gates": {
            "repeat_stable": True,
            "exact_runtime_object_or_storage_identity_only": True,
            "candidate_execution_unmodified": True,
        },
        "claim_boundary": (
            "Terminal generated outputs are bound to actual post-accumulation parameter gradients "
            "by same-run object/storage identity. Numerical equivalence is not granted."
        ),
    }
    payload["result_sha256"] = digest(payload)
    write(args.output, payload)
    print(json.dumps({"output": str(args.output), "denominator": payload["denominator"]}))


if __name__ == "__main__":
    main()
