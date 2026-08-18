#!/usr/bin/env python3
"""Run strict 32-state complete-carrier T3 for all positive T2 rows in one cell."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/tzh/miniconda3/envs/pt_nightly/bin/python")
CONFIG = {
    "qwen": ("qwen", "/data1/tzh/models/Qwen/Qwen3-1.7B", False),
    "phi4": ("phi", "/data1/tzh/models/microsoft/Phi-4-mini-instruct", True),
    "mamba": ("mamba", "/data1/tzh/models/state-spaces/mamba-130m-hf", False),
    "deepseek8b": ("deepseek8", "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", False),
}


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def pack_spool_batches(cases: list[dict], max_bytes: int) -> list[tuple[list[dict], int]]:
    """Deterministically minimize model reloads without changing case evidence."""
    bins: list[tuple[list[dict], int]] = []
    ordered = sorted(
        cases,
        key=lambda case: (
            -int(case["carrier_numel"]) * 32 * 8,
            str(case["task_id"]),
        ),
    )
    for case in ordered:
        # run_t3_carrier_batch persists complete signed carrier deltas as f64.
        estimated = int(case["carrier_numel"]) * 32 * 8
        # An individually oversized complete carrier remains a singleton.  It
        # cannot be split without changing the complete-vector T3 certificate.
        if estimated > max_bytes:
            bins.append(([case], estimated))
            continue
        for index, (batch, used) in enumerate(bins):
            if used <= max_bytes and used + estimated <= max_bytes:
                batch.append(case)
                bins[index] = (batch, used + estimated)
                break
        else:
            bins.append(([case], estimated))
    return bins


def reachable_carriers(row: dict) -> tuple[set[str], dict[str, int]]:
    """Return carriers reached in every pilot state without reading direction."""
    discovery = row.get("carrier_discovery_first_state", [])
    if discovery:
        names = {
            str(value["parameter"]) for value in discovery
            if value.get("status") != "PRESENCE_CHANGED"
        }
        sizes = {
            str(value["parameter"]): int(value["parameter_numel"])
            for value in discovery if "parameter_numel" in value
        }
        if names <= set(sizes):
            return names, sizes
    per_observation = []
    sizes: dict[str, int] = {}
    for state in row["states"]:
        repair = state["repeats"][0]["arms"]["REPAIR"]["gradient_delta"]["parameters"]
        names = set()
        for parameter in repair:
            if parameter.get("status") == "PRESENCE_CHANGED":
                continue
            name = str(parameter["parameter"])
            if "parameter_numel" not in parameter:
                return set(), {}
            names.add(name); sizes[name] = int(parameter["parameter_numel"])
        per_observation.append(names)
    stable = set.intersection(*per_observation) if per_observation else set()
    return stable, sizes


def select_local_carrier(row: dict, campaign_by_region: dict[str, dict],
                         aot_modules_by_task: dict[str, list[str]]) -> tuple[str, int] | None:
    """Select the F+B-local real parameter from frozen semantic source nodes."""
    stable, sizes = reachable_carriers(row)
    if not stable:
        return None
    task_id = str(row["task_id"]); region = ":".join(task_id.split(":")[:2])
    campaign_row = campaign_by_region.get(region)
    source_nodes = ([str(value) for value in campaign_row["source_nodes"]]
                    if campaign_row is not None else [])
    candidates: list[str] = []
    for module in aot_modules_by_task.get(task_id, []):
        candidates.extend([module + ".weight", module + ".bias"])
        if module.endswith(".self_attn"):
            candidates.extend(
                module + suffix for suffix in (
                    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
                    ".o_proj.weight", ".q_norm.weight", ".k_norm.weight",
                )
            )
        # An input-gradient endpoint inside an attention projection is
        # topologically upstream of that projection's already-materialized
        # parameter gradient.  Its nearest still-live parameter carrier can
        # therefore be the enclosing layer norm.  Propose these only for the
        # exact enclosing transformer layer and still require intersection
        # with the T2-observed real gradient delta below.
        if re.fullmatch(r"(?:model\.)?layers\.\d+", module):
            candidates.extend([
                module + ".input_layernorm.weight",
                module + ".post_attention_layernorm.weight",
            ])
    for prefix, parameter in (("q_embed", "q_proj"), ("k_embed", "k_proj")):
        matches = [re.fullmatch(prefix + r"(?:_(\d+))?", value) for value in source_nodes]
        layers = [int(match.group(1) or 0) for match in matches if match]
        if layers:
            layer = max(layers)
            candidates.append(f"model.layers.{layer}.self_attn.{parameter}.weight")
            norm = "q_norm" if parameter == "q_proj" else "k_norm"
            candidates.append(f"model.layers.{layer}.self_attn.{norm}.weight")
    rsqrts = [int(match.group(1)) for value in source_nodes
              if (match := re.fullmatch(r"rsqrt_(\d+)", value))]
    if rsqrts:
        number = max(rsqrts)
        if number % 4 == 3:
            candidates.append(f"model.layers.{(number - 3) // 4}.self_attn.q_proj.weight")
        elif number % 4 == 0:
            candidates.append(f"model.layers.{number // 4 - 1}.mlp.gate_proj.weight")
    attention = [int(match.group(1)) for value in source_nodes
                 if (match := re.fullmatch(r"attn_output_(\d+)", value))]
    if attention and max(attention) % 4 == 0:
        layer = max(attention) // 4
        candidates.extend([
            f"model.layers.{layer}.self_attn.o_proj.weight",
            f"model.layers.{layer}.post_attention_layernorm.weight",
            f"model.layers.{layer + 1}.input_layernorm.weight",
        ])
    local = [name for name in candidates if name in stable]
    if not local:
        return None
    name = local[0]
    return name, sizes[name]


def aot_modules_for_tasks(cell: str, plan: dict, bridge: dict) -> dict[str, list[str]]:
    """Recover module topology for external AOT MM/BMM endpoints."""
    capture_sha = str(bridge["bindings"]["aot_capture_sha256"])
    capture = None
    prefix, seq_token, _repeat = cell.rsplit("_", 2)
    candidates = [
        ROOT / "results/coverage/runtime_releases" / cell / "default_aot_capture.json.gz",
        ROOT / "results/coverage/runtime_releases" / cell / "default_aot_capture_raw.json.gz",
    ]
    candidates += sorted(
        (ROOT / "results/coverage/standard_aot").glob(
            f"{prefix}_{seq_token}*capture*.json.gz"
        )
    )
    for path in candidates:
        if not path.exists():
            continue
        payload = load(path)
        if str(payload.get("capture", {}).get("capture_sha256")) == capture_sha:
            capture = payload["capture"]; break
    if capture is None:
        return {}
    nodes = {}
    for graph in capture["graphs"]:
        graph_index = int(graph["graph_index"])
        for node in graph["nodes"]:
            nodes[f'{str(node["phase"]).lower()}:graph{graph_index}:{node["name"]}'] = node
    bridge_by_region = {str(value["candidate_region_id"]): value for value in bridge["rows"]}
    result = {}
    for task in plan["rows"]:
        task_id = str(task["task_id"]); region = str(task["candidate_region_id"])
        modules = []
        for node_id in bridge_by_region.get(region, {}).get("aot_node_ids", []):
            node = nodes.get(str(node_id), {})
            stack = node.get("nn_module_stack") or node.get("fwd_nn_module_stack") or {}
            for value in stack.values():
                if not isinstance(value, (list, tuple)) or not value:
                    continue
                path = str(value[0])
                marker = ".model."
                if marker in path:
                    normalized = "model." + path.split(marker, 1)[1]
                    variants = [normalized]
                elif path.startswith("L['self']."):
                    local = path.removeprefix("L['self'].")
                    # AOT module stacks are rooted at the compiled inner model,
                    # while named_parameters() is rooted at the causal-LM
                    # wrapper for Phi/Mamba.  Keep both exact rooted variants;
                    # intersection with T2-reached parameter names decides.
                    variants = [local, "model." + local]
                else:
                    variants = []
                for normalized in variants:
                    if normalized not in modules:
                        modules.append(normalized)
        if modules:
            # Exact leaf modules precede their enclosing attention/layer paths.
            result[task_id] = sorted(modules, key=lambda value: (-value.count("."), value))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-prefix", choices=tuple(CONFIG), required=True)
    parser.add_argument("--sequence-length", choices=(64, 128, 256), type=int, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument(
        "--input-bank", type=Path,
        help="override the frozen input-bank path when the canonical copy moved",
    )
    parser.add_argument(
        "--max-spool-bytes", type=int, default=64_000_000_000,
        help="Maximum estimated complete-vector spool retained by one T3 subprocess.",
    )
    args = parser.parse_args()
    architecture, model, graph_breaks = CONFIG[args.model_prefix]
    cell = f"{args.model_prefix}_seq{args.sequence_length}_r1"
    input_bank = args.input_bank or (
        ROOT / f"results/coverage/{args.model_prefix}_seq{args.sequence_length}_input_bank.json"
    )
    causal = ROOT / "results/coverage/cases/causal" / cell
    marker = causal / "queue_complete.json"
    if not marker.exists():
        raise RuntimeError(f"T2 queue is not complete: {marker}")
    rows = []
    for path in sorted(causal.glob("*.json.gz")):
        payload = load(path)
        for row in payload.get("rows", []):
            if row.get("causal_t2_positive", row.get("causal_t2_t3_positive", False)):
                rows.append(row)
    campaign = load(ROOT / "results/coverage/runtime_releases" / cell / "campaign.json.gz")
    campaign_by_region = {str(row["region_id"]): row for row in campaign["rows"]}
    plan = load(ROOT / "results/coverage/runtime_releases" / cell / "same_dtype_tasks.json.gz")
    bridge = load(ROOT / "results/coverage/runtime_releases" / cell / "candidate_fb_bridge.json.gz")
    aot_modules_by_task = aot_modules_for_tasks(cell, plan, bridge)
    out_dir = ROOT / "results/coverage/cases/carrier" / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    unresolved = []
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.device_index),
        "HF_HOME": "/data1/tzh/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
        "XDG_CACHE_HOME": "/data1/tzh/cache/xdg",
        "TORCHINDUCTOR_CACHE_DIR": f"/data1/tzh/cache/torchinductor/frozen/{cell}",
    })
    pending_cases = []
    for row in sorted(rows, key=lambda value: str(value["task_id"])):
        task_id = str(row["task_id"])
        selected = select_local_carrier(row, campaign_by_region, aot_modules_by_task)
        if selected is None:
            unresolved.append(task_id); continue
        carrier, numel = selected
        token = hashlib.sha256((task_id + "\0" + carrier).encode()).hexdigest()[:16]
        output = out_dir / f"{token}.json.gz"
        if output.exists():
            continue
        pending_cases.append({
            "task_id": task_id, "carrier": carrier, "carrier_numel": numel,
            "spool_dir": f"/data1/tzh/cache/kernel_analyzer_spool/{cell}_t3/{token}",
            "output": str(output),
        })
        print(json.dumps({"event": "T3_QUEUED", "cell": cell, "task_id": task_id,
                          "carrier": carrier, "carrier_numel": numel,
                          "selection": "FB_LOCAL_SEMANTIC_CARRIER_AMONG_ALL_PILOT_REACHED"}), flush=True)
    if args.max_spool_bytes < 1:
        raise ValueError("max spool bytes must be positive")
    batches = pack_spool_batches(pending_cases, args.max_spool_bytes)
    for batch_index, (batch, estimated_bytes) in enumerate(batches):
        case_plan = out_dir / f".batch_plan_{batch_index:04d}.json"
        case_plan.write_text(json.dumps({"cases": batch}, sort_keys=True, separators=(",", ":")) + "\n")
        command = [
            str(PYTHON), str(ROOT / "scripts/run_t3_carrier_batch.py"),
            "--architecture", architecture, "--model", model,
            "--input-bank", str(input_bank),
            "--release-dir", str(ROOT / "results/coverage/runtime_releases" / cell),
            "--case-plan", str(case_plan),
        ]
        if graph_breaks:
            command.append("--allow-graph-breaks")
        print(json.dumps({"event": "T3_BATCH_START", "cell": cell,
                          "batch_index": batch_index, "tasks": len(batch),
                          "estimated_spool_bytes": estimated_bytes,
                          "max_spool_bytes": args.max_spool_bytes}), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        case_plan.unlink(missing_ok=True)
    summary = {"schema": "kernel-analyzer-t3-carrier-queue-summary-v1", "cell": cell,
               "positive_t2_rows": len(rows), "unresolved_carrier_selection": unresolved,
               "selection_rule": "FB_LOCAL_SEMANTIC_CARRIER_AMONG_ALL_PILOT_REACHED"}
    (out_dir / "queue_complete.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"event": "T3_QUEUE_COMPLETE", **summary}), flush=True)


if __name__ == "__main__":
    main()
