"""Freeze the exact graph-break wrappers warmed by a replay process."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def bind_runtime_schedule(
    *, modules: Iterable[Any], work_dir: Path, manifest: Path,
    inventory: Path, campaign: Path | None, architecture: str,
    state: dict[str, Any], input_digests: dict[str, str],
    values: tuple[Any, ...], modality: str, gradient_checkpointing: bool,
    allow_graph_breaks: bool,
) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    wrappers = []
    for module in modules:
        source = Path(module.__file__).resolve()
        match = re.search(
            r"# AOT ID: \['\d+_(forward|backward|inference)'\]",
            source.read_text(errors="ignore")[:512],
        )
        if match is not None:
            wrappers.append((source, match.group(1)))
    phase_counts = {"forward": 0, "backward": 0}
    rows = []
    for ordinal, (source, aot_kind) in enumerate(wrappers):
        phase = "forward" if aot_kind == "inference" else aot_kind
        segment = phase_counts[phase]
        phase_counts[phase] += 1
        target = work_dir / (
            f"torchinductor/model__{ordinal}_{phase}_segment{segment}_executed/output_code.py"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({
            "phase": phase.upper(), "segment": segment,
            "execution_ordinal": ordinal, "aot_kind": aot_kind,
            "executed_source": str(source), "captured_source": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
    if not phase_counts["forward"] or not phase_counts["backward"]:
        raise RuntimeError("runtime binding lacks complete F+B wrappers")
    input_manifest = {
        "state": 0,
        "state_id": state.get("state_id", state.get("sequence_id", "runtime-warm")),
        "sequence_length": int(values[0].numel()),
        "token_ids_sha256": input_digests["token_ids_sha256"],
        **({"image_sha256": input_digests["image_sha256"]} if "image_sha256" in input_digests else {}),
    }
    payload = {
        "schema": "kernel-analyzer-executed-inductor-schedule-v1",
        "status": "COMPLETE_EXACT_EXECUTED_FORWARD_BACKWARD_SOURCE_CAPTURE",
        "architecture": architecture, "backend": "inductor",
        "allow_graph_breaks": allow_graph_breaks, "input": input_manifest,
        "modality": modality,
        "gradient_checkpointing": gradient_checkpointing,
        "repeat_stable": "DEFERRED_TO_REPLAY_SCREEN",
        "runs": [], "modules": rows,
        "phase_module_counts": {key.upper(): value for key, value in phase_counts.items()},
    }
    payload["result_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = f"{ROOT / 'archive/round1_code/src'}:{ROOT}:{env.get('PYTHONPATH', '')}"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/build_current_qwen_generated_inventory.py"),
        "--trace-dir", str(work_dir), "--capture", str(manifest),
        "--output", str(inventory), "--allow-partial-dataflow",
    ], check=True, cwd=ROOT, env=env)
    if campaign is not None:
        subprocess.run([
            sys.executable, str(ROOT / "scripts/build_generated_fp32_campaign.py"),
            "--inventory", str(inventory), "--output", str(campaign),
        ], check=True, cwd=ROOT, env=env)
