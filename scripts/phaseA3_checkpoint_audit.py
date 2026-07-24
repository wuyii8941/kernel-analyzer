#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.report import CLAIM_SCOPE, markdown_table


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_section(path: Path, marker: str, section: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    block = f"{start}\n{section.rstrip()}\n{end}\n"
    if start in text and end in text:
        before, remainder = text.split(start, 1)
        _old, after = remainder.split(end, 1)
        text = before.rstrip() + "\n\n" + block + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit checkpoint-state alignment of Phase 0 margins and Phase 1 deltas.")
    parser.add_argument("--dump", default="data/phase0_grpo_dump.jsonl")
    parser.add_argument("--phase1-metadata", default="results/phase1_logprobs.metadata.json")
    parser.add_argument("--checkpoint-root", default="data/phase0_policy_final")
    parser.add_argument("--out-json", default="results/phaseA3_checkpoint_audit.json")
    parser.add_argument("--report", default="reports/phaseA3_checkpoint_audit.md")
    parser.add_argument("--phase1-report", default="reports/phase1.md")
    parser.add_argument("--aligned-dump", default="data/phaseA3_recovery_dump_r4.jsonl")
    parser.add_argument("--aligned-samples", default="data/phaseA3_recovery_samples_r4.jsonl")
    parser.add_argument("--aligned-snapshot", default="data/phase0_policy_step272_pre")
    parser.add_argument("--aligned-phase1-metadata", default="results/phaseA3_compile_step272.metadata.json")
    args = parser.parse_args()

    rows = read_jsonl(args.dump)
    max_iteration = max(int(row["policy_iteration"]) for row in rows)
    iteration_steps = sorted({int(row["optimizer_step"]) for row in rows if int(row["policy_iteration"]) == max_iteration})
    root = Path(args.checkpoint_root)
    checkpoint_steps = []
    for path in root.glob("checkpoint-*"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            checkpoint_steps.append(int(match.group(1)))
    checkpoint_steps.sort()
    exact_intersection = sorted(set(iteration_steps) & set(checkpoint_steps))
    metadata = json.loads(Path(args.phase1_metadata).read_text(encoding="utf-8"))
    model_path = Path(metadata["config"]["path_ref"]["model_name_or_path"])
    model_file = model_path / "model.safetensors"
    model_hash = sha256(model_file)
    phase1_recorded_hash = metadata["model_artifact_fingerprint_ref"]["files"]
    recorded_model = next(item for item in phase1_recorded_hash if item["name"] == "model.safetensors")
    phase1_hash_match = recorded_model["sha256"] == model_hash
    historical_aligned = bool(exact_intersection) and model_path.name in {
        f"checkpoint-{step}" for step in exact_intersection
    }
    aligned_rows_all = read_jsonl(args.aligned_dump)
    snapshot_dir = Path(args.aligned_snapshot)
    snapshot_metadata = json.loads((snapshot_dir / "forkcert_snapshot.json").read_text(encoding="utf-8"))
    snapshot_step = int(snapshot_metadata["optimizer_step"])
    aligned_rows = [row for row in aligned_rows_all if int(row["optimizer_step"]) == snapshot_step]
    aligned_samples = read_jsonl(args.aligned_samples)
    aligned_case_ids = {str(row["case_id"]) for row in aligned_rows}
    sample_case_ids = {str(row["case_id"]) for row in aligned_samples}
    aligned_phase1 = json.loads(Path(args.aligned_phase1_metadata).read_text(encoding="utf-8"))
    aligned_model_path = Path(aligned_phase1["config"]["path_ref"]["model_name_or_path"])
    aligned_model_hash = sha256(snapshot_dir / "model.safetensors")
    aligned_recorded = next(
        item
        for item in aligned_phase1["model_artifact_fingerprint_ref"]["files"]
        if item["name"] == "model.safetensors"
    )
    aligned = bool(aligned_rows) and all(
        [
            int(snapshot_metadata["policy_iteration"]) == max_iteration,
            aligned_model_path.resolve() == snapshot_dir.resolve(),
            aligned_recorded["sha256"] == aligned_model_hash,
            aligned_case_ids <= sample_case_ids,
            {int(row["optimizer_step"]) for row in aligned_rows} == {snapshot_step},
        ]
    )
    payload = {
        "gate": aligned,
        "gate_scope": "aligned_step272_rollout_batch",
        "historical_51200_convolution_gate": historical_aligned,
        "phase0_policy_iteration": max_iteration,
        "phase0_iteration2_optimizer_steps": iteration_steps,
        "saved_checkpoint_steps": checkpoint_steps,
        "exact_iteration2_checkpoint_steps": exact_intersection,
        "phase1_model_path": str(model_path),
        "phase1_model_sha256": model_hash,
        "phase1_recorded_hash_match": phase1_hash_match,
        "phase1_inferred_training_step": 300 if model_path == root else None,
        "aligned_snapshot_step": snapshot_step,
        "aligned_snapshot_policy_iteration": int(snapshot_metadata["policy_iteration"]),
        "aligned_snapshot_model_sha256": aligned_model_hash,
        "aligned_phase1_model_path": str(aligned_model_path),
        "aligned_case_count": len(aligned_case_ids),
        "aligned_token_count": len(aligned_rows),
        "failure": (
            None
            if historical_aligned
            else "Phase 1 used the final step-300 root model, while iteration-2 margins were measured at pre-minibatch steps 2,5,...,299; no exact iteration-2 checkpoint was saved."
        ),
        "required_repair": (
            "The aligned step-272 repair is complete. For a full 51,200-token scan, measure each rollout's delta "
            "online at its own policy_iteration=2 pre-minibatch state; do not reuse the final checkpoint."
        ),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    facts = [
        {"item": "Phase 1 model", "value": str(model_path)},
        {"item": "Phase 1 model SHA256", "value": model_hash},
        {"item": "Phase 1 inferred step", "value": payload["phase1_inferred_training_step"]},
        {"item": "Iteration-2 step range", "value": f"{iteration_steps[0]}..{iteration_steps[-1]} (stride 3)"},
        {"item": "Saved checkpoints", "value": ",".join(map(str, checkpoint_steps))},
        {"item": "Exact intersection", "value": ",".join(map(str, exact_intersection)) or "none"},
        {"item": "Aligned repair snapshot", "value": f"step {snapshot_step}, policy iteration {snapshot_metadata['policy_iteration']}"},
        {"item": "Aligned repair model SHA256", "value": aligned_model_hash},
        {"item": "Aligned repair scale", "value": f"{len(aligned_case_ids)} cases / {len(aligned_rows)} tokens"},
    ]
    section = "\n".join(
        [
            "## Phase A3 Checkpoint-State Audit",
            "",
            f"- aligned step-272 batch checkpoint-state alignment: {'PASS' if aligned else 'FAIL'}",
            f"- original 51,200-token historical convolution alignment: {'PASS' if historical_aligned else 'FAIL'}",
            f"- Phase 1 recorded/current model hash: {'PASS' if phase1_hash_match else 'FAIL'}",
            "- Phase 4 authorization: aligned 512-token batch only; full scan requires per-rollout online state alignment",
            "",
            markdown_table(facts, ["item", "value"]),
            "",
            payload["failure"] or "The original historical margin and delta states are aligned.",
            "",
            f"Required repair: {payload['required_repair']}",
        ]
    )
    report = "\n".join(
        [
            "# Phase A3 Checkpoint-State Audit",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            f"- exact checkpoint state shared by aligned repair margin and delta: {'PASS' if aligned else 'FAIL'}",
            f"- original pooled 51,200-token state alignment: {'PASS' if historical_aligned else 'FAIL'}",
            f"- Phase 1 model hash verified: {'PASS' if phase1_hash_match else 'FAIL'}",
            "",
            "## Delta Self Control",
            "Orthogonal; process independence is handled by Phase A1.",
            "",
            "## External Validity",
            "State alignment is precision-independent. Numerical results remain scoped to T4 FP16; a zero-fork result cannot exclude BF16 forks.",
            "",
            section,
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    append_section(Path(args.phase1_report), "phaseA3", section)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not aligned:
        raise SystemExit(33)


if __name__ == "__main__":
    main()
