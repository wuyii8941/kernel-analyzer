#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from forkcert.config import load_config


CANONICAL_OUTPUTS = [
    ("Conda explicit environment", Path("results/conda_environment.explicit.txt")),
    ("Conda pip freeze", Path("results/conda_environment.pip_freeze.txt")),
    ("Phase 0 TRL GRPO dump", Path("data/phase0_grpo_dump.jsonl")),
    ("Phase 0 GRPO samples", Path("data/phase0_grpo_samples.jsonl")),
    ("Phase 0 final rollout", Path("data/phase0_final_rollout.jsonl")),
    ("Phase 0 final checkpoint", Path("data/phase0_policy_final/config.json")),
    ("Phase 0 final checkpoint configs", Path("results/configs/phase0_final_configs.json")),
    ("Phase 0 report", Path("reports/phase0.md")),
    ("Phase 0 margin histogram", Path("reports/phase0_margin_hist.svg")),
    ("Phase 1 logprobs", Path("results/phase1_logprobs.jsonl")),
    ("Phase 1 debug logprobs", Path("results/phase1_debug_fp32_bf16.jsonl")),
    ("Phase 1 SDPA logprobs", Path("results/phase1_sdpa_logprobs.jsonl")),
    ("Phase 1 pair manifest", Path("results/phase1_pair_manifest.json")),
    ("Phase 1 report", Path("reports/phase1.md")),
    ("Phase 1.5 measurements", Path("results/phase15_measurements.jsonl")),
    ("Phase 1.5 propagation curves", Path("reports/phase15_propagation.svg")),
    ("Phase 2 bounds", Path("results/phase2_bounds.json")),
    ("Phase 2 analytic source draft", Path("results/phase2_sources.analytic_draft.json")),
    ("Phase 3 controlled certificates", Path("results/phase3_controlled_certificates.jsonl")),
    ("Phase 3 calibration model", Path("results/phase3_calibration.json")),
    ("Phase 4 certificates", Path("results/phase4_certificates.jsonl")),
    ("Phase 5 bug certificates", Path("results/phase5_bug_certificates.jsonl")),
    ("Phase 6 autograd certificates", Path("results/phase6_grad_certificates.jsonl")),
    ("Phase 6 twin trajectory", Path("results/phase6_twin_trajectory.jsonl")),
    ("Phase 6 twin summary", Path("results/phase6_twin_summary.json")),
    ("Phase 6 proxy smoke certificates", Path("results/phase6_grad_proxy_certificates.jsonl")),
    ("Implementation/evidence audit", Path("reports/completion_audit.md")),
    ("Final audit", Path("reports/audit.md")),
]


def file_status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    size = path.stat().st_size
    if size == 0:
        return "EMPTY"
    return f"PRESENT ({size} bytes)"


def config_model_ids(paths: list[str]) -> list[str]:
    model_ids = []
    for path in paths:
        cfg = load_config(path)
        for key in ["model", "policy", "path_ref", "path_alt"]:
            item = cfg.get(key)
            if isinstance(item, dict) and item.get("model_name_or_path"):
                model_ids.append(str(item["model_name_or_path"]))
    return sorted(set(model_ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a concise current-state report for ForkCert execution.")
    parser.add_argument("--out", default="reports/status.md")
    parser.add_argument(
        "--config",
        action="append",
        default=[
            "configs/phase0_grpo.example.yaml",
            "configs/hf_pair.example.yaml",
            "configs/hf_debug_fp32_bf16.example.yaml",
            "configs/hf_sdpa_math_flash.example.yaml",
            "configs/hf_logsoftmax_upcast.example.yaml",
            "configs/hf_rmsnorm_reference.example.yaml",
            "configs/hf_materialization.example.yaml",
            "configs/hf_matmul_reduction.example.yaml",
        ],
    )
    args = parser.parse_args()

    models = config_model_ids(args.config)
    versions = {}
    for package in ["torch", "transformers", "trl", "datasets", "accelerate"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    rows = [(name, str(path), file_status(path)) for name, path in CANONICAL_OUTPUTS]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ForkCert Current Status",
        "",
        "## Model Configs",
        "",
        "- Default configured model ids: " + ", ".join(models),
        "- Qwen2.5 is not used by the default configs.",
        "- Installed package versions: " + ", ".join(f"{key}={value}" for key, value in versions.items()),
        "- Canonical Python: /data1/tzh/conda-envs/forkcert/bin/python",
        f"- Qwen3 local cache: {file_status(Path('results/model_prefetch.qwen3_0_6b.json'))}",
        "",
        "## Canonical Experiment Outputs",
        "",
        "| Artifact | Path | Status |",
        "| --- | --- | --- |",
    ]
    for name, path, status in rows:
        lines.append(f"| {name} | `{path}` | {status} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Files ending in `.example.*` are smoke-test fixtures, not real experimental evidence.",
            "- A real run is not complete until canonical outputs such as `results/phase1_logprobs.jsonl`, `results/phase4_certificates.jsonl`, and `reports/audit.md` exist and pass the audit gates.",
            "- Keep generated data, checkpoints, logs, and caches under `/data1/tzh/forkcert`.",
            "- The generated Phase 2 source file is deliberately marked `empirical_heuristic`; the canonical runner stops at exit 22 until an `analytic_legal` source with verified assumptions is supplied.",
            "- The user GPU shell reports CUDA available with 14 devices. The Codex execution channel still times out during CUDA initialization, so GPU phases are launched from the user shell and audited through shared logs/artifacts.",
            "- Qwen3 is prefetched in the standard `$HF_HOME/hub` layout and has passed an offline Transformers config/tokenizer load check.",
            "",
            "## Next Command",
            "",
            "```bash",
            "cd /data1/tzh/forkcert",
            "./run_phase1_gpu_resume.sh",
            "```",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "models": models, "versions": versions, "canonical_outputs": rows}, indent=2))


if __name__ == "__main__":
    main()
