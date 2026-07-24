#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.report import markdown_table, write_phase_report
from forkcert.stats import percentile


def summarize_pair(name: str, role: str, path: Path) -> dict:
    rows = read_jsonl(path)
    deltas = [float(row["logprob_delta"]) for row in rows]
    self_ref = [float(row.get("delta_self_ref", 0.0)) for row in rows]
    self_alt = [float(row.get("delta_self_alt", 0.0)) for row in rows]
    samples = len({str(row["case_id"]) for row in rows})
    cross_p50 = percentile(deltas, 50) if deltas else 0.0
    ref_p99 = percentile(self_ref, 99) if self_ref else 0.0
    alt_p99 = percentile(self_alt, 99) if self_alt else 0.0
    scale_gate = 100 <= samples <= 500 and len(rows) >= 50_000
    self_gate = cross_p50 > 0 and ref_p99 < 0.1 * cross_p50 and alt_p99 < 0.1 * cross_p50
    metadata = (rows[0].get("metadata") or {}) if rows else {}
    fingerprint_ref = metadata.get("model_artifact_fingerprint_ref") or {}
    fingerprint_alt = metadata.get("model_artifact_fingerprint_alt") or {}
    weights_gate = (
        fingerprint_ref.get("verified_local_files") is True
        and fingerprint_alt.get("verified_local_files") is True
        and bool(fingerprint_ref.get("aggregate_sha256"))
        and fingerprint_ref.get("aggregate_sha256") == fingerprint_alt.get("aggregate_sha256")
    )
    env = metadata.get("env") or {}
    torch_env = env.get("torch") or {}
    deterministic_env = env.get("deterministic_env") or {}
    determinism_gate = (
        torch_env.get("deterministic_algorithms") is True
        and torch_env.get("deterministic_warn_only") is True
        and torch_env.get("cudnn_benchmark") is False
        and deterministic_env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and deterministic_env.get("PYTHONHASHSEED") == "0"
    )
    return {
        "name": name,
        "role": role,
        "path": str(path),
        "samples": samples,
        "tokens": len(rows),
        "delta_p50": cross_p50,
        "delta_p99": percentile(deltas, 99) if deltas else 0.0,
        "self_ref_p99": ref_p99,
        "self_alt_p99": alt_p99,
        "scale_gate": scale_gate,
        "self_gate": self_gate,
        "weights_gate": weights_gate,
        "determinism_gate": determinism_gate,
        "pair_gate": scale_gate and self_gate and weights_gate and determinism_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the Phase 1 path-pair manifest.")
    parser.add_argument("--debug", required=True)
    parser.add_argument("--claim-compile", required=True)
    parser.add_argument("--claim-sdpa", required=True)
    parser.add_argument("--claim-vllm", default=None)
    parser.add_argument("--env-audit", required=True)
    parser.add_argument("--out", default="results/phase1_pair_manifest.json")
    parser.add_argument("--report", default="reports/phase1_pairs.md")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    args = parser.parse_args()

    pairs = [
        summarize_pair("debug_fp32_bf16", "debug_only", Path(args.debug)),
        summarize_pair("claim_eager_compile_same_dtype", "claim", Path(args.claim_compile)),
        summarize_pair("claim_sdpa_backend_same_dtype", "claim", Path(args.claim_sdpa)),
    ]
    env = json.loads(Path(args.env_audit).read_text(encoding="utf-8"))
    vllm_version = (env.get("packages") or {}).get("vllm")
    if args.claim_vllm:
        vllm = summarize_pair("claim_hf_vllm_bf16", "claim_optional", Path(args.claim_vllm))
        vllm.update({"available": bool(vllm_version), "version": vllm_version, "status": "measured"})
        vllm_gate = bool(vllm_version) and vllm["pair_gate"]
    else:
        vllm = {
            "name": "claim_hf_vllm_bf16",
            "role": "claim_optional",
            "available": bool(vllm_version),
            "version": vllm_version,
            "status": "required_but_missing" if vllm_version else "skipped_package_unavailable",
        }
        vllm_gate = not bool(vllm_version)
    required_gate = all(pair["pair_gate"] for pair in pairs)
    payload = {
        "required_pairs_gate": required_gate,
        "optional_vllm_gate": vllm_gate,
        "debug_results_excluded_from_claim": True,
        "pairs": pairs,
        "optional_vllm": vllm,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_phase_report(
        args.report,
        title="Phase 1 Path Pair Manifest",
        confound_checklist={
            "debug_pair_excluded_from_claim": True,
            "compile_claim_pair_present": pairs[1]["pair_gate"],
            "sdpa_flash_claim_pair_present": pairs[2]["pair_gate"],
            "vllm_measured_or_unavailable": vllm_gate,
        },
        delta_self_summary="Every measured pair independently enforces self p99 < 0.1 * cross p50.",
        summary="Required debug and claim path pairs were validated independently.",
        sections={
            "Measured Pairs": markdown_table(pairs, list(pairs[0].keys())),
            "Optional vLLM": markdown_table([vllm], list(vllm.keys())),
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_incomplete and not (required_gate and vllm_gate):
        print("Phase 1 path-pair manifest gate failed.", file=sys.stderr)
        raise SystemExit(21)


if __name__ == "__main__":
    main()
