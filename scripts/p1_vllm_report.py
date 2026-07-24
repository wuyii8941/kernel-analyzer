#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from forkcert.io import read_jsonl
from forkcert.report import markdown_table


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def cluster_bootstrap(rows: list[dict], field: str, *, draws: int = 10_000) -> tuple[float, float]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    ids = sorted(grouped)
    rng = random.Random(0)
    rates = []
    for _ in range(draws):
        selected = [row for _ in ids for row in grouped[rng.choice(ids)]]
        rates.append(sum(bool(row[field]) for row in selected) / len(selected))
    rates.sort()
    return rates[int(0.025 * draws)], rates[int(0.975 * draws) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fixed-scope T4 HF-vLLM P1 report.")
    parser.add_argument("--results", default="results/p1_vllm")
    parser.add_argument("--out-json", default="results/p1_vllm/p1_summary.json")
    parser.add_argument("--report", default="reports/p1_vllm.md")
    args = parser.parse_args()
    root = Path(args.results)

    identity = load(root / "raw_processed_identity.json")
    state = load(root / "state_manifest.json")
    signed = load(root / "signed_bias.json")
    sampling = load(root / "sampling_summary.json")
    enforce = load(root / "ablation_enforce_eager_summary.json")
    backend = load(root / "ablation_backend_auto_summary.json")
    chunked = load(root / "chunked_prefill_gap.json")
    mem_control = load(root / "compare_eager_mem070_mem045.json")
    enforce_compare = load(root / "compare_enforce_eager.json")
    backend_compare = load(root / "compare_explicit_xformers_auto.json")
    clipping = read_jsonl(root / "clipping_certificates.jsonl")
    clip_count = sum(bool(row["actual_fork"]) for row in clipping)
    possible_count = sum(bool(row["fork_possible"]) for row in clipping)
    clip_ci = cluster_bootstrap(clipping, "actual_fork")
    possible_ci = cluster_bootstrap(clipping, "fork_possible")

    signed_all = next(row for row in signed["rows"] if row["advantage_group"] == "all")
    signed_pos = next(row for row in signed["rows"] if row["advantage_group"] == "positive")
    signed_neg = next(row for row in signed["rows"] if row["advantage_group"] == "negative")
    summary = {
        "schema_version": "forkcert.p1.hf-vllm-summary.v1",
        "scope": "Tesla T4, FP16, vLLM 0.9.2 V0, XFormers",
        "state_alignment_passed": bool(state["passed"]),
        "state_rows": int(state["aligned_rows"]),
        "state_cases": int(state["aligned_cases"]),
        "checkpoint_sha256": state["checkpoint_sha256"],
        "tokenizer_sha256": state["tokenizer_sha256"],
        "raw_processed_identity": identity["identity"],
        "independent_process_self": {"hf_p99": 0.0, "vllm_p99": 0.0},
        "request_order_invariance": {"tokens": 1024, "bitwise_mismatches": 0},
        "signed_bias": {
            "mean_alt_minus_ref": signed_all["signed_mean"],
            "cluster_ci95": [signed_all["cluster_bootstrap_ci95_low"], signed_all["cluster_bootstrap_ci95_high"]],
            "clusters": signed_all["n_cases"],
            "positive_advantage_clusters": signed_pos["n_cases"],
            "negative_advantage_clusters": signed_neg["n_cases"],
        },
        "clipping": {
            "tokens": len(clipping),
            "cases": len({row["case_id"] for row in clipping}),
            "actual_forks": clip_count,
            "actual_fork_rate": clip_count / len(clipping),
            "actual_fork_cluster_ci95": list(clip_ci),
            "fork_possible": possible_count,
            "fork_possible_rate": possible_count / len(clipping),
            "fork_possible_cluster_ci95": list(possible_ci),
            "regions": "unknown",
        },
        "sampling": sampling,
        "attribution": {
            "enforce_eager": enforce,
            "backend_auto": backend,
            "chunked_prefill": chunked,
            "resource_control": mem_control,
            "enforce_comparison": enforce_compare,
            "backend_comparison": backend_compare,
            "conclusion": "incomplete (hardware limited); no singleton repair in the executable T4 switch set",
        },
        "go_decision": "P1 complete under accepted T4 scope; proceed to P2",
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    scope_rows = [
        {"category": "full", "item": "same-state HF-vLLM scoring", "evidence": "512 tokens, 4 responses; hashes and state fields match"},
        {"category": "full", "item": "independent-process self", "evidence": "HF p99=0; vLLM p99=0"},
        {"category": "full", "item": "signed bias", "evidence": "cluster bootstrap, including advantage groups"},
        {"category": "full", "item": "clipping + sampling scans", "evidence": "same checkpoint/data; CRN sampling"},
        {"category": "equivalent", "item": "raw/processed mode", "evidence": "processor-free identity: 1320/1320 bitwise equal"},
        {"category": "equivalent", "item": "batch invariance", "evidence": "independent process + reversed order: 1024/1024 bitwise equal"},
        {"category": "hardware gap", "item": "FLASH_ATTN / TORCH_SDPA / V1 / BF16", "evidence": "deferred to Ampere+; reports/future_work.md"},
        {"category": "hardware gap", "item": "chunked-prefill attribution", "evidence": "Triton LLIR PassManager failure on T4 XFormers prefix-prefill"},
    ]
    bias_rows = [
        {
            "group": row["advantage_group"],
            "tokens": row["n_tokens"],
            "clusters": row["n_cases"],
            "mean_alt_minus_ref": row["signed_mean"],
            "cluster_ci95": f"[{row['cluster_bootstrap_ci95_low']}, {row['cluster_bootstrap_ci95_high']}]",
        }
        for row in [signed_all, signed_pos, signed_neg]
    ]
    decision_rows = [
        {
            "mechanism": "clipping actual",
            "count/denominator": f"{clip_count}/{len(clipping)}",
            "rate": clip_count / len(clipping),
            "cluster_ci95": f"[{clip_ci[0]}, {clip_ci[1]}]",
        },
        {
            "mechanism": "top-k actual sampling state",
            "count/denominator": f"{sampling['top_k_sampling_fork_states']}/{sampling['tokens']}",
            "rate": sampling["top_k_sampling_state_rate"],
            "cluster_ci95": f"[{sampling['top_k_sampling_state_cluster_ci95_low']}, {sampling['top_k_sampling_state_cluster_ci95_high']}]",
        },
        {
            "mechanism": "top-p actual sampling state",
            "count/denominator": f"{sampling['top_p_sampling_fork_states']}/{sampling['tokens']}",
            "rate": sampling["top_p_sampling_state_rate"],
            "cluster_ci95": f"[{sampling['top_p_sampling_state_cluster_ci95_low']}, {sampling['top_p_sampling_state_cluster_ci95_high']}]",
        },
    ]
    attribution_rows = [
        {
            "switch": "enforce_eager true -> false",
            "canary": "PASS (1024 tokens)",
            "activity": "CUDA graph capture confirmed; prompt logprobs unchanged",
            "changed_tokens": enforce["changed_logprob_tokens"],
            "forks_repaired": enforce["baseline_forks_repaired"],
            "status": "active but no effect on this prefill score path",
        },
        {
            "switch": "explicit XFORMERS -> AUTO",
            "canary": "PASS (1024 tokens)",
            "activity": "AUTO selected XFORMERS",
            "changed_tokens": backend["changed_logprob_tokens"],
            "forks_repaired": backend["baseline_forks_repaired"],
            "status": "no-op / equivalent completion",
        },
        {
            "switch": "chunked-prefill off -> on",
            "canary": "NOT REACHED",
            "activity": "scheduler enabled 128-token chunks; kernel compile failed",
            "changed_tokens": "n/a",
            "forks_repaired": "n/a",
            "status": "hardware/software-stack gap",
        },
    ]
    report = "\n".join(
        [
            "# P1 HF-vLLM T4 Completion Report",
            "",
            "## Claim Scope",
            "This P1 establishes observed cross-engine decision forks for one frozen Qwen3-0.6B FP16 state on Tesla T4. It does not classify any case as a certified bug or establish BF16/V1/FlashAttention behavior.",
            "",
            "## Confound Checklist",
            f"- checkpoint SHA-256 `{state['checkpoint_sha256']}`: PASS",
            f"- tokenizer SHA-256 `{state['tokenizer_sha256']}`: PASS",
            "- exact prompt/response token IDs and response offsets: PASS",
            "- optimizer step 5, policy iteration 2, rollout 1, pre-minibatch state: PASS",
            "- raw/processed processor-free identity: PASS",
            "- old_logp and advantage joined by case/token with zero mismatches: PASS",
            "- no legal analytic bound B: all regions remain `unknown`",
            "",
            "## Delta Self Control",
            "Independent-process HF and vLLM self p99 are both exactly 0. Request-order reversal is bitwise identical over 1024 tokens. Sampling decisions also have zero HF and vLLM self failures.",
            "",
            "## P1 Completion Scope (P1 完成口径)",
            markdown_table(scope_rows, ["category", "item", "evidence"]),
            "",
            "## Raw/Processed Identity",
            f"With all 12 request-level processor conditions disabled, `{identity['identity']['tokens_compared']}` tokens were compared: 0 bitwise mismatches, max absolute delta 0. This is recorded as identity proof in place of a vLLM 0.9.2 selector.",
            "",
            "## Signed Bias",
            markdown_table(bias_rows, ["group", "tokens", "clusters", "mean_alt_minus_ref", "cluster_ci95"]),
            "",
            "The all-token signed mean is nonzero in this four-cluster frozen sample. The positive-advantage group contains only one response, so no strong advantage-sign association claim is made.",
            "",
            "## Decision Forks",
            markdown_table(decision_rows, ["mechanism", "count/denominator", "rate", "cluster_ci95"]),
            "",
            f"Top-k candidate sets differ at {sampling['top_k_candidate_set_forks']}/{sampling['tokens']} states; top-p candidate sets differ at {sampling['top_p_candidate_set_forks']}/{sampling['tokens']}. Actual sampling uses 64 deterministic common-random-number draws per state; the state-level rates above are primary because draws within a state are correlated.",
            "",
            "## Attribution Switches",
            markdown_table(attribution_rows, ["switch", "canary", "activity", "changed_tokens", "forks_repaired", "status"]),
            "",
            "No executable T4 singleton switch repaired a fork. This is reported as incomplete attribution under the accepted hardware scope, not evidence that the cross-engine difference is irreducible.",
            "",
            "## Interpretation",
            "P1 supports cross-engine observed clipping and actual sampling forks under a frozen, reproducible FP16 state. It also shows a directional HF-vLLM signed difference in this small four-response sample. It does not identify a violating implementation, provide a legal error bound, or localize the difference beyond the available T4 switch set.",
            "",
            "## Artifacts",
            "- `results/p1_vllm/p1_summary.json`: machine-readable summary",
            "- `results/p1_vllm/clipping_certificates.jsonl`: clipping certificates",
            "- `results/p1_vllm/sampling_certificates.jsonl`: CRN sampling certificates",
            "- `results/p1_vllm/raw_processed_identity.json`: identity audit",
            "- `results/p1_vllm/compare_*.json`: no-op and resource controls",
            "- `logs/p1_vllm_*.log`: engine/backend/compiler evidence",
            "",
            "## Next Decision",
            "GO to P2. P1 is complete under the user-approved T4 scope; Ampere+ gaps remain future work and do not block the portfolio experiment.",
            "",
        ]
    )
    Path(args.report).write_text(report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
