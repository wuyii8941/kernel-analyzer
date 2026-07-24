#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.detector import detect_clipping_fork
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import (
    PathConfig,
    _encode_sample,
    attention_backend_context,
    cleanup_memory,
    configure_determinism,
    load_hf_path,
    precision_context,
)
from forkcert.report import markdown_table, write_phase_report


BUGS = [
    {
        "name": "logsumexp_reduction_off_by_one",
        "description": "Execute a shard-boundary off-by-one reduction that omits the maximal denominator term.",
        "injection_kind": "kernel_execution",
    },
    {
        "name": "attention_mask_missing_column",
        "description": "Execute model forward with the prompt column immediately before the response incorrectly disabled.",
        "injection_kind": "kernel_execution",
    },
    {
        "name": "fp16_logits_intermediate",
        "description": "Execute an erroneous naive fp16 exp-sum-log intermediate without the required stable upcast.",
        "injection_kind": "kernel_execution",
    },
]


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["case_id"]), int(row["token_index"])


def path_config(data: dict[str, Any]) -> PathConfig:
    item = data["path_ref"]
    return PathConfig(
        name=item["name"],
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        autocast_dtype=item.get("autocast_dtype"),
        device=item.get("device", "cuda"),
        compile_model=False,
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
        model_training_mode=item.get("model_training_mode", False),
        gradient_checkpointing=item.get("gradient_checkpointing", False),
    )


def execute_bug(tokenizer, model, config: PathConfig, sample: dict[str, Any], bug_name: str) -> list[dict[str, Any]]:
    import torch

    encoded = _encode_sample(tokenizer, sample, config.device)
    input_ids = encoded["input_ids"]
    prompt_len = encoded["prompt_len"]
    attention_mask = torch.ones_like(input_ids)
    if bug_name == "attention_mask_missing_column":
        attention_mask[:, max(prompt_len - 1, 0)] = 0
    with torch.inference_mode(), attention_backend_context(config), precision_context(config):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        if bug_name == "fp16_logits_intermediate":
            logits = logits.to(torch.float16)
        elif config.logits_upcast_fp32:
            logits = logits.float()
        target_ids = input_ids[:, 1:]
        if bug_name == "logsumexp_reduction_off_by_one":
            omitted = logits.clone()
            omitted.scatter_(-1, torch.argmax(omitted, dim=-1, keepdim=True), -torch.inf)
            denominator = torch.logsumexp(omitted, dim=-1)
            target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            target_logps = target_logits - denominator
        elif bug_name == "fp16_logits_intermediate":
            # Disable the surrounding mixed-precision autocast so this bug
            # actually executes exp/sum/log with FP16 intermediates.
            with torch.autocast(device_type="cuda", enabled=False):
                denominator = torch.log(torch.exp(logits.to(torch.float16)).sum(dim=-1)).float()
            denominator = torch.nan_to_num(denominator, nan=65504.0, posinf=65504.0, neginf=-65504.0)
            target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).float()
            target_logps = target_logits - denominator
        elif bug_name == "none":
            target_logps = torch.nn.functional.log_softmax(logits.float(), dim=-1).gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
        else:
            target_logps = torch.nn.functional.log_softmax(logits, dim=-1).gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
    rows = []
    for full_pos in range(prompt_len, input_ids.shape[1]):
        token_id = int(input_ids[0, full_pos].item())
        rows.append(
            {
                "case_id": str(sample["case_id"]),
                "token_index": full_pos - prompt_len,
                "token_id": token_id,
                "logp_bug": float(target_logps[0, full_pos - 1].item()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 executed HF semantic bug injections.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--logprob-jsonl", required=True)
    parser.add_argument("--rollout-jsonl", required=True)
    parser.add_argument("--valid-certificates", required=True, help="Phase 4 legal path-pair certificates used as negative controls.")
    parser.add_argument("--bounds-json", required=True)
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--out-jsonl", default="results/phase5_bug_certificates.jsonl")
    parser.add_argument("--report", default="reports/phase5.md")
    args = parser.parse_args()

    bounds = json.loads(Path(args.bounds_json).read_text(encoding="utf-8"))
    if bounds.get("certificate_kind") != "analytic_legal" or not str(bounds.get("decision", "")).startswith("GO"):
        raise SystemExit("Phase 5 requires an analytic_legal Phase 2 GO bound.")
    delta_bound = float(bounds["logprob_bound_worst"])
    cfg_data = load_config(args.config)
    config = path_config(cfg_data)
    configure_determinism(seed=int(cfg_data.get("seed", 0)))
    samples = read_jsonl(args.samples)
    base = {key(row): row for row in read_jsonl(args.logprob_jsonl)}
    rollout = {key(row): row for row in read_jsonl(args.rollout_jsonl)}
    tokenizer, model = load_hf_path(config)
    certs = []
    zero_advantage_skipped = 0
    try:
        for bug in BUGS:
            for sample in samples:
                for bug_row in execute_bug(tokenizer, model, config, sample, bug["name"]):
                    row_key = key(bug_row)
                    ref = base.get(row_key)
                    state = rollout.get(row_key)
                    if ref is None or state is None:
                        continue
                    if int(ref["token_id"]) != int(bug_row["token_id"]) or int(state["token_id"]) != int(bug_row["token_id"]):
                        raise ValueError(f"token alignment mismatch for {row_key}")
                    sign = int(state.get("advantage_sign", 0))
                    if sign == 0:
                        zero_advantage_skipped += 1
                        continue
                    cert = detect_clipping_fork(
                        case_id=f"{bug['name']}:{ref['case_id']}",
                        token_index=int(ref["token_index"]),
                        token_id=int(ref["token_id"]),
                        token_text=ref.get("token_text"),
                        path_ref=ref.get("path_ref", config.name),
                        path_alt=f"{config.name}+{bug['name']}",
                        logp_ref=float(ref["logp_ref"]),
                        logp_alt=float(bug_row["logp_bug"]),
                        old_logp=float(state["old_logp"]),
                        advantage=None,
                        advantage_sign_value=sign,
                        eps=args.eps,
                        delta_self_ref=ref.get("delta_self_ref"),
                        delta_self_alt=ref.get("delta_self_alt"),
                        delta_bound_legal=delta_bound,
                        metadata={
                            "phase": "phase5_hf_bug_injection",
                            "bug": bug,
                            "phase5_expected_bug": True,
                            "rollout_alignment": {"token_id_match": True},
                        },
                    )
                    certs.append(cert.to_json_dict())
    finally:
        del model
        del tokenizer
        cleanup_memory()

    valid_controls = []
    for row in read_jsonl(args.valid_certificates):
        item = dict(row)
        metadata = dict(item.get("metadata") or {})
        metadata["phase5_expected_bug"] = False
        metadata["phase5_control_source"] = args.valid_certificates
        item["metadata"] = metadata
        valid_controls.append(item)
    output_rows = valid_controls + certs
    write_jsonl(args.out_jsonl, output_rows)
    by_bug = []
    for bug in BUGS:
        rows = [row for row in certs if ((row.get("metadata") or {}).get("bug") or {}).get("name") == bug["name"]]
        classified = sum(1 for row in rows if row.get("region") == "bug")
        by_bug.append(
            {
                "bug": bug["name"],
                "certificates": len(rows),
                "classified_bug": classified,
                "recall": classified / len(rows) if rows else 0.0,
            }
        )
    true_positive = sum(1 for row in output_rows if (row.get("metadata") or {}).get("phase5_expected_bug") is True and row.get("region") == "bug")
    false_negative = sum(1 for row in output_rows if (row.get("metadata") or {}).get("phase5_expected_bug") is True and row.get("region") != "bug")
    false_positive = sum(1 for row in output_rows if (row.get("metadata") or {}).get("phase5_expected_bug") is False and row.get("region") == "bug")
    true_negative = sum(1 for row in output_rows if (row.get("metadata") or {}).get("phase5_expected_bug") is False and row.get("region") != "bug")
    confusion = {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "bug_recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "valid_pair_false_positive_rate": false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0,
        "zero_advantage_injected_rows_skipped": zero_advantage_skipped,
    }
    write_phase_report(
        args.report,
        title="Phase 5 Executed Bug Injection",
        confound_checklist={
            "kernel_execution_injections": True,
            "analytic_legal_bound": True,
            "exact_token_alignment": True,
            "posthoc_logprob_shift": False,
        },
        delta_self_summary="Uses Phase 1 self-consistency fields; injected paths execute altered model operations.",
        summary="Three executed semantic bug paths were evaluated against the legal bound.",
        sections={
            "Confusion Matrix": markdown_table([confusion], list(confusion.keys())),
            "By Injected Bug": markdown_table(by_bug, list(by_bug[0].keys()) if by_bug else []),
        },
    )
    print(json.dumps({"n_certificates": len(output_rows), "confusion": confusion, "by_bug": by_bug}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
