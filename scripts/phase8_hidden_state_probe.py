#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import attention_backend_context, cleanup_memory, configure_determinism, load_hf_path, precision_context
from scripts.phase6_twin_training import path_config
from scripts.phase8_case_attribution import make_batch, target_batch


def run(cfg, samples, cert):
    import torch
    tokenizer, model = load_hf_path(cfg)
    ids, mask, max_prompt = make_batch(tokenizer, samples, cfg.device)
    target = next(i for i, row in enumerate(samples) if row["case_id"] == cert["case_id"])
    position = max_prompt + int(cert["token_index"])
    if cfg.compile_model:
        with torch.no_grad(), attention_backend_context(cfg), precision_context(cfg):
            model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    with torch.no_grad(), attention_backend_context(cfg), precision_context(cfg):
        output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        logits = output.logits.float()
        logp = torch.log_softmax(logits[target, position - 1], dim=-1)[ids[target, position]]
        hidden = [value.detach().float().cpu() for value in output.hidden_states]
    del model, tokenizer
    cleanup_memory()
    return float(logp.item()), hidden


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe native output_hidden_states without user forward hooks.")
    parser.add_argument("--config", default="configs/hf_compile_sdpa_math_step5.yaml")
    parser.add_argument("--samples", default="data/phase6_step5_replay_samples.jsonl")
    parser.add_argument("--certificates", default="results/phase4_certificates.jsonl")
    parser.add_argument("--case-id", default="grpo_000001_2817771126c0")
    parser.add_argument("--token-index", type=int, default=80)
    parser.add_argument("--out", default="results/attribution/step5_hidden_state_probe.json")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    configure_determinism(0)
    config = load_config(args.config)
    cert = next(row for row in read_jsonl(args.certificates) if row.get("actual_fork") and row["case_id"] == args.case_id and int(row["token_index"]) == args.token_index)
    samples = target_batch(read_jsonl(args.samples), cert)
    ref_logp, ref_hidden = run(path_config(config, "path_ref"), samples, cert)
    alt_logp, alt_hidden = run(path_config(config, "path_alt"), samples, cert)
    if len(ref_hidden) != len(alt_hidden):
        raise ValueError("hidden-state count mismatch")
    layers = []
    for index, (ref, alt) in enumerate(zip(ref_hidden, alt_hidden, strict=True)):
        diff = alt - ref
        layers.append({"index": index, "l2": float(diff.norm().item()), "max_abs": float(diff.abs().max().item()), "nonzero": int(torch.count_nonzero(diff).item())})
    payload = {
        "schema_version": "forkcert.hidden_probe.v1",
        "fork_id": f"clip-step5-{args.case_id}-t{args.token_index}",
        "ref_logp": ref_logp,
        "alt_logp": alt_logp,
        "expected_ref_logp": cert["logp_ref"],
        "expected_alt_logp": cert["logp_alt"],
        "ref_replay_error": ref_logp - float(cert["logp_ref"]),
        "alt_replay_error": alt_logp - float(cert["logp_alt"]),
        "observation_preserves_compile_fork": abs(alt_logp - float(cert["logp_alt"])) <= 1e-6,
        "layers": layers,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in payload if k != "layers"}, indent=2))


if __name__ == "__main__":
    import torch
    main()
