#!/usr/bin/env python3
"""Development-only natural-language training, with a common evaluation path.

Reuse model, batch and Liger boundary helpers. No direction injection, repeated
within-batch samples, character vocabulary, or quality-improvement claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import numpy as np
import torch
from scripts.run_training_numerical_v2 import BASE, ROOT, save_new
from scripts.run_liger_single_boundary_collapse import (
    batch_from_stream, build_model, backward_pass, file_sha256,
    make_loss_modules, materialize,
)

PILOT = BASE / "language_training_pilot"
TOKENIZER = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
DATA = Path("/data1/tzh/cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3")
PACKAGE = Path("/data1/tzh/envs/liger/lib/python3.10/site-packages/liger_kernel")


def prepare():
    import pyarrow as pa
    from transformers import AutoTokenizer
    if PILOT.exists():
        raise SystemExit("Pilot directory exists; preserve its frozen inputs")
    PILOT.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    sources = {}
    for split, filename, rows in (
        ("train", "wikitext-train-00000-of-00002.arrow", 20000),
        ("validation", "wikitext-validation.arrow", 2000),
    ):
        source = DATA / filename
        with pa.memory_map(str(source), "r") as memory:
            table = pa.ipc.open_stream(memory).read_all().slice(0, rows)
            text = "\n".join(table.column("text").to_pylist())
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= 1024:
            raise RuntimeError("Insufficient natural-language tokens")
        with (PILOT / f"{split}.npy").open("xb") as handle:
            np.save(handle, np.asarray(tokens, dtype=np.int64))
        sources[split] = {"path": str(source), "sha256": file_sha256(source),
                          "first_rows": rows, "token_count": len(tokens),
                          "encoded_sha256": file_sha256(PILOT / f"{split}.npy")}
    save_new(PILOT / "protocol.json", {
        "schema": "liger-natural-language-development-pilot-v1",
        "data_use": "DEVELOPMENT_ONLY_NOT_HELDOUT_CONFIRMATION",
        "purpose": "Execution cost, deterministic control and paired-loss feasibility",
        "data": sources, "tokenizer": str(TOKENIZER),
        "tokenizer_sha256": file_sha256(TOKENIZER / "tokenizer.json"),
        "model": {"vocab_size": len(tokenizer), "sequence_length": 128,
                  "embedding_dimension": 256, "layers": 4, "heads": 4,
                  "initialization_seed": 20260906, "batch_size": 2},
        "implementation": {"target_parameter": "transformer.wte.weight",
                           "changed_boundary": "Liger dW accumulation dtype only"},
        "steps": 64, "eval_every": 32, "evaluation_batches": 4,
        "optimizer": {"lr": 0.0001, "betas": [0.9, 0.95], "eps": 1e-8,
                      "weight_decay": 0.0, "foreach": False, "fused": False},
        "parameter_representation": "FP32_MASTER_BF16_MODEL_ALL_PARAMETERS",
        "evaluation": "Shared FP32 linear plus cross_entropy, identical held-out tokens",
        "conditions": ["candidate", "reference", "reference_repeat"],
        "source_sha256": {name: file_sha256(ROOT / name) for name in (
            "scripts/run_liger_language_pilot.py", "scripts/run_liger_single_boundary_collapse.py")},
        "liger_sources": {str(p.relative_to(PACKAGE)): file_sha256(p) for p in sorted(PACKAGE.rglob("*.py"))},
        "quality_claim": "NOT_ASSESSED_NO_POWER_DESIGN_YET",
    })


def train(condition, device):
    protocol = json.loads((PILOT / "protocol.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise RuntimeError("Frozen pilot source changed")
    for name, digest in protocol["liger_sources"].items():
        if file_sha256(PACKAGE / name) != digest:
            raise RuntimeError("Liger package changed")
    out = PILOT / condition
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("liger_kernel", PACKAGE / "__init__.py",
                                                submodule_search_locations=[str(PACKAGE)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["liger_kernel"] = module
    spec.loader.exec_module(module)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    train_tokens = np.load(PILOT / "train.npy")
    validation_tokens = np.load(PILOT / "validation.npy")
    for split in ("train", "validation"):
        if file_sha256(PILOT / f"{split}.npy") != protocol["data"][split]["encoded_sha256"]:
            raise RuntimeError("Frozen token data changed")
    model, target = build_model(protocol, device)
    modules = make_loss_modules(protocol, device)
    selected = modules["CANDIDATE" if condition == "candidate" else "REPAIR"]
    master = {n: torch.nn.Parameter(p.detach().float().clone()) for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(list(master.values()), **protocol["optimizer"])
    initial_digest = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in master.values())).hexdigest()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    evaluations = []
    with (out / "steps.jsonl").open("x") as log:
        for step in range(protocol["steps"]):
            materialize(model, master)
            inputs, labels, offsets = batch_from_stream(train_tokens, batch_size=2,
                sequence_length=128, stream=0, step=step, seed=20260906,
                repeat_within_batch=False, device=device)
            loss, _ = backward_pass(model, selected, inputs, labels, target_name=target)
            for name, parameter in model.named_parameters():
                master[name].grad = parameter.grad.detach().float() if parameter.grad is not None else None
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            row = {"step": step + 1, "loss": loss, "offsets": offsets}
            log.write(json.dumps(row) + "\n"); log.flush()
            if not np.isfinite(loss):
                save_new(out / "status.json", {"status": "NONFINITE", **row})
                return
            if (step + 1) % protocol["eval_every"] == 0:
                materialize(model, master)
                values = []
                with torch.no_grad():
                    for index in range(protocol["evaluation_batches"]):
                        x, y, _ = batch_from_stream(validation_tokens, batch_size=2,
                            sequence_length=128, stream=0, step=index, seed=20260907,
                            repeat_within_batch=False, device=device)
                        hidden = model.transformer(input_ids=x, use_cache=False).last_hidden_state
                        logits = torch.nn.functional.linear(hidden.float(), model.lm_head.weight.float())
                        values.append(float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))))
                evaluations.append({"step": step + 1, "shared_evaluation_loss": float(np.mean(values))})
                print(json.dumps({"condition": condition, **evaluations[-1]}), flush=True)
    torch.cuda.synchronize(device)
    checkpoint = out / "final.pt"
    torch.save({"master": {n: p.detach().cpu() for n, p in master.items()},
                "optimizer": optimizer.state_dict(), "steps": protocol["steps"]}, checkpoint)
    save_new(out / "status.json", {"status": "COMPLETE_DEVELOPMENT_PILOT",
        "condition": condition, "initial_parameters_sha256": initial_digest,
        "final_checkpoint_sha256": file_sha256(checkpoint), "evaluations": evaluations,
        "elapsed_seconds_including_evaluation": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "quality_claim": "NOT_ASSESSED_SINGLE_SHORT_DEVELOPMENT_RUN"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("prepare", "train"))
    p.add_argument("--condition", choices=("candidate", "reference", "reference_repeat"))
    p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        if args.condition is None:
            p.error("train requires --condition")
        train(args.condition, torch.device(args.device))
