#!/usr/bin/env python3
"""Generate a compact manifest and real model checkpoints from natural LM training.

The bank is intentionally separate from the frozen text-state campaign.  Each
entry below step 0 is a model state *after an optimizer update* on contiguous
Wikitext-103 teacher-forcing batches.  Checkpoint tensors live under
``/data1/tzh/cache``; only the manifest and reproducible protocol belong in the
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data1/tzh").resolve()


def under_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if DATA_ROOT not in (resolved, *resolved.parents):
        raise ValueError(f"{label} must stay under {DATA_ROOT}: {resolved}")
    return resolved


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def digest_tensor(value: Any) -> str:
    import torch

    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(repr(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(digest_tensor(parameter).encode())
    return digest.hexdigest()


def gradient_summary(model: Any) -> dict[str, Any]:
    import torch

    total_sq = 0.0
    max_abs = 0.0
    non_none = 0
    digest = hashlib.sha256()
    top: list[tuple[float, str]] = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            digest.update(f"{name}:NONE".encode())
            continue
        non_none += 1
        value = grad.detach().float()
        norm = float(value.norm())
        total_sq += norm * norm
        max_abs = max(max_abs, float(value.abs().max()))
        top.append((norm, name))
        digest.update(name.encode())
        digest.update(digest_tensor(grad).encode())
    top.sort(reverse=True)
    return {
        "non_none": non_none,
        "l2": total_sq**0.5,
        "max_abs": max_abs,
        "top_parameter_norms": [{"name": n, "l2": x} for x, n in top[:12]],
        "sha256": digest.hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("/data1/tzh/cache/kernel_analyzer/qwen3_1p7b_natural_bank"))
    parser.add_argument("--manifest", type=Path, default=Path("results/final/natural_bank.json"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_checkpoint(model: Any, path: Path) -> str:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    # CPU copies prevent the serialized file from retaining GPU storage.
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(path), metadata={"format": "pt", "source": "natural_bank"})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    del state
    return digest


def main() -> None:
    args = parse_args()
    if args.seq_len < 8 or args.steps < 1 or args.batch_size < 1:
        raise ValueError("seq-len, steps, and batch-size must be positive")
    model_path = under_root(args.model, "model")
    output_dir = under_root(args.output_dir, "output-dir")
    manifest_path = under_root(args.manifest, "manifest")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"non-empty checkpoint bank: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep every cache explicit and under /data1/tzh for reproducibility.
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen natural checkpoint bank")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if dtype == torch.float16 and args.device.startswith("cuda"):
        raise ValueError("the first bank must use BF16; FP16 is reserved for a separate campaign")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    print("loading local natural dataset", flush=True)
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        revision="main",
        download_mode="reuse_dataset_if_exists",
    )
    required = args.steps * args.batch_size * (args.seq_len + 1)
    # Consume only as many natural documents as needed. Joining all 1.8M
    # Wikitext rows makes the protocol needlessly memory-heavy and obscures
    # whether the experiment is actually using a local cache.
    texts: list[str] = []
    token_ids: list[int] = []
    for row in dataset:
        text = str(row["text"]).strip()
        if not text:
            continue
        texts.append(text)
        token_ids.extend(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])
        if len(token_ids) >= required:
            break
    if len(token_ids) < required:
        raise RuntimeError(f"natural token stream too short: {len(token_ids)} < {required}")
    # Contiguous, deterministic blocks are a declared training protocol, not
    # the independent frozen text states used by the old campaign.
    blocks = [token_ids[i : i + args.seq_len + 1] for i in range(0, required, args.seq_len + 1)]
    block_digest = digest_json(blocks)

    print("loading Qwen checkpoint", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    checkpoint_steps = sorted({0, 1, 2, 4, 8, 16, 32, args.steps})
    checkpoint_steps = [step for step in checkpoint_steps if step <= args.steps]
    rows: list[dict[str, Any]] = []

    def write_checkpoint(step: int, loss: float | None, grad: dict[str, Any] | None) -> None:
        checkpoint = output_dir / f"step_{step:04d}.safetensors"
        file_sha = save_checkpoint(model, checkpoint)
        rows.append({
            "step": step,
            "kind": "INITIAL" if step == 0 else "POST_OPTIMIZER_UPDATE",
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "file_sha256": file_sha,
            "parameter_sha256": parameter_digest(model),
            "loss": loss,
            "gradient": grad,
        })

    print("saving initial checkpoint", flush=True)
    write_checkpoint(0, None, None)
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        print(f"training step {step}/{args.steps}", flush=True)
        start = (step - 1) * args.batch_size
        batch = torch.tensor(blocks[start : start + args.batch_size], dtype=torch.long, device=device)
        input_ids, labels = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True)
        loss = output.loss
        loss.backward()
        grad = gradient_summary(model)
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step in checkpoint_steps:
            print(f"saving checkpoint step {step}", flush=True)
            write_checkpoint(step, loss_value, grad)
        del output, loss, batch, input_ids, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    protocol = {
        "schema": "kernel-analyzer-natural-checkpoint-bank-v1",
        "subject": "Qwen3-1.7B dense causal LM",
        "model_path": str(model_path),
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "split": args.dataset_split,
            "revision": "main",
            "fingerprint": getattr(dataset, "_fingerprint", None),
            "nonempty_documents": len(texts),
            "token_count_used": required,
            "block_sha256": block_digest,
        },
        "training": {
            "seed": args.seed,
            "dtype": args.dtype,
            "attention_implementation": "sdpa",
            "optimizer": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "steps": args.steps,
            "checkpoint_steps": checkpoint_steps,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": getattr(__import__("datasets"), "__version__", None),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "natural_training": True,
        "not_frozen_text_state_campaign": True,
        "losses": losses,
        "checkpoints": rows,
    }
    protocol["protocol_sha256"] = digest_json(protocol)
    manifest_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "checkpoints": len(rows), "steps": args.steps, "loss_first": losses[0], "loss_last": losses[-1]}, sort_keys=True))


if __name__ == "__main__":
    main()
