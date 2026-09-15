#!/usr/bin/env python3
"""Paired training consequence for a same-FP32 Liger reduction-order change.

The two conditions use the same tiny language-model configuration, token
stream, initialization, and optimizer.  They differ only in the order of the
FP32 chunk contributions to the fused CE weight gradient.  This is a bounded
training consequence experiment for the already measured order effect.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

TEXT_PATH = Path("/data1/tzh/cache/kernel-analyzer/single_point_collapse_v1/tinyshakespeare.txt")
STEPS = 1024
SEQUENCE_LENGTH = 256
BATCH_SIZE = 8
EMBEDDING_DIMENSION = 256
LAYERS = 4
HEADS = 4
VOCAB_SIZE = 8192
INITIALIZATION_SEED = 20260905
BATCH_SEED = 20260905
PEAK_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 100
BETAS = (0.9, 0.95)
EPSILON = 1e-8


def _order_values(kind: str, count: int) -> list[int]:
    if kind == "original":
        return list(range(count))
    if kind == "even_then_odd":
        return list(range(0, count, 2)) + list(range(1, count, 2))
    raise ValueError(kind)


def install_order(kind: str, source: str, fused: Any) -> None:
    marker = "    for chunk_id in range(num_chunks):"
    if source.count(marker) != 1:
        raise RuntimeError("Liger chunk loop was not uniquely identified")
    transformed = source.replace(marker, "    for chunk_id in _declared_chunk_order(num_chunks):", 1)
    namespace = dict(fused.__dict__)
    namespace["_declared_chunk_order"] = lambda count: _order_values(kind, count)
    exec(compile(transformed, f"<liger-order-training-{kind}>", "exec"), namespace)
    fused.fused_linear_cross_entropy_forward = namespace["fused_linear_cross_entropy_forward"]


def encode_text() -> tuple[np.ndarray, np.ndarray]:
    text = TEXT_PATH.read_text(encoding="utf-8")
    symbols = sorted(set(text))
    mapping = {symbol: index for index, symbol in enumerate(symbols)}
    encoded = np.fromiter((mapping[symbol] for symbol in text), dtype=np.int64)
    split = int(len(encoded) * 0.9)
    return encoded[:split], encoded[split:]


def batch_from_stream(tokens: np.ndarray, *, step: int, device: torch.device):
    rng = np.random.default_rng(np.random.SeedSequence([BATCH_SEED, step]))
    offsets = rng.integers(0, len(tokens) - SEQUENCE_LENGTH - 1, size=BATCH_SIZE)
    inputs = np.stack([tokens[offset:offset + SEQUENCE_LENGTH] for offset in offsets])
    labels = np.stack([tokens[offset + 1:offset + SEQUENCE_LENGTH + 1] for offset in offsets])
    return (
        torch.from_numpy(inputs).to(device=device, dtype=torch.long),
        torch.from_numpy(labels).to(device=device, dtype=torch.long),
    )


def build_model(device: torch.device):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(INITIALIZATION_SEED)
    torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=SEQUENCE_LENGTH,
        n_ctx=SEQUENCE_LENGTH,
        n_embd=EMBEDDING_DIMENSION,
        n_layer=LAYERS,
        n_head=HEADS,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    ))
    model.tie_weights()
    return model.to(device=device, dtype=torch.float32).train()


def learning_rate(step: int) -> float:
    if step <= WARMUP_STEPS:
        return PEAK_LR * step / max(WARMUP_STEPS, 1)
    progress = min(1.0, (step - WARMUP_STEPS) / max(STEPS - WARMUP_STEPS, 1))
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * progress)) * (PEAK_LR - MIN_LR)


@torch.no_grad()
def evaluate(model: torch.nn.Module, module: Any, tokens: np.ndarray, device: torch.device) -> float:
    model.eval()
    inputs, labels = batch_from_stream(tokens, step=10_000, device=device)
    hidden = model.transformer(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
    loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1).contiguous())
    model.train()
    return float(loss.detach().float().item())


def run_condition(kind: str, train: np.ndarray, valid: np.ndarray, device: torch.device,
                  fused: Any, source: str) -> dict[str, Any]:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    install_order(kind, source, fused)
    model = build_model(device)
    module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PEAK_LR, betas=BETAS, eps=EPSILON,
        weight_decay=0.0, foreach=False, fused=False,
    )
    evaluations: dict[str, float] = {"0": evaluate(model, module, valid, device)}
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        inputs, labels = batch_from_stream(train, step=step, device=device)
        optimizer.zero_grad(set_to_none=True)
        hidden = model.transformer(input_ids=inputs, use_cache=False, return_dict=True).last_hidden_state
        loss = module(
            model.lm_head.weight,
            hidden.reshape(-1, hidden.shape[-1]),
            labels.reshape(-1).contiguous(),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        losses.append(float(loss.detach().float().item()))
        if step in (256, 512, 768, STEPS):
            torch.cuda.synchronize()
            evaluations[str(step)] = evaluate(model, module, valid, device)
        if step % 256 == 0:
            print(json.dumps({"event": "LIGER_ORDER_TRAINING", "kind": kind,
                              "step": step, "loss": losses[-1]}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "kind": kind,
        "status": "COMPLETE",
        "steps": STEPS,
        "training_loss": losses,
        "validation_loss_by_step": evaluations,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del optimizer, module, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("host GPU required")
    if not TEXT_PATH.is_file():
        raise RuntimeError(f"training text is unavailable: {TEXT_PATH}")

    import liger_kernel.ops.fused_linear_cross_entropy as fused
    source = inspect.getsource(fused.fused_linear_cross_entropy_forward)
    train, valid = encode_text()
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for kind in ("original", "even_then_odd"):
        results.append(run_condition(kind, train, valid, device, fused, source))

    by_kind = {row["kind"]: row for row in results}
    differences = {
        str(step): by_kind["original"]["validation_loss_by_step"][str(step)]
        - by_kind["even_then_odd"]["validation_loss_by_step"][str(step)]
        for step in (0, 256, 512, 768, STEPS)
    }
    payload = {
        "schema": "kernel-analyzer-liger-fp32-order-training-v1",
        "status": "COMPLETE",
        "comparison": {
            "candidate": "original sequential FP32 chunk order",
            "variant": "even-then-odd FP32 chunk order",
            "invariants": [
                "same model initialization",
                "same input batches and validation batch",
                "same optimizer and learning-rate schedule",
                "FP32 model, FP32 inputs, FP32 contribution accumulation",
                "only dW chunk-addition order differs",
            ],
            "primary_endpoint": "validation loss difference at step 1024",
            "difference_definition": "original minus even-then-odd",
        },
        "results": results,
        "validation_loss_difference": differences,
        "claim_boundary": (
            "One small GPT-2 model, one character-level text split, one fixed AdamW setting, "
            "and 1024 paired steps. This tests a same-FP32 implementation change and does not "
            "establish a general quality or stability result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
