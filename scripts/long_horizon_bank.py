#!/usr/bin/env python3
"""Build the resumable seq1024 Qwen3-1.7B natural-training state bank.

One optimizer update is exactly one full forward/backward unit.  The runner
keeps only immutable milestone model weights and one replaceable resume state.
No invocation tensors are retained; every 32 updates it stores compact,
deterministic gradient-carrier summaries.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data1/tzh").resolve()


def under_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if DATA_ROOT not in (resolved, *resolved.parents):
        raise ValueError(f"{label} must stay under {DATA_ROOT}: {resolved}")
    return resolved


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data1/tzh/cache/kernel_analyzer/long_horizon"))
    parser.add_argument("--manifest", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sketch-every", type=int, default=32)
    parser.add_argument("--resume-every", type=int, default=256)
    parser.add_argument("--milestones", type=int, nargs="+", default=[0, 64, 256, 1024, 2048, 4096])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def prepare_token_stream(
    *, tokenizer: Any, dataset: Any, path: Path, metadata_path: Path, token_count: int, protocol: dict[str, Any]
) -> dict[str, Any]:
    import numpy as np

    expected = {
        "schema": "kernel-analyzer-fixed-token-stream-v1",
        "token_count": token_count,
        "dtype": "int32",
        "protocol_sha256": canonical_digest(protocol),
    }
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in expected.items()) and path.stat().st_size == token_count * 4:
            return metadata
        raise RuntimeError("existing token stream does not match the frozen protocol; move it explicitly before rebuilding")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    stream = np.memmap(temporary, dtype=np.int32, mode="w+", shape=(token_count,))
    written = 0
    documents = 0
    pending: list[str] = []

    def flush() -> None:
        nonlocal written, documents
        if not pending or written >= token_count:
            pending.clear()
            return
        encoded = tokenizer(pending, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        for ids in encoded:
            if written >= token_count:
                break
            take = min(len(ids), token_count - written)
            if take:
                stream[written : written + take] = ids[:take]
                written += take
            documents += 1
        pending.clear()

    for row in dataset:
        text = str(row["text"]).strip()
        if text:
            pending.append(text)
        if len(pending) >= 256:
            flush()
        if written >= token_count:
            break
    flush()
    if written != token_count:
        del stream
        temporary.unlink()
        raise RuntimeError(f"natural token stream too short: {written} < {token_count}")
    stream.flush()
    del stream
    os.replace(temporary, path)
    metadata = {
        **expected,
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
        "nonempty_documents_consumed": documents,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
    }
    atomic_json(metadata_path, metadata)
    return metadata


class FixedSketch:
    """Candidate-blind fixed projections and rank proxies for parameter gradients."""

    def __init__(self, device: Any, samples: int = 256, power_iterations: int = 2) -> None:
        self.device = device
        self.samples = samples
        self.power_iterations = power_iterations
        self.sample_cache: dict[tuple[str, int], tuple[Any, Any]] = {}
        self.vector_cache: dict[tuple[str, int], Any] = {}

    @staticmethod
    def seed(name: str, suffix: str) -> int:
        return int.from_bytes(hashlib.sha256(f"{name}:{suffix}".encode()).digest()[:8], "little") % (2**31)

    def sample(self, name: str, size: int) -> tuple[Any, Any]:
        import torch

        key = (name, size)
        if key not in self.sample_cache:
            count = min(self.samples, size)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed(name, "projection"))
            if size <= 4096:
                indices = torch.randperm(size, generator=generator)[:count]
            else:
                # randperm(size) is O(size) memory and is catastrophic for
                # hundred-million-element embeddings.  Draw only O(count)
                # candidates and resolve the rare collisions on CPU.
                chosen: set[int] = set()
                while len(chosen) < count:
                    draws = torch.randint(0, size, (2 * (count - len(chosen)),), generator=generator)
                    chosen.update(int(value) for value in draws)
                indices = torch.tensor(sorted(chosen)[:count], dtype=torch.long)
            indices = indices.to(self.device)
            signs = torch.randint(0, 2, (count,), generator=generator, dtype=torch.int8)
            signs = signs.to(self.device, dtype=torch.float32).mul_(2).sub_(1).div_(math.sqrt(count))
            self.sample_cache[key] = (indices, signs)
        return self.sample_cache[key]

    def vector(self, name: str, size: int, dtype: Any) -> Any:
        import torch

        key = (name, size)
        if key not in self.vector_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed(name, "spectral"))
            vector = torch.randn(size, generator=generator, dtype=torch.float32)
            vector /= vector.norm().clamp_min(1e-30)
            self.vector_cache[key] = vector.to(self.device, dtype=dtype)
        return self.vector_cache[key]

    def summarize(self, model: Any) -> dict[str, Any]:
        import torch

        rows: list[dict[str, Any]] = []
        all_finite = True
        total_l2_sq = 0.0
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                rows.append({"name": name, "status": "NO_GRADIENT"})
                continue
            flat = gradient.detach().reshape(-1)
            indices, signs = self.sample(name, flat.numel())
            selected = flat.index_select(0, indices).float()
            projection = float(torch.dot(selected, signs))
            frobenius = float(torch.linalg.vector_norm(flat.float()))
            finite = math.isfinite(frobenius) and math.isfinite(projection)
            all_finite = all_finite and finite
            total_l2_sq += frobenius * frobenius
            row: dict[str, Any] = {
                "name": name,
                "status": "OK" if finite else "NONFINITE",
                "numel": flat.numel(),
                "l2": frobenius,
                "signed_projection": projection,
            }
            if gradient.ndim >= 2:
                matrix = gradient.detach().reshape(gradient.shape[0], -1)
                vector = self.vector(name, matrix.shape[1], matrix.dtype)
                for _ in range(self.power_iterations):
                    left = matrix @ vector
                    left = (left.float() / left.float().norm().clamp_min(1e-30)).to(matrix.dtype)
                    vector = matrix.transpose(0, 1) @ left
                    vector = (vector.float() / vector.float().norm().clamp_min(1e-30)).to(matrix.dtype)
                sigma = float((matrix @ vector).float().norm())
                # This is a deterministic power-iteration estimate.  The
                # corresponding stable rank is a declared effective-rank
                # proxy, not an exact entropy rank.
                stable_rank = (frobenius / sigma) ** 2 if sigma > 0.0 else None
                row.update({
                    "spectral_norm_power_estimate": sigma,
                    "effective_rank_proxy_stable_rank": stable_rank,
                    "power_iterations": self.power_iterations,
                })
            rows.append(row)
        return {
            "schema": "fixed-gradient-carrier-sketch-v1",
            "selection_used_candidate_values": False,
            "projection_samples_per_parameter": self.samples,
            "spectral_method": "fixed initialization, two power iterations",
            "effective_rank_boundary": "stable-rank proxy from full Frobenius norm and estimated spectral norm",
            "all_finite": all_finite,
            "global_l2": math.sqrt(total_l2_sq),
            "parameters": rows,
        }


def save_model(model: Any, path: Path) -> dict[str, Any]:
    from safetensors.torch import save_file

    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, str(temporary), metadata={"format": "pt", "source": "long_horizon_bank"})
    del state
    os.replace(temporary, path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_digest(path)}


def save_resume(model: Any, optimizer: Any, step: int, protocol_sha256: str, path: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.cuda.synchronize()
    payload = {
        "schema": "kernel-analyzer-long-horizon-resume-v1",
        "protocol_sha256": protocol_sha256,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }
    torch.save(payload, temporary)
    del payload
    os.replace(temporary, path)
    return {"path": str(path), "bytes": path.stat().st_size, "step": step}


def restore_resume(model: Any, optimizer: Any, path: Path, protocol_sha256: str, device: Any) -> int:
    import numpy as np
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("resume state protocol does not match current arguments")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in payload["rng"]["cuda"]])
    step = int(payload["step"])
    del payload
    gc.collect()
    return step


def main() -> None:
    args = parse_args()
    stop_after = args.steps if args.stop_after is None else args.stop_after
    if not (1 <= stop_after <= args.steps):
        raise ValueError("stop-after must be within [1, steps]")
    if args.seq_len != 1024 or args.batch_size != 4:
        raise ValueError("this frozen campaign requires the pilot-selected seq_len=1024, batch_size=4")
    if args.sketch_every < 1 or args.resume_every < 1:
        raise ValueError("sketch-every and resume-every must be positive")

    model_path = under_root(args.model, "model")
    output_dir = under_root(args.output_dir, "output-dir")
    manifest_path = under_root(args.manifest, "manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    protocol = {
        "schema": "kernel-analyzer-long-horizon-protocol-v1",
        "model": str(model_path),
        "dataset": ["Salesforce/wikitext", "wikitext-103-raw-v1", "train", "main"],
        "seed": args.seed,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "causal_label_alignment": "labels_equal_input_ids; exactly one internal model shift",
        "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0, "foreach": False},
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "sketch_every": args.sketch_every,
        "resume_every": args.resume_every,
        "milestones": sorted(set(args.milestones)),
    }
    protocol_sha256 = canonical_digest(protocol)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", revision="main",
        download_mode="reuse_dataset_if_exists",
    )
    token_path = output_dir / "tokens.int32"
    token_metadata_path = output_dir / "tokens.json"
    token_count = args.steps * args.batch_size * args.seq_len
    token_metadata = prepare_token_stream(
        tokenizer=tokenizer,
        dataset=dataset,
        path=token_path,
        metadata_path=token_metadata_path,
        token_count=token_count,
        protocol=protocol,
    )
    tokens = np.memmap(token_path, dtype=np.int32, mode="r", shape=(token_count,))

    print("loading Qwen3-1.7B", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(device)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False
    )
    resume_path = output_dir / "latest_resume.pt"
    start_step = 0
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume requested but missing: {resume_path}")
        print("restoring latest resume state", flush=True)
        start_step = restore_resume(model, optimizer, resume_path, protocol_sha256, device)
    elif resume_path.exists() or manifest_path.exists():
        raise FileExistsError("existing long-horizon state found; use --resume or move it explicitly")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("manifest protocol mismatch")
        if int(manifest.get("completed_step", -1)) != start_step:
            raise RuntimeError("manifest and resume state disagree on completed step")
        # Sketch rows after the last durable resume point may have been written
        # shortly before an interruption.  They are replayed, never duplicated.
        manifest["sketches"] = [row for row in manifest.get("sketches", []) if row["step"] <= start_step]
    else:
        manifest = {
            "schema": "kernel-analyzer-long-horizon-bank-v1",
            "protocol": protocol,
            "protocol_sha256": protocol_sha256,
            "token_stream": token_metadata,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(device),
            },
            "training_unit": "one full forward + one full backward + one AdamW update",
            "candidate_tensor_values_used_for_state_selection": False,
            "milestones": [{"step": 0, "kind": "IMMUTABLE_LOCAL_SOURCE_MODEL", "path": str(model_path)}],
            "sketches": [],
            "completed_step": 0,
            "status": "RUNNING",
        }
        atomic_json(manifest_path, manifest)

    manifest["execution_boundary"] = {
        "sdpa_memory_efficient_backward_bitwise_deterministic": False,
        "observed_runtime_warning": "PyTorch reports memory-efficient attention backward as nondeterministic",
        "claim": "saved states are real; from-scratch bitwise trajectory replay is not claimed",
        "bias_evaluation_requirement": "repeat each saved state and use reference replacement",
    }

    if not args.resume:
        print("saving initial replaceable resume state 0", flush=True)
        manifest["latest_resume"] = save_resume(model, optimizer, 0, protocol_sha256, resume_path)
        atomic_json(manifest_path, manifest)

    if start_step >= stop_after:
        print(json.dumps({"status": "ALREADY_AT_TARGET", "step": start_step}), flush=True)
        return

    sketcher = FixedSketch(device)
    milestones = set(args.milestones)
    run_started = time.perf_counter()
    for step in range(start_step + 1, stop_after + 1):
        offset = (step - 1) * args.batch_size * args.seq_len
        block = np.array(tokens[offset : offset + args.batch_size * args.seq_len], dtype=np.int64, copy=True)
        batch = torch.from_numpy(block.reshape(args.batch_size, args.seq_len)).to(device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        output = model(input_ids=batch, labels=batch, use_cache=False, return_dict=True)
        loss = output.loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        sketch = None
        if step % args.sketch_every == 0:
            sketch = sketcher.summarize(model)
            if not sketch["all_finite"]:
                raise FloatingPointError(f"nonfinite gradient sketch at step {step}")
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_started
        loss_value = float(loss.detach().cpu())
        if sketch is not None:
            manifest["sketches"].append({
                "step": step,
                "state_before_update": step - 1,
                "batch_token_offset": offset,
                "loss": loss_value,
                "step_seconds": elapsed,
                "tokens_per_second": args.batch_size * args.seq_len / elapsed,
                "gradient": sketch,
            })
        del output, loss, batch, block

        checkpoint_due = step in milestones or step % args.resume_every == 0 or step == stop_after
        if step in milestones:
            print(f"saving immutable milestone {step}", flush=True)
            checkpoint = output_dir / f"step_{step:04d}.safetensors"
            saved = save_model(model, checkpoint)
            manifest["milestones"] = [row for row in manifest["milestones"] if row["step"] != step]
            manifest["milestones"].append({"step": step, "kind": "POST_OPTIMIZER_UPDATE", **saved})
            manifest["milestones"].sort(key=lambda row: row["step"])
        if checkpoint_due:
            print(f"saving replaceable resume state {step}", flush=True)
            manifest["latest_resume"] = save_resume(model, optimizer, step, protocol_sha256, resume_path)
            manifest["completed_step"] = step
            manifest["status"] = "COMPLETE" if step == args.steps else "PAUSED_RESUMABLE"
            manifest["last_loss"] = loss_value
            manifest["run_wall_seconds"] = manifest.get("run_wall_seconds", 0.0) + (time.perf_counter() - run_started)
            atomic_json(manifest_path, manifest)
            run_started = time.perf_counter()
        elif sketch is not None:
            # The sketch is evidence, but completed_step only advances when a
            # matching resumable state exists.  An interrupted interval is
            # deterministically replayed from the previous resume point.
            atomic_json(manifest_path, manifest)
        if step == 1 or step % 16 == 0:
            print(json.dumps({
                "step": step,
                "target": stop_after,
                "loss": loss_value,
                "seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }), flush=True)

    print(json.dumps({"status": manifest["status"], "completed_step": manifest["completed_step"]}), flush=True)


if __name__ == "__main__":
    main()
