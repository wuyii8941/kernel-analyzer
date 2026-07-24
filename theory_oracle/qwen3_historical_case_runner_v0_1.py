#!/usr/bin/env python
"""Run an opaque Qwen3 historical case and record generic local evidence.

The runner deliberately has no knowledge of a bug, a patch, or a preferred
operator.  It only instantiates the declared Qwen3 Attention subject, records
natural module boundaries, and emits a generic TorchDispatch operation trace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from forkcert.relational_oracle import compute_endpoint_oracle, compute_repeatability


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: Any) -> str:
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def tensor_record(value: Any) -> dict[str, Any]:
    value = value.detach()
    return {
        "sha256": tensor_hash(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "device": str(value.device),
    }


def first_tensor(value: Any) -> Any | None:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, dict):
        for item in value.values():
            result = first_tensor(item)
            if result is not None:
                return result
    return None


def output_records(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "detach"):
        return [tensor_record(value)]
    if isinstance(value, (tuple, list)):
        records: list[dict[str, Any]] = []
        for item in value:
            records.extend(output_records(item))
        return records
    if isinstance(value, dict):
        records = []
        for item in value.values():
            records.extend(output_records(item))
        return records
    return []


def clear_torchtitan_modules() -> None:
    """Allow a fresh process/import to select the requested source worktree."""

    for name in list(sys.modules):
        if name == "torchtitan" or name.startswith("torchtitan."):
            del sys.modules[name]


class GenericTrace:
    """Small generic dispatcher trace; no operation names are preselected."""

    def __init__(self, torch: Any) -> None:
        from torch.utils._python_dispatch import TorchDispatchMode

        self.records: list[dict[str, Any]] = []

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(inner, func: Any, types: Any, args: Any = (), kwargs: Any = None):
                result = func(*args, **(kwargs or {}))
                outputs = output_records(result)
                packet = getattr(func, "_overloadpacket", func)
                name = getattr(packet, "__name__", str(packet))
                owner = inner.inner_owner
                owner.records.append(
                    {"index": len(owner.records), "op": str(name), "outputs": outputs}
                )
                return result

        self.mode = Mode()
        self.mode.inner_owner = self  # type: ignore[attr-defined]

    def __enter__(self) -> "GenericTrace":
        self.mode.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self.mode.__exit__(*args)


def load_subject(source_root: Path, torch: Any) -> tuple[Any, Any, Any, Any]:
    clear_torchtitan_modules()
    # TorchTitan's historical case used the older ``is_causal`` keyword.  The
    # current nightly exposes the same operation as ``window_size=(-1, 0)``.
    # This harness-only adapter preserves the historical call contract for
    # both buggy and fixed worktrees; it is not a case-specific repair.
    import torch.nn.attention.varlen as varlen_module

    native_varlen = varlen_module.varlen_attn

    def historical_varlen_compat(*args: Any, is_causal: bool = False, scale: Any = None, **kwargs: Any) -> Any:
        window_size = (-1, 0) if is_causal else (-1, -1)
        # The historical Qwen3 case predates the current varlen API and its
        # native implementation requires Ampere+.  Tesla T4 is part of the
        # project's fixed test hardware, so use an equivalent per-document
        # SDPA fallback there.  It preserves the packed layout contract and
        # is shared by buggy/fixed sources; it is not a repair of the case.
        query = args[0]
        if query.is_cuda and torch.cuda.get_device_capability(query.device)[0] < 8:
            key, value, cu_q, cu_k, max_q, max_k = args[1:7]
            pieces = []
            q_bounds = cu_q.detach().cpu().tolist()
            k_bounds = cu_k.detach().cpu().tolist()
            for index in range(len(q_bounds) - 1):
                q_start, q_end = int(q_bounds[index]), int(q_bounds[index + 1])
                k_start, k_end = int(k_bounds[index]), int(k_bounds[index + 1])
                q_piece = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
                k_piece = key[k_start:k_end].transpose(0, 1).unsqueeze(0)
                v_piece = value[k_start:k_end].transpose(0, 1).unsqueeze(0)
                piece = torch.nn.functional.scaled_dot_product_attention(
                    q_piece,
                    k_piece,
                    v_piece,
                    is_causal=is_causal,
                    scale=scale,
                )
                pieces.append(piece.squeeze(0).transpose(0, 1))
            return torch.cat(pieces, dim=0)
        return native_varlen(*args, scale=scale, window_size=window_size, **kwargs)

    varlen_module.varlen_attn = historical_varlen_compat
    sys.path.insert(0, str(source_root))
    try:
        model_mod = importlib.import_module("torchtitan.models.qwen3.model.model")
        args_mod = importlib.import_module("torchtitan.models.qwen3.model.args")
        attention_mod = importlib.import_module("torchtitan.models.attention")
        return (
            model_mod.Attention,
            args_mod.Qwen3ModelArgs,
            model_mod.precompute_rope_cache,
            attention_mod.create_varlen_metadata_for_document,
        )
    finally:
        # Keep the imported module alive for the execution, but do not leave
        # this worktree ahead of the caller's normal import path.
        sys.path.remove(str(source_root))


def execute_case(source_root: Path, case_dir: Path, out_dir: Path) -> dict[str, Any]:
    import torch

    case_dir = case_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((case_dir / "case_manifest.json").read_text())
    config = manifest["subject"]["config"]
    Attention, Qwen3ModelArgs, precompute_rope_cache, make_metadata = load_subject(
        source_root.resolve(), torch
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3 historical case requires CUDA")
    torch.manual_seed(int(manifest["input"]["seed"]))
    torch.cuda.manual_seed_all(int(manifest["input"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    dtype = getattr(torch, str(config["dtype"]).split(".")[-1])
    args = Qwen3ModelArgs(**{key: value for key, value in config.items() if key != "dtype"})
    subject = Attention(args).to(device=device, dtype=dtype)
    state = torch.load(case_dir / manifest["artifacts"]["weights"], map_location="cpu", weights_only=True)
    subject.load_state_dict(state)
    subject.eval()
    packed = torch.load(case_dir / manifest["artifacts"]["inputs"], map_location="cpu", weights_only=True)
    x = packed["x"].to(device=device, dtype=dtype)
    tokens = packed["tokens"].to(device=device)
    rope = packed["rope_cache"].to(device=device, dtype=torch.float32)
    metadata = (
        make_metadata(tokens, int(config["eos_id"]))
        if config["attn_type"] == "varlen"
        else None
    )

    captured: dict[str, Any] = {}
    hooks = []
    region_names = list(manifest["subject"]["regions"])
    for name in region_names:
        module: Any = subject
        if name != "__root__":
            for part in name.split("."):
                module = getattr(module, part)
        def hook(_module: Any, _inputs: Any, output: Any, region: str = name) -> None:
            tensor = first_tensor(output)
            if tensor is not None:
                captured[region] = tensor.detach().cpu()
        hooks.append(module.register_forward_hook(hook))

    traces: list[dict[str, Any]] = []
    outputs: list[Any] = []
    with torch.no_grad():
        for _ in range(2):
            trace = GenericTrace(torch)
            with trace:
                result = subject(x, rope, metadata)
            torch.cuda.synchronize()
            outputs.append(result.detach().cpu())
            traces.append({"records": trace.records})
    for hook in hooks:
        hook.remove()

    output_dir = out_dir / "regions"
    output_dir.mkdir()
    region_manifest: dict[str, Any] = {}
    for name, value in captured.items():
        filename = name.replace(".", "__") + ".pt"
        path = output_dir / filename
        torch.save(value, path)
        region_manifest[name] = {
            "path": str(path.relative_to(out_dir)),
            "sha256": sha256_file(path),
            "tensor": tensor_record(value),
        }
    endpoint_path = out_dir / "endpoint.pt"
    torch.save(outputs[0], endpoint_path)
    trace_path = out_dir / "trace.json"
    trace_path.write_text(json.dumps(traces[0], indent=2, sort_keys=True) + "\n")
    source_files = [
        source_root / "torchtitan/models/qwen3/model/model.py",
        source_root / "torchtitan/models/attention.py",
    ]
    result = {
        "schema_version": "forkcert.qwen3-historical-case-run.v0.1",
        "case_id": manifest["case_id"],
        "source_root": str(source_root.resolve()),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "source_hashes": {str(path.relative_to(source_root)): sha256_file(path) for path in source_files},
        "endpoint": {
            "path": str(endpoint_path.relative_to(out_dir)),
            "sha256": sha256_file(endpoint_path),
            "tensor": tensor_record(outputs[0]),
            "repeatability": compute_repeatability(outputs),
        },
        "regions": region_manifest,
        "trace": {"path": str(trace_path.relative_to(out_dir)), "sha256": sha256_file(trace_path), "records": len(traces[0]["records"])},
        "raw_artifacts": {"input_fingerprint": tensor_record(x), "token_fingerprint": tensor_record(tokens)},
    }
    (out_dir / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(execute_case(args.source_root, args.case_dir, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
