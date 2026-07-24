from __future__ import annotations

import hashlib
import json
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable
from pathlib import Path


@dataclass(frozen=True)
class PathConfig:
    name: str
    model_name_or_path: str
    dtype: str = "bf16"
    autocast_dtype: str | None = None
    device: str = "cuda"
    compile_model: bool = False
    attn_implementation: str | None = None
    attention_backend: str | None = None
    logits_upcast_fp32: bool = True
    rmsnorm_reference: bool = False
    rmsnorm_no_upcast: bool = False
    rmsnorm_compile: bool = False
    materialize_bf16_outputs: bool = False
    materialization_dtype: str | None = None
    allow_bf16_reduced_precision_reduction: bool | None = None
    allow_fp16_reduced_precision_reduction: bool | None = None
    model_training_mode: bool = False
    gradient_checkpointing: bool = False


def _require_torch_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return torch, AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Phase 1 requires torch and transformers. Run in an environment with the ML dependencies installed."
        ) from exc


def _dtype(torch: Any, name: str) -> Any:
    table = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype: {name}")
    return table[name]


def configure_determinism(seed: int = 0, warn_only: bool = True) -> None:
    torch, _, _ = _require_torch_transformers()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False


def cleanup_memory() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


@contextmanager
def precision_context(config: PathConfig):
    torch, _, _ = _require_torch_transformers()
    if config.autocast_dtype is None:
        with nullcontext():
            yield
        return
    device_type = str(config.device).split(":", 1)[0]
    with torch.autocast(device_type=device_type, dtype=_dtype(torch, config.autocast_dtype)):
        yield


def load_hf_path(config: PathConfig):
    torch, AutoModelForCausalLM, AutoTokenizer = _require_torch_transformers()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "dtype": _dtype(torch, config.dtype),
        "trust_remote_code": True,
        "device_map": None,
    }
    if config.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if config.model_training_mode:
        model.train()
    else:
        model.eval()
    model.to(config.device)
    if config.rmsnorm_reference:
        patch_rmsnorm_reference(model)
    if config.rmsnorm_no_upcast:
        patch_rmsnorm_no_upcast(model)
    if config.rmsnorm_compile:
        compile_rmsnorm_modules(model)
    if config.materialize_bf16_outputs:
        from .hooks import attach_dtype_roundtrip

        roundtrip_dtype = _dtype(torch, config.materialization_dtype or config.dtype)
        model._forkcert_materialization_handles = attach_dtype_roundtrip(
            model,
            roundtrip_dtype,
            materialization_module_filter,
        )
    if config.compile_model:
        model = torch.compile(model)
    return tokenizer, model


def materialization_module_filter(name: str, module: Any) -> bool:
    if any(module.children()):
        return False
    return any(part in name.lower() for part in ["attn", "mlp", "proj", "norm", "act"])


def _reference_rmsnorm_forward(module: Any, hidden_states: Any) -> Any:
    torch, _, _ = _require_torch_transformers()
    input_dtype = hidden_states.dtype
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
    normalized = hidden_states.float() * torch.rsqrt(variance + eps)
    return module.weight * normalized.to(input_dtype)


def patch_rmsnorm_reference(model: Any) -> int:
    patched = 0
    for module in model.modules():
        if "rmsnorm" not in module.__class__.__name__.lower() or not hasattr(module, "weight"):
            continue
        module.forward = types.MethodType(_reference_rmsnorm_forward, module)
        patched += 1
    if patched == 0:
        raise ValueError("rmsnorm_reference requested but no RMSNorm modules were found")
    return patched


def _no_upcast_rmsnorm_forward(module: Any, hidden_states: Any) -> Any:
    torch, _, _ = _require_torch_transformers()
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
    return module.weight * (hidden_states * torch.rsqrt(variance + eps))


def patch_rmsnorm_no_upcast(model: Any) -> int:
    patched = 0
    for module in model.modules():
        if "rmsnorm" not in module.__class__.__name__.lower() or not hasattr(module, "weight"):
            continue
        module.forward = types.MethodType(_no_upcast_rmsnorm_forward, module)
        patched += 1
    if patched == 0:
        raise ValueError("rmsnorm_no_upcast requested but no RMSNorm modules were found")
    return patched


def compile_rmsnorm_modules(model: Any) -> int:
    torch, _, _ = _require_torch_transformers()
    compiled = 0
    for module in model.modules():
        if "rmsnorm" not in module.__class__.__name__.lower() or not hasattr(module, "weight"):
            continue
        module.forward = torch.compile(module.forward)
        compiled += 1
    if compiled == 0:
        raise ValueError("rmsnorm_compile requested but no RMSNorm modules were found")
    return compiled


@contextmanager
def attention_backend_context(config: PathConfig):
    torch, _, _ = _require_torch_transformers()
    backend = config.attention_backend
    old_reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    old_fp16_reduced = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    if config.allow_bf16_reduced_precision_reduction is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config.allow_bf16_reduced_precision_reduction
    if config.allow_fp16_reduced_precision_reduction is not None:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = config.allow_fp16_reduced_precision_reduction
    backend_context = nullcontext()
    if backend is None:
        try:
            with backend_context:
                yield
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_fp16_reduced
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        raise RuntimeError("attention_backend requires torch.nn.attention.sdpa_kernel support") from exc

    mapping = {
        "math": SDPBackend.MATH,
        "sdpa_math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "flash_attention": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", None),
    }
    key = backend.lower()
    selected = mapping.get(key)
    if selected is None:
        raise ValueError(f"unsupported attention_backend: {backend}")
    try:
        with sdpa_kernel(selected):
            yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduced
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = old_fp16_reduced


def _tokenize_prompt_response(tokenizer: Any, prompt: str, response: str, device: str) -> dict[str, Any]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    full_ids = tokenizer(prompt + response, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    if full_ids.numel() <= prompt_ids.numel():
        raise ValueError("response produced no additional tokens; check prompt/response formatting")
    if not bool((full_ids[: prompt_ids.numel()] == prompt_ids).all().item()):
        raise ValueError(
            "prompt tokenization is not a prefix of prompt+response tokenization; "
            "adjust prompt/response boundary, usually by moving the leading space into response."
        )
    return {
        "input_ids": full_ids.unsqueeze(0).to(device),
        "prompt_len": int(prompt_ids.numel()),
        "prompt_ids": prompt_ids.tolist(),
        "full_ids": full_ids.tolist(),
        "response_ids": full_ids[prompt_ids.numel() :].tolist(),
    }


def _ids_hash(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=16)
def model_artifact_fingerprint(model_name_or_path: str) -> dict[str, Any]:
    path = Path(model_name_or_path)
    if not path.is_dir():
        return {"kind": "remote_model_id", "model_id": model_name_or_path, "verified_local_files": False}
    candidates = []
    for item in sorted(path.iterdir()):
        if not item.is_file():
            continue
        if item.suffix in {".safetensors", ".bin", ".json", ".model"} or item.name.startswith("tokenizer"):
            candidates.append(item)
    digest = hashlib.sha256()
    files = []
    for item in candidates:
        file_digest = hashlib.sha256()
        with item.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                file_digest.update(chunk)
        relative = item.name
        value = file_digest.hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(value.encode("ascii"))
        files.append({"name": relative, "size": item.stat().st_size, "sha256": value})
    return {
        "kind": "local_checkpoint_files",
        "path": str(path.resolve()),
        "verified_local_files": bool(files),
        "aggregate_sha256": digest.hexdigest() if files else None,
        "files": files,
    }


def _encode_sample(tokenizer: Any, sample: dict[str, Any], device: str) -> dict[str, Any]:
    if "prompt_ids" not in sample and "response_ids" not in sample:
        return _tokenize_prompt_response(tokenizer, sample["prompt"], sample["response"], device)
    if "prompt_ids" not in sample or "response_ids" not in sample:
        raise ValueError("tokenized samples must contain both prompt_ids and response_ids")
    torch, _, _ = _require_torch_transformers()
    prompt_ids = [int(value) for value in sample["prompt_ids"]]
    response_ids = [int(value) for value in sample["response_ids"]]
    if not prompt_ids or not response_ids:
        raise ValueError("tokenized samples require non-empty prompt_ids and response_ids")
    full_ids = prompt_ids + response_ids
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0),
        "prompt_len": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "response_ids": response_ids,
    }


def tokenization_fingerprint(tokenizer: Any, prompt: str, response: str, device: str) -> dict[str, Any]:
    encoded = _tokenize_prompt_response(tokenizer, prompt, response, device)
    return {
        "prompt_token_count": len(encoded["prompt_ids"]),
        "response_token_count": len(encoded["response_ids"]),
        "full_token_count": len(encoded["full_ids"]),
        "prompt_token_hash": _ids_hash(encoded["prompt_ids"]),
        "response_token_hash": _ids_hash(encoded["response_ids"]),
        "full_token_hash": _ids_hash(encoded["full_ids"]),
    }


def tokenization_fingerprint_for_sample(tokenizer: Any, sample: dict[str, Any], device: str) -> dict[str, Any]:
    encoded = _encode_sample(tokenizer, sample, device)
    return {
        "prompt_token_count": len(encoded["prompt_ids"]),
        "response_token_count": len(encoded["response_ids"]),
        "full_token_count": len(encoded["full_ids"]),
        "prompt_token_hash": _ids_hash(encoded["prompt_ids"]),
        "response_token_hash": _ids_hash(encoded["response_ids"]),
        "full_token_hash": _ids_hash(encoded["full_ids"]),
    }


def token_logprob_with_grad(
    tokenizer: Any,
    model: Any,
    config: PathConfig,
    prompt: str,
    response: str,
    token_index: int,
    *,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
):
    torch, _, _ = _require_torch_transformers()
    sample = {"prompt": prompt, "response": response}
    if prompt_ids is not None or response_ids is not None:
        sample.update({"prompt_ids": prompt_ids, "response_ids": response_ids})
    encoded = _encode_sample(tokenizer, sample, config.device)
    input_ids = encoded["input_ids"]
    prompt_len = encoded["prompt_len"]
    full_pos = prompt_len + token_index
    if full_pos <= 0 or full_pos >= input_ids.shape[1]:
        raise IndexError(f"token_index {token_index} is outside response token range")
    with attention_backend_context(config), precision_context(config):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        if config.logits_upcast_fp32:
            logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits[:, full_pos - 1, :], dim=-1)
        return log_probs[0, input_ids[0, full_pos]]


def response_logprobs(tokenizer: Any, model: Any, config: PathConfig, prompt: str, response: str) -> list[dict[str, Any]]:
    torch, _, _ = _require_torch_transformers()
    encoded = _tokenize_prompt_response(tokenizer, prompt, response, config.device)
    input_ids = encoded["input_ids"]
    prompt_len = encoded["prompt_len"]
    rows: list[dict[str, Any]] = []
    with torch.inference_mode(), attention_backend_context(config), precision_context(config):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        if config.logits_upcast_fp32:
            logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        entropies = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        target_ids = input_ids[:, 1:]
        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    for full_pos in range(prompt_len, input_ids.shape[1]):
        pred_pos = full_pos - 1
        token_id = int(input_ids[0, full_pos].item())
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        rows.append(
            {
                "token_index": full_pos - prompt_len,
                "token_id": token_id,
                "token_text": token_text,
                "logp": float(target_log_probs[0, pred_pos].item()),
                "entropy": float(entropies[0, pred_pos].item()),
            }
        )
    return rows


def response_logprobs_for_sample(tokenizer: Any, model: Any, config: PathConfig, sample: dict[str, Any]) -> list[dict[str, Any]]:
    torch, _, _ = _require_torch_transformers()
    encoded = _encode_sample(tokenizer, sample, config.device)
    input_ids = encoded["input_ids"]
    prompt_len = encoded["prompt_len"]
    rows: list[dict[str, Any]] = []
    with torch.inference_mode(), attention_backend_context(config), precision_context(config):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        if config.logits_upcast_fp32:
            logits = logits.float()
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        entropies = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        target_ids = input_ids[:, 1:]
        target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    for full_pos in range(prompt_len, input_ids.shape[1]):
        pred_pos = full_pos - 1
        token_id = int(input_ids[0, full_pos].item())
        rows.append(
            {
                "token_index": full_pos - prompt_len,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
                "logp": float(target_log_probs[0, pred_pos].item()),
                "entropy": float(entropies[0, pred_pos].item()),
            }
        )
    return rows


def run_path_twice(
    config: PathConfig,
    samples: Iterable[dict[str, Any]],
    seed: int = 0,
    warmup_passes: int = 0,
) -> list[list[dict[str, Any]]]:
    configure_determinism(seed=seed)
    tokenizer, model = load_hf_path(config)
    all_runs: list[list[dict[str, Any]]] = []
    sample_list = list(samples)
    for _ in range(warmup_passes):
        for sample in sample_list:
            response_logprobs_for_sample(tokenizer, model, config, sample)
    for _ in range(2):
        run_rows: list[dict[str, Any]] = []
        for sample in sample_list:
            fingerprint = tokenization_fingerprint_for_sample(tokenizer, sample, config.device)
            token_rows = response_logprobs_for_sample(tokenizer, model, config, sample)
            for row in token_rows:
                run_rows.append({"case_id": sample["case_id"], **fingerprint, **row})
        all_runs.append(run_rows)
    return all_runs


def merge_pair_outputs(
    *,
    ref_runs: list[list[dict[str, Any]]],
    alt_runs: list[list[dict[str, Any]]],
    path_ref: str,
    path_alt: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(ref_runs) != 2 or len(alt_runs) != 2:
        raise ValueError("expected exactly two self runs for each path")
    if not (len(ref_runs[0]) == len(ref_runs[1]) == len(alt_runs[0]) == len(alt_runs[1])):
        raise ValueError("path outputs have different token counts")

    merged: list[dict[str, Any]] = []
    for i, (r0, r1, a0, a1) in enumerate(zip(ref_runs[0], ref_runs[1], alt_runs[0], alt_runs[1])):
        key = (r0["case_id"], r0["token_index"], r0["token_id"])
        for row in [r1, a0, a1]:
            if (row["case_id"], row["token_index"], row["token_id"]) != key:
                raise ValueError(f"token alignment mismatch at row {i}: {key} vs {row}")
            for field in ["prompt_token_hash", "response_token_hash", "full_token_hash"]:
                if row.get(field) != r0.get(field):
                    raise ValueError(f"tokenization fingerprint mismatch at row {i}: {field}")
        logp_ref = float(r0["logp"])
        logp_alt = float(a0["logp"])
        merged.append(
            {
                "case_id": r0["case_id"],
                "token_index": r0["token_index"],
                "token_id": r0["token_id"],
                "token_text": r0["token_text"],
                "path_ref": path_ref,
                "path_alt": path_alt,
                "logp_ref": logp_ref,
                "logp_alt": logp_alt,
                "logprob_delta": abs(logp_alt - logp_ref),
                "delta_self_ref": abs(float(r1["logp"]) - logp_ref),
                "delta_self_alt": abs(float(a1["logp"]) - logp_alt),
                "entropy_ref": float(r0.get("entropy", 0.0)),
                "entropy_alt": float(a0.get("entropy", 0.0)),
                "entropy_delta": abs(float(a0.get("entropy", 0.0)) - float(r0.get("entropy", 0.0))),
                "prompt_token_hash": r0["prompt_token_hash"],
                "response_token_hash": r0["response_token_hash"],
                "full_token_hash": r0["full_token_hash"],
                "prompt_token_count": r0["prompt_token_count"],
                "response_token_count": r0["response_token_count"],
                "full_token_count": r0["full_token_count"],
                "metadata": metadata,
            }
        )
    return merged
