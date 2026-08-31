#!/usr/bin/env python3
"""Frozen five-case recapture for Training Bias Profile v2.

Every state starts from the same checkpoint/carrier and zero AdamW moments.
The first 16 input states select directions and the last 16 confirm them.  The
runner stores only exact small vectors or three value-blind CountSketch views,
plus the joint Grams needed to reproduce every reported statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor")

import numpy as np
import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "src"), str(ROOT / "archive/round1_code/src"),
    str(ROOT / "scripts"),
]

from kernel_analyzer.short_persistence import _splitmix64  # noqa: E402
from kernel_analyzer.training_bias_profile import matched_training_bias_profile  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import load_model, tensor_digest  # noqa: E402
from scripts.run_heldout_lmhead_consequence import adam_delta  # noqa: E402
from scripts.run_phi64_lmhead_dx_repair import MMRepair  # noqa: E402
from scripts.run_qwen128_vproj_repair import VProjRepair  # noqa: E402
from scripts.run_qwen256_lmhead_property_confirmation import ShapeObserver  # noqa: E402


SKETCH_DIMENSION = 4096
SKETCH_SEEDS = (20260831, 20260861, 20260891)
CALIBRATION = tuple(range(16))
CONFIRMATION = tuple(range(16, 32))
_PACKED_SKETCH_CACHE: dict[tuple[int, int], np.ndarray] = {}

CONFIG = {
    "phi": {
        "case_id": "phi4_seq64_lmhead_dx",
        "architecture": "phi",
        "model": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "bank": "results/coverage/phi4_seq64_input_bank.json",
        "carrier": "model.norm.weight",
        "lr": 1e-4,
        "kind": "PHI_LMHEAD",
    },
    "qwen_lmhead": {
        "case_id": "qwen_seq128_lmhead_dx",
        "architecture": "qwen",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "bank": "results/coverage/qwen_seq128_input_bank.json",
        "carrier": "model.norm.weight",
        "lr": 1e-4,
        "kind": "QWEN_LMHEAD",
        "left_shape": (128, 151936),
        "right_shape": (151936, 2048),
    },
    "qwen_vproj": {
        "case_id": "qwen128_vproj_output",
        "architecture": "qwen",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "bank": "results/coverage/qwen_seq128_input_bank.json",
        "carrier": "model.layers.0.self_attn.v_proj.weight",
        "lr": 1e-5,
        "kind": "VPROJ",
        "output_shape": (128, 1024),
        "preflight_match_index": 2,
    },
    "mamba_inproj": {
        "case_id": "mamba_seq64_in_proj",
        "architecture": "mamba",
        "model": "/data1/tzh/models/state-spaces/mamba-130m-hf",
        "bank": "results/coverage/mamba_seq64_input_bank.json",
        "carrier": "backbone.layers.0.mixer.in_proj.weight",
        "lr": 1e-5,
        "kind": "INPROJ",
        "target_sha": "9c03ef3fc9b93005efed225a176c3e97efa91d33af2fae0b27cbb28e695c3cee",
    },
}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(state: dict[str, Any]) -> list[int]:
    values = state.get("token_ids", state.get("input_ids"))
    if values is None:
        raise RuntimeError("input state has no token IDs")
    return values


def _state_id(state: dict[str, Any], index: int) -> str:
    return str(state.get("state_id", state.get("sequence_id", index)))


def _compact_views(value: torch.Tensor | np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
    flat = tensor.detach().float().reshape(-1)
    coordinates = int(flat.numel())
    if coordinates <= SKETCH_DIMENSION:
        return {"EXACT": flat.cpu().numpy().copy()}, coordinates
    # Transfer a large tensor to host once.  The value-blind coordinate hashes
    # depend only on vector length and frozen seed, so cache their packed
    # bucket/sign codes and reuse them across states and stages.  Accumulation
    # order and float64 arithmetic remain identical to count_sketch_chunks.
    host = flat.cpu().numpy()
    result: dict[str, np.ndarray] = {}
    for seed in SKETCH_SEEDS:
        cache_key = (coordinates, seed)
        packed = _PACKED_SKETCH_CACHE.get(cache_key)
        if packed is None:
            packed = np.empty(coordinates, dtype=np.uint16)
            for start in range(0, coordinates, 1_000_000):
                stop = min(coordinates, start + 1_000_000)
                indices = np.arange(start, stop, dtype=np.uint64)
                hashed = _splitmix64(indices + np.uint64(seed))
                buckets = (hashed % np.uint64(SKETCH_DIMENSION)).astype(np.uint16)
                packed[start:stop] = buckets | (
                    ((hashed & np.uint64(1)) != 0).astype(np.uint16) << np.uint16(12)
                )
            _PACKED_SKETCH_CACHE[cache_key] = packed
        sketch = np.zeros(SKETCH_DIMENSION, dtype=np.float64)
        for start in range(0, coordinates, 1_000_000):
            stop = min(coordinates, start + 1_000_000)
            codes = packed[start:stop]
            buckets = (codes & np.uint16(SKETCH_DIMENSION - 1)).astype(np.int64)
            signs = np.where((codes & np.uint16(1 << 12)) == 0, 1.0, -1.0)
            values = np.asarray(host[start:stop], dtype=np.float64)
            sketch += np.bincount(
                buckets, weights=signs * values, minlength=SKETCH_DIMENSION,
            )
        result[f"SKETCH_SEED_{seed}"] = sketch.astype(np.float32)
    return result, coordinates


def _new_stage_store() -> dict[str, dict[str, dict[str, Any]]]:
    return {name: {} for name in ("LOCAL", "PARAMETER_GRADIENT", "ADAMW_UPDATE")}


def _append_views(
    store: dict[str, dict[str, dict[str, Any]]],
    stage: str,
    effect_views: dict[str, np.ndarray],
    repair_views: dict[str, np.ndarray],
    coordinate_count: int,
) -> None:
    if effect_views.keys() != repair_views.keys():
        raise RuntimeError(f"{stage}: effect and repair summaries differ")
    for view in effect_views:
        slot = store[stage].setdefault(view, {
            "effects": [], "repairs": [], "coordinate_count": coordinate_count,
        })
        if slot["coordinate_count"] != coordinate_count:
            raise RuntimeError(f"{stage}: coordinate count changed")
        slot["effects"].append(np.asarray(effect_views[view], dtype=np.float64))
        slot["repairs"].append(np.asarray(repair_views[view], dtype=np.float64))


def _append_contrast(
    store: dict[str, dict[str, dict[str, Any]]],
    stage: str,
    effect: torch.Tensor | np.ndarray,
    repair: torch.Tensor | np.ndarray,
) -> None:
    effect_views, effect_count = _compact_views(effect)
    repair_views, repair_count = _compact_views(repair)
    if effect_count != repair_count:
        raise RuntimeError(f"{stage}: effect and repair coordinates differ")
    _append_views(store, stage, effect_views, repair_views, effect_count)


def _finish_stages(store: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    unit_ids = [f"input-state-{index:02d}" for index in range(32)]
    result: dict[str, Any] = {}
    for stage_index, (stage, views) in enumerate(store.items()):
        result[stage] = {}
        for view_index, (view, slot) in enumerate(sorted(views.items())):
            effects = np.stack(slot["effects"])
            repairs = np.stack(slot["repairs"])
            if effects.shape[0] != 32:
                raise RuntimeError(f"{stage}/{view}: incomplete state population")
            result[stage][view] = {
                "coordinate_count": slot["coordinate_count"],
                "profile": matched_training_bias_profile(
                    effects,
                    repairs,
                    calibration_indices=CALIBRATION,
                    confirmation_indices=CONFIRMATION,
                    inference_unit_ids=unit_ids,
                    minimum_independent_units=8,
                    signflip_draws=4000,
                    seed=20261101 + stage_index * 100 + view_index * 3,
                    include_joint_gram=True,
                ),
            }
    return result


def _gradient_repeat_gate(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    equal = bool(torch.equal(first, second))
    return {
        "exact_equal": equal,
        "max_abs_difference": 0.0 if equal else float((first - second).abs().max().item()),
    }


def _make_compiled_observer(
    config: dict[str, Any], modules: list[Any], mode: str, target_sha: str | None,
) -> Any:
    kind = config["kind"]
    if kind == "PHI_LMHEAD":
        return MMRepair(
            modules,
            "SHAM" if mode == "SHAM" else "REPAIR_FP32_CAST_BF16",
            target_sha=target_sha,
            allow_shape_fallback=target_sha is None,
        )
    if kind == "QWEN_LMHEAD":
        return ShapeObserver(
            modules,
            "sham" if mode == "SHAM" else "fp32",
            [],
            left_shape=config["left_shape"],
            right_shape=config["right_shape"],
            target_sha=target_sha,
        )
    if kind == "VPROJ":
        return VProjRepair(
            modules,
            "SHAM" if mode == "SHAM" else "REPAIR_FP32_CAST_BF16",
            target_sha,
            expected_output_shape=config["output_shape"],
            expected_match_index=(config["preflight_match_index"] if target_sha is None else None),
        )
    if kind == "INPROJ":
        return VProjRepair(
            modules,
            "SHAM" if mode == "SHAM" else "REPAIR_FP32_CAST_BF16",
            target_sha,
        )
    raise ValueError(kind)


def _observer_vectors(observer: Any) -> tuple[Any, Any]:
    local = getattr(observer, "local_vector", None)
    repair = getattr(observer, "repair_vector", None)
    if local is None or repair is None:
        raise RuntimeError("repair did not expose local effect and repair vectors")
    return local, repair


def _observer_identity(observer: Any) -> dict[str, Any]:
    seen = getattr(observer, "seen", None)
    if not isinstance(seen, list) or len(seen) != 1:
        raise RuntimeError(f"repair did not bind one runtime source identity: {seen}")
    return dict(seen[0])


def run_compiled(case: str, device: torch.device) -> dict[str, Any]:
    config = CONFIG[case]
    protocol_path = ROOT / "results/property/training_bias_profile_v2/empirical_protocol.json"
    amendment_path = ROOT / "results/property/training_bias_profile_v2/empirical_protocol_amendment_1.json"
    optimizer_amendment_path = ROOT / "results/property/training_bias_profile_v2/empirical_protocol_amendment_2.json"
    bank_path = ROOT / config["bank"]
    bank = json.loads(bank_path.read_text())
    states = list(bank.get("states", bank.get("records")))
    if len(states) != 32:
        raise RuntimeError("v2 recapture requires exactly 32 frozen input states")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configure_candidate_runtime(24_000)
    model = load_model(config["architecture"], Path(config["model"]), device).eval()
    parameter = dict(model.named_parameters())[config["carrier"]]
    if config["kind"] != "INPROJ":
        for name, candidate_parameter in model.named_parameters():
            candidate_parameter.requires_grad_(name == config["carrier"])
    start = len(PyCodeCache.modules)
    compiled = torch.compile(LossStep(model), backend="inductor", fullgraph=False, dynamic=False)
    warm = torch.tensor([_tokens(states[0])], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    compiled(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    base = parameter.detach().float().clone()

    def branch(state: dict[str, Any], index: int, mode: str) -> tuple[str, torch.Tensor, Any | None]:
        with torch.no_grad():
            parameter.copy_(base.to(parameter.dtype))
        torch.manual_seed(24_000 + index)
        torch.cuda.manual_seed_all(24_000 + index)
        model.zero_grad(set_to_none=True)
        value = torch.tensor([_tokens(state)], dtype=torch.long, device=device)
        observer = None if mode == "CANDIDATE" else _make_compiled_observer(
            config, modules, mode, runtime_target_sha,
        )
        if observer is None:
            loss = compiled(value)
            loss.backward()
        else:
            with observer:
                loss = compiled(value)
                loss.backward()
        torch.cuda.synchronize(device)
        if parameter.grad is None:
            raise RuntimeError("declared carrier gradient is absent")
        return tensor_digest(loss), parameter.grad.detach().float().clone(), observer

    # Discover only the runtime call identity before measuring any case effect.
    with torch.no_grad():
        parameter.copy_(base.to(parameter.dtype))
    model.zero_grad(set_to_none=True)
    preflight = _make_compiled_observer(
        config, modules, "SHAM", config.get("target_sha"),
    )
    with preflight:
        compiled(warm).backward()
    torch.cuda.synchronize(device)
    preflight_identity = _observer_identity(preflight)
    runtime_target_sha = str(preflight_identity["source_sha"])
    preflight_local, _ = _observer_vectors(preflight)
    preflight_changed = float(np.linalg.norm(np.asarray(preflight_local).reshape(-1)))
    if preflight_changed != 0.0:
        raise RuntimeError(f"preflight sham changed the target output: {preflight_changed}")

    store = _new_stage_store()
    ids: list[str] = []
    determinism: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        loss_c, grad_c, _ = branch(state, index, "CANDIDATE")
        loss_repeat, grad_repeat, _ = branch(state, index, "CANDIDATE")
        repeat = _gradient_repeat_gate(grad_c, grad_repeat)
        repeat["loss_digest_equal"] = loss_c == loss_repeat
        repeat["state_id"] = _state_id(state, index)
        if not repeat["exact_equal"] or not repeat["loss_digest_equal"]:
            raise RuntimeError(f"candidate determinism failed at state {index}")
        loss_r, grad_r, observer = branch(state, index, "REPAIR")
        if config["kind"] in {"PHI_LMHEAD", "QWEN_LMHEAD"} and loss_c != loss_r:
            raise RuntimeError(f"backward-only repair changed forward loss at state {index}")
        if observer is None:
            raise RuntimeError("repair observer is absent")
        identity = _observer_identity(observer)
        if identity["source_sha"] != runtime_target_sha:
            raise RuntimeError("runtime repair source changed after preflight")
        local_effect, local_repair = _observer_vectors(observer)
        zeros = torch.zeros_like(grad_c)
        update_c, _, _ = adam_delta(
            grad_c, zeros, zeros, 1,
            learning_rate=config["lr"], beta1=0.9, beta2=0.95,
        )
        update_r, _, _ = adam_delta(
            grad_r, zeros, zeros, 1,
            learning_rate=config["lr"], beta1=0.9, beta2=0.95,
        )
        _append_contrast(store, "LOCAL", local_effect, local_repair)
        _append_contrast(store, "PARAMETER_GRADIENT", grad_c - grad_r, grad_r)
        _append_contrast(store, "ADAMW_UPDATE", update_c - update_r, update_r)
        ids.append(_state_id(state, index))
        determinism.append(repeat)
        boundary_rows.append(identity)
        print(json.dumps({
            "event": "TRAINING_BIAS_PROFILE_V2_STATE",
            "case": case,
            "step": index + 1,
        }), flush=True)
        del grad_c, grad_repeat, grad_r, update_c, update_r, zeros
        torch.cuda.empty_cache()

    return {
        "schema": "kernel-analyzer-training-bias-profile-v2-raw-case",
        "status": "COMPLETE",
        "case_id": config["case_id"],
        "case_key": case,
        "protocol_sha256": _file_sha(protocol_path),
        "protocol_amendment_sha256": _file_sha(amendment_path),
        "optimizer_amendment_sha256": _file_sha(optimizer_amendment_path),
        "input_bank": config["bank"],
        "input_bank_sha256": _file_sha(bank_path),
        "state_ids": ids,
        "calibration_state_ids": ids[:16],
        "confirmation_state_ids": ids[16:],
        "optimizer": {
            "name": "AdamW", "weight_decay": 0.0, "lr": config["lr"],
            "betas": [0.9, 0.95], "epsilon": 1e-8,
            "moments": "ZERO_AT_EVERY_INPUT_STATE",
        },
        "carrier": config["carrier"],
        "runtime_boundary": {
            "preflight": preflight_identity,
            "source_sha256": runtime_target_sha,
            "one_call_per_state": len(boundary_rows) == 32,
            "identities": boundary_rows,
        },
        "determinism": {
            "all_exact": all(
                row["exact_equal"] and row["loss_digest_equal"] for row in determinism
            ),
            "rows": determinism,
        },
        "stages": _finish_stages(store),
        "claim_boundary": (
            "32 frozen non-overlapping input windows at one checkpoint under cold-start "
            "AdamW with zero weight decay; not a random population sample, warm-moment, "
            "or independent-run result."
        ),
    }


class MultiViewLigerEndpoint:
    def __init__(self) -> None:
        self.original = None
        self.views: dict[str, np.ndarray] | None = None
        self.coordinate_count = 0
        self.calls = 0

    def __enter__(self) -> "MultiViewLigerEndpoint":
        import liger_kernel.ops.fused_linear_cross_entropy as fused
        self.original = fused.fused_linear_cross_entropy_forward
        original = self.original

        def wrapped(*args: Any, **kwargs: Any):
            result = original(*args, **kwargs)
            endpoint = result[4]
            if endpoint is None:
                raise RuntimeError("Liger fused dW endpoint is absent")
            self.views, self.coordinate_count = _compact_views(endpoint)
            self.calls += 1
            return result

        fused.fused_linear_cross_entropy_forward = wrapped
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        import liger_kernel.ops.fused_linear_cross_entropy as fused
        fused.fused_linear_cross_entropy_forward = self.original
        if self.calls != 1 or self.views is None:
            raise RuntimeError(f"Liger endpoint executed {self.calls} times")


def run_liger(device: torch.device) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    protocol_path = ROOT / "results/property/training_bias_profile_v2/empirical_protocol.json"
    amendment_path = ROOT / "results/property/training_bias_profile_v2/empirical_protocol_amendment_1.json"
    optimizer_amendment_path = ROOT / "results/property/training_bias_profile_v2/empirical_protocol_amendment_2.json"
    design_path = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/supplementary_state_design_v1.json"
    trajectory_path = ROOT / "results/trajectory/liger_protocol.json"
    design = json.loads(design_path.read_text())
    trajectory = json.loads(trajectory_path.read_text())
    records = {row["sequence_id"]: row for row in design["records"]}
    states = [records[key] for key in trajectory["trajectory"]["state_order"]]
    if len(states) != 32:
        raise RuntimeError("Liger recapture requires 32 frozen input states")

    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        "/data1/tzh/models/Qwen/Qwen3-1.7B",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    parameter = model.model.embed_tokens.weight
    for candidate_parameter in model.parameters():
        candidate_parameter.requires_grad_(candidate_parameter is parameter)
    base = parameter.detach().float().clone()
    candidate_module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=None,
    ).to(device)
    repair_module = LigerFusedLinearCrossEntropyLoss(
        ignore_index=-100, reduction="mean", accum_dtype=torch.float32,
    ).to(device)

    def branch(state: dict[str, Any], index: int, module: Any, *, capture_endpoint: bool = True):
        with torch.no_grad():
            parameter.copy_(base.to(parameter.dtype))
        torch.manual_seed(34_070 + index)
        torch.cuda.manual_seed_all(34_070 + index)
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([state["input_ids"]], dtype=torch.long, device=device)
        hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        observer = MultiViewLigerEndpoint() if capture_endpoint else None
        if observer is None:
            loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
            loss_digest = tensor_digest(loss)
            loss.backward()
        else:
            with observer:
                loss = module(model.lm_head.weight, hidden.reshape(-1, hidden.shape[-1]), labels)
                loss_digest = tensor_digest(loss)
                loss.backward()
        if parameter.grad is None or (observer is not None and observer.views is None):
            raise RuntimeError("Liger branch did not expose the declared F+B path")
        return (
            loss_digest,
            None if observer is None else observer.views,
            0 if observer is None else observer.coordinate_count,
            parameter.grad.detach().float().clone(),
        )

    store = _new_stage_store()
    ids: list[str] = []
    determinism: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        loss_c, endpoint_c, endpoint_count, grad_c = branch(state, index, candidate_module)
        loss_repeat, _, _, grad_repeat = branch(
            state, index, candidate_module, capture_endpoint=False,
        )
        repeat = _gradient_repeat_gate(grad_c, grad_repeat)
        repeat["loss_digest_equal"] = loss_c == loss_repeat
        repeat["state_id"] = str(state["sequence_id"])
        if not all((repeat["exact_equal"], repeat["loss_digest_equal"])):
            raise RuntimeError(f"Liger candidate determinism failed at state {index}")
        loss_r, endpoint_r, repair_count, grad_r = branch(state, index, repair_module)
        if loss_c != loss_r or endpoint_count != repair_count or endpoint_c.keys() != endpoint_r.keys():
            raise RuntimeError(f"Liger repair boundary failed at state {index}")
        zeros = torch.zeros_like(grad_c)
        update_c, _, _ = adam_delta(
            grad_c, zeros, zeros, 1, learning_rate=1e-4, beta1=0.9, beta2=0.95,
        )
        update_r, _, _ = adam_delta(
            grad_r, zeros, zeros, 1, learning_rate=1e-4, beta1=0.9, beta2=0.95,
        )
        _append_views(
            store,
            "LOCAL",
            {key: endpoint_c[key] - endpoint_r[key] for key in endpoint_c},
            endpoint_r,
            endpoint_count,
        )
        _append_contrast(store, "PARAMETER_GRADIENT", grad_c - grad_r, grad_r)
        _append_contrast(store, "ADAMW_UPDATE", update_c - update_r, update_r)
        ids.append(str(state["sequence_id"]))
        determinism.append(repeat)
        print(json.dumps({
            "event": "TRAINING_BIAS_PROFILE_V2_STATE", "case": "liger", "step": index + 1,
        }), flush=True)
        del grad_c, grad_repeat, grad_r, update_c, update_r, zeros
        torch.cuda.empty_cache()

    return {
        "schema": "kernel-analyzer-training-bias-profile-v2-raw-case",
        "status": "COMPLETE",
        "case_id": "liger_fused_ce_t128",
        "case_key": "liger",
        "protocol_sha256": _file_sha(protocol_path),
        "protocol_amendment_sha256": _file_sha(amendment_path),
        "optimizer_amendment_sha256": _file_sha(optimizer_amendment_path),
        "input_bank": str(design_path.relative_to(ROOT)),
        "input_bank_sha256": _file_sha(design_path),
        "state_order_sha256": _file_sha(trajectory_path),
        "state_ids": ids,
        "calibration_state_ids": ids[:16],
        "confirmation_state_ids": ids[16:],
        "optimizer": {
            "name": "AdamW", "weight_decay": 0.0, "lr": 1e-4,
            "betas": [0.9, 0.95], "epsilon": 1e-8,
            "moments": "ZERO_AT_EVERY_INPUT_STATE",
        },
        "carrier": "model.model.embed_tokens.weight (tied lm_head weight)",
        "runtime_boundary": {
            "function": "liger_kernel.ops.fused_linear_cross_entropy.fused_linear_cross_entropy_forward",
            "endpoint": "returned dW accumulator at tuple index 4",
            "one_call_per_state_per_branch": True,
        },
        "determinism": {
            "all_exact": all(
                row["exact_equal"] and row["loss_digest_equal"]
                for row in determinism
            ),
            "rows": determinism,
        },
        "stages": _finish_stages(store),
        "claim_boundary": (
            "32 frozen non-overlapping input windows at one checkpoint under cold-start "
            "AdamW with zero weight decay; not a random population sample, warm-moment, "
            "or independent-run result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("liger", *CONFIG), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    try:
        payload = run_liger(device) if args.case == "liger" else run_compiled(args.case, device)
    except Exception as error:
        payload = {
            "schema": "kernel-analyzer-training-bias-profile-v2-raw-case",
            "status": "ABSTAIN_EXECUTION_FAILURE",
            "case_key": args.case,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
