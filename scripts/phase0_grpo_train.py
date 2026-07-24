#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


class CompileAudit:
    def __init__(self) -> None:
        self.backend_compiles = 0
        self.runtime_invocations = 0
        self.graph_code_sha256: list[str] = []
        self.graph_node_counts: list[int] = []


def make_tracking_backend(audit: CompileAudit):
    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")

    def backend(graph_module, example_inputs):
        audit.backend_compiles += 1
        audit.graph_code_sha256.append(
            hashlib.sha256(graph_module.code.encode("utf-8")).hexdigest()
        )
        audit.graph_node_counts.append(sum(1 for _ in graph_module.graph.nodes))
        compiled = inductor(graph_module, example_inputs)

        def counted(*args):
            audit.runtime_invocations += 1
            return compiled(*args)

        return counted

    return backend


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cpu_clone_tree(value: Any) -> Any:
    """Clone a nested training-state value to CPU without retaining GPU views."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone_tree(item) for item in value)
    return value


def tree_tensor_manifest(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if torch.is_tensor(value):
        rows.append(
            {
                "path": prefix or "<root>",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_sha256(value),
            }
        )
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            rows.extend(tree_tensor_manifest(value[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.extend(tree_tensor_manifest(item, f"{prefix}[{index}]"))
    return rows


def gradient_signature(model: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    non_none = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            digest.update(f"{name}:NONE\n".encode("utf-8"))
            continue
        non_none += 1
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(gradient).encode("utf-8"))
    return {
        "parameter_count": sum(1 for _ in model.parameters()),
        "non_none_gradient_count": non_none,
        "sha256": digest.hexdigest(),
    }


def tensor_version_signature(model: Any) -> str:
    digest = hashlib.sha256()
    for kind, values in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in values:
            digest.update(f"{kind}:{name}:{int(value._version)}\n".encode("utf-8"))
    return digest.hexdigest()


def rng_snapshot() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def restore_rng(snapshot: dict[str, Any]) -> None:
    torch.random.set_rng_state(snapshot["cpu"])
    torch.cuda.set_rng_state_all(snapshot["cuda"])


def rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return torch.equal(left["cpu"], right["cpu"]) and len(left["cuda"]) == len(
        right["cuda"]
    ) and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"], strict=True))


def numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def trainer_compute_dtype(training_args: Any) -> str:
    if bool(getattr(training_args, "bf16", False)):
        return "bf16"
    if bool(getattr(training_args, "fp16", False)):
        return "fp16"
    raise ValueError("ForkCert online scan requires exactly one of bf16/fp16 training compute")


def target_number(answer: str) -> float | None:
    text = answer.split("####")[-1]
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def numeric_reward(completions: list[Any], answer: list[str], **_: Any) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        if isinstance(completion, list):
            text = str(completion[0].get("content", "")) if completion else ""
        else:
            text = str(completion)
        expected = target_number(gold)
        predictions = NUMBER_RE.findall(text)
        if expected is None or not predictions:
            rewards.append(-1.0)
            continue
        try:
            predicted = float(predictions[-1].replace(",", ""))
        except ValueError:
            rewards.append(-1.0)
            continue
        error = abs(predicted - expected)
        scale = max(1.0, abs(expected))
        proximity = 1.0 / (1.0 + error / scale)
        rewards.append(proximity + (1.0 if math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-6) else 0.0))
    return rewards


def case_id(rollout_batch: int, prompt_ids: list[int], completion_ids: list[int]) -> str:
    payload = json.dumps([prompt_ids, completion_ids], separators=(",", ":")).encode("utf-8")
    return f"grpo_{rollout_batch:06d}_{hashlib.sha256(payload).hexdigest()[:12]}"


class InstrumentedGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args: Any,
        dump_path: Path,
        samples_path: Path,
        append: bool = False,
        snapshot_step: int | None = None,
        snapshot_dir: Path | None = None,
        online_compile_scan_path: Path | None = None,
        online_logsoftmax_scan_path: Path | None = None,
        online_grad_compile_scan_path: Path | None = None,
        online_grad_compile_state_path: Path | None = None,
        transition_capture_step: int | None = None,
        transition_capture_dir: Path | None = None,
        transition_capture_targets: list[dict[str, Any]] | None = None,
        stop_after_snapshot: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dump_path = dump_path
        self.samples_path = samples_path
        self.dump_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._dump_fh = self.dump_path.open(mode, encoding="utf-8")
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self._samples_fh = self.samples_path.open(mode, encoding="utf-8")
        existing_samples = read_jsonl(self.samples_path) if append and self.samples_path.exists() else []
        self._forkcert_sample_ids = {str(row["case_id"]) for row in existing_samples}
        self._forkcert_snapshot_step = snapshot_step
        self._forkcert_snapshot_dir = snapshot_dir
        self._forkcert_snapshot_saved = False
        self._forkcert_stop_after_snapshot = stop_after_snapshot
        self._forkcert_online_scan_path = online_compile_scan_path
        self._forkcert_online_scan_fh = None
        self._forkcert_logsoftmax_scan_fh = None
        self._forkcert_grad_scan_fh = None
        self._forkcert_grad_state_fh = None
        self._forkcert_compiled_model = None
        self._forkcert_compile_audit = CompileAudit()
        self._forkcert_grad_compiled_model = None
        self._forkcert_grad_compile_audit = CompileAudit()
        self._forkcert_transition_capture_step = transition_capture_step
        self._forkcert_transition_capture_dir = transition_capture_dir
        self._forkcert_transition_snapshot_saved = False
        self._forkcert_transition_history: list[dict[str, Any]] = []
        if transition_capture_targets is not None and transition_capture_step is not None:
            raise ValueError("multi-target and legacy single-target transition capture are exclusive")
        if transition_capture_targets is None:
            transition_capture_targets = []
            if transition_capture_step is not None:
                if transition_capture_dir is None:
                    raise ValueError("transition_capture_step requires transition_capture_dir")
                transition_capture_targets.append(
                    {
                        "optimizer_step": int(transition_capture_step),
                        "state_id": f"legacy-step-{int(transition_capture_step)}",
                        "capture_dir": transition_capture_dir,
                        "plan_digest": None,
                        "history_selection": "FINAL_POLICY_ITERATION_ONLY",
                    }
                )
        steps = [int(item["optimizer_step"]) for item in transition_capture_targets]
        directories = [str(Path(item["capture_dir"]).resolve()) for item in transition_capture_targets]
        if len(steps) != len(set(steps)) or len(directories) != len(set(directories)):
            raise ValueError("transition capture target steps and directories must be unique")
        self._forkcert_transition_targets: list[dict[str, Any]] = [
            {
                **item,
                "optimizer_step": int(item["optimizer_step"]),
                "capture_dir": Path(item["capture_dir"]),
                "history": [],
                "saved": False,
            }
            for item in sorted(transition_capture_targets, key=lambda value: int(value["optimizer_step"]))
        ]
        if online_compile_scan_path is not None:
            online_compile_scan_path.parent.mkdir(parents=True, exist_ok=True)
            self._forkcert_online_scan_fh = online_compile_scan_path.open("w", encoding="utf-8")
        if online_logsoftmax_scan_path is not None:
            online_logsoftmax_scan_path.parent.mkdir(parents=True, exist_ok=True)
            self._forkcert_logsoftmax_scan_fh = online_logsoftmax_scan_path.open("w", encoding="utf-8")
        if (online_grad_compile_scan_path is None) != (online_grad_compile_state_path is None):
            raise ValueError("grad compile token and state JSONL paths must be supplied together")
        if online_grad_compile_scan_path is not None:
            online_grad_compile_scan_path.parent.mkdir(parents=True, exist_ok=True)
            online_grad_compile_state_path.parent.mkdir(parents=True, exist_ok=True)
            self._forkcert_grad_scan_fh = online_grad_compile_scan_path.open("w", encoding="utf-8")
            self._forkcert_grad_state_fh = online_grad_compile_state_path.open("w", encoding="utf-8")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._save_pre_minibatch_snapshot(model)
        self._capture_transition_state(model, inputs)
        self._dump_margin_inputs(model, inputs)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    def _capture_transition_state(self, model: Any, inputs: dict[str, Any]) -> None:
        for target_state in self._forkcert_transition_targets:
            self._capture_transition_state_for_target(model, inputs, target_state)

    def _capture_transition_state_for_target(
        self, model: Any, inputs: dict[str, Any], target_state: dict[str, Any]
    ) -> None:
        target = int(target_state["optimizer_step"])
        step = int(self.state.global_step)
        policy_iteration = int(self._step % self.num_iterations)
        history_selection = target_state.get(
            "history_selection", "FINAL_POLICY_ITERATION_ONLY"
        )
        if history_selection not in {"FINAL_POLICY_ITERATION_ONLY", "EVERY_OPTIMIZER_PRE_STEP"}:
            raise ValueError(f"unsupported transition history selection: {history_selection}")
        if step > target or (
            history_selection == "FINAL_POLICY_ITERATION_ONLY"
            and policy_iteration != self.num_iterations - 1
        ):
            return

        capture_dir = Path(target_state["capture_dir"])
        history_dir = capture_dir / "compiler_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_path = history_dir / f"step_{step:06d}_inputs.pt"
        if history_path.exists():
            raise FileExistsError(f"refuse to overwrite transition history: {history_path}")

        rng_before = rng_snapshot()
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        gradients_before = gradient_signature(model)
        versions_before = tensor_version_signature(model)
        cpu_inputs = cpu_clone_tree(inputs)
        torch.save(cpu_inputs, history_path)
        history_record = {
            "optimizer_step": step,
            "trainer_internal_step": int(self._step),
            "policy_iteration": policy_iteration,
            "rollout_batch": int(self._step // self.num_iterations),
            "path": str(history_path.resolve()),
            "tensor_manifest": tree_tensor_manifest(cpu_inputs),
        }
        del cpu_inputs
        target_state["history"].append(history_record)

        if step == target:
            if target_state["saved"]:
                raise RuntimeError("transition snapshot target encountered more than once")
            capture_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(model)
            model_state = self.accelerator.get_state_dict(model)
            unwrapped.save_pretrained(capture_dir, state_dict=model_state, safe_serialization=True)
            del model_state
            self.processing_class.save_pretrained(capture_dir)

            optimizer_state = cpu_clone_tree(self.optimizer.state_dict())
            torch.save(optimizer_state, capture_dir / "optimizer.pt")
            del optimizer_state
            scheduler_state = (
                cpu_clone_tree(self.lr_scheduler.state_dict()) if self.lr_scheduler is not None else None
            )
            torch.save(scheduler_state, capture_dir / "scheduler.pt")
            del scheduler_state
            scaler = getattr(self.accelerator, "scaler", None)
            scaler_state = cpu_clone_tree(scaler.state_dict()) if scaler is not None else None
            torch.save(scaler_state, capture_dir / "scaler.pt")
            del scaler_state
            full_rng = {
                "torch": cpu_clone_tree(rng_before),
                "python": python_rng_before,
                "numpy": numpy_rng_before,
            }
            torch.save(full_rng, capture_dir / "rng_state.pth")
            del full_rng
            self.state.save_to_json(str(capture_dir / "trainer_state.json"))
            metadata = {
                "schema_version": "forkcert.full-pre-minibatch-transition-state.v0.1",
                "state": "pre_minibatch",
                "optimizer_step": step,
                "trainer_internal_step": int(self._step),
                "policy_iteration": policy_iteration,
                "num_iterations": int(self.num_iterations),
                "rollout_batch": int(self._step // self.num_iterations),
                "state_id": str(target_state["state_id"]),
                "training_horizon_optimizer_steps": int(self.args.max_steps),
                "transition_capture_plan_digest": target_state.get("plan_digest"),
                "history_selection": history_selection,
                "capture_target_identity": {
                    key: target_state.get(key)
                    for key in (
                        "query_id",
                        "trajectory_id",
                        "trajectory_anchor",
                        "trajectory_seed",
                        "data_slice_id",
                        "phase",
                        "eligible_step_population",
                        "state_selection_prng_seed",
                        "state_id",
                    )
                },
                "compiler_history": target_state["history"],
                "target_minibatch_path": history_record["path"],
                "gradient_signature_before": gradients_before,
                "tensor_versions_before": versions_before,
                "contains": [
                    "model parameters and buffers",
                    "optimizer state",
                    "scheduler state",
                    "AMP scaler state or explicit null",
                    "Python/NumPy/Torch CPU and CUDA RNG state",
                    "exact target minibatch",
                    "prior measured-shape input history",
                    "Trainer state",
                ],
            }
            (capture_dir / "forkcert_transition_snapshot.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            target_state["saved"] = True
            gc.collect()

        rng_after = rng_snapshot()
        python_rng_after = random.getstate()
        numpy_rng_after = np.random.get_state()
        gradients_after = gradient_signature(model)
        versions_after = tensor_version_signature(model)
        rng_preserved = rng_equal(rng_before, rng_after)
        python_rng_preserved = python_rng_before == python_rng_after
        numpy_rng_preserved = numpy_rng_equal(numpy_rng_before, numpy_rng_after)
        gradients_preserved = gradients_before == gradients_after
        versions_preserved = versions_before == versions_after
        history_record.update(
            {
                "torch_rng_preserved": rng_preserved,
                "python_rng_preserved": python_rng_preserved,
                "numpy_rng_preserved": numpy_rng_preserved,
                "gradients_preserved": gradients_preserved,
                "tensor_versions_preserved": versions_preserved,
            }
        )
        if step == target:
            metadata.update(
                {
                    "gradient_signature_after": gradients_after,
                    "tensor_versions_after": versions_after,
                    "torch_rng_preserved": rng_preserved,
                    "python_rng_preserved": python_rng_preserved,
                    "numpy_rng_preserved": numpy_rng_preserved,
                    "gradients_preserved": gradients_preserved,
                    "tensor_versions_preserved": versions_preserved,
                }
            )
            (capture_dir / "forkcert_transition_snapshot.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if not rng_preserved:
            raise RuntimeError("transition capture changed Torch RNG state")
        if not python_rng_preserved:
            raise RuntimeError("transition capture changed Python RNG state")
        if not numpy_rng_preserved:
            raise RuntimeError("transition capture changed NumPy RNG state")
        if not gradients_preserved:
            raise RuntimeError("transition capture changed model gradients")
        if not versions_preserved:
            raise RuntimeError("transition capture changed parameter/buffer tensor versions")

    def _save_pre_minibatch_snapshot(self, model: Any) -> None:
        if self._forkcert_snapshot_saved or self._forkcert_snapshot_step is None:
            return
        if int(self.state.global_step) != self._forkcert_snapshot_step:
            return
        if self._forkcert_snapshot_dir is None:
            raise ValueError("snapshot_step requires snapshot_dir")
        snapshot_dir = self._forkcert_snapshot_dir
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(model)
        state_dict = self.accelerator.get_state_dict(model)
        unwrapped.save_pretrained(snapshot_dir, state_dict=state_dict, safe_serialization=True)
        self.processing_class.save_pretrained(snapshot_dir)
        metadata = {
            "state": "pre_minibatch",
            "optimizer_step": int(self.state.global_step),
            "trainer_internal_step": int(self._step),
            "policy_iteration": int(self._step % self.num_iterations),
            "rollout_batch": int(self._step // self.num_iterations),
        }
        (snapshot_dir / "forkcert_snapshot.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._forkcert_snapshot_saved = True
        if self._forkcert_stop_after_snapshot:
            self.control.should_training_stop = True

    @torch.no_grad()
    def _dump_margin_inputs(self, model, inputs: dict[str, Any]) -> None:
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        current_logps, _, _ = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            completion_ids.size(1),
            compute_entropy=False,
        )
        old_logps = inputs.get("old_per_token_logps")
        if old_logps is None:
            old_logps = current_logps.detach()
        advantages = inputs["advantages"]
        rollout_batch = int(self._step // self.num_iterations)
        policy_iteration = int(self._step % self.num_iterations)

        if policy_iteration == self.num_iterations - 1 and self._forkcert_online_scan_fh is not None:
            self._scan_compile_pair(
                model=model,
                inputs=inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                current_logps=current_logps,
                old_logps=old_logps,
                advantages=advantages,
                rollout_batch=rollout_batch,
                policy_iteration=policy_iteration,
            )
        if policy_iteration == self.num_iterations - 1 and self._forkcert_logsoftmax_scan_fh is not None:
            self._scan_logsoftmax_input_pair(
                model=model,
                inputs=inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                current_logps=current_logps,
                old_logps=old_logps,
                advantages=advantages,
                rollout_batch=rollout_batch,
                policy_iteration=policy_iteration,
            )
        if policy_iteration == self.num_iterations - 1 and self._forkcert_grad_scan_fh is not None:
            with torch.enable_grad():
                self._scan_grad_compile_pair(
                    model=model,
                    inputs=inputs,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    old_logps=old_logps,
                    advantages=advantages,
                    rollout_batch=rollout_batch,
                    policy_iteration=policy_iteration,
                )

        for batch_index in range(completion_ids.size(0)):
            prompt_tokens = prompt_ids[batch_index][prompt_mask[batch_index].bool()].tolist()
            valid = completion_mask[batch_index].bool()
            response_tokens = completion_ids[batch_index][valid].tolist()
            cid = case_id(rollout_batch, prompt_tokens, response_tokens)
            prompt_text = self.processing_class.decode(prompt_tokens, skip_special_tokens=False)
            response_text = self.processing_class.decode(response_tokens, skip_special_tokens=False)
            sample_row = {
                "case_id": cid,
                "prompt": prompt_text,
                "response": response_text,
                "prompt_ids": prompt_tokens,
                "response_ids": response_tokens,
                "metadata": {
                    "source": "trl_grpo_rollout",
                    "rollout_batch": rollout_batch,
                },
            }
            if cid not in self._forkcert_sample_ids:
                self._samples_fh.write(json.dumps(sample_row, ensure_ascii=False, sort_keys=True) + "\n")
                self._forkcert_sample_ids.add(cid)
            advantage = float(advantages[batch_index].item())
            for token_index, token_id in enumerate(response_tokens):
                row = {
                    "case_id": cid,
                    "token_index": token_index,
                    "token_id": int(token_id),
                    "token_text": self.processing_class.decode([token_id], skip_special_tokens=False),
                    "old_logp": float(old_logps[batch_index, token_index].item()),
                    "new_logp": float(current_logps[batch_index, token_index].item()),
                    "advantage": advantage,
                    "advantage_sign": 1 if advantage > 0 else -1 if advantage < 0 else 0,
                    "optimizer_step": int(self.state.global_step),
                    "rollout_batch": rollout_batch,
                    "policy_iteration": policy_iteration,
                    "epoch": rollout_batch,
                    "minibatch": policy_iteration,
                    "state": "pre_minibatch",
                    "training_kind": "trl_grpo",
                    "advantage_source": "trl_group_normalized_rewards",
                    "old_logp_source": "trl_old_per_token_logps",
                }
                self._dump_fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._dump_fh.flush()
        self._samples_fh.flush()

    @torch.no_grad()
    def _scan_compile_pair(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
        input_ids: Any,
        attention_mask: Any,
        current_logps: Any,
        old_logps: Any,
        advantages: Any,
        rollout_batch: int,
        policy_iteration: int,
    ) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        compute_dtype = trainer_compute_dtype(self.args)

        def score(target_model: Any):
            with sdpa_kernel(SDPBackend.MATH):
                values, _, _ = self._get_per_token_logps_and_entropies(
                    target_model,
                    input_ids,
                    attention_mask,
                    completion_ids.size(1),
                    compute_entropy=False,
                )
            return values.detach()

        # Recompute both measured eager runs under the same explicit MATH lock.
        ref_first = score(model)
        ref_second = score(model)
        if self._forkcert_compiled_model is None:
            self._forkcert_compiled_model = torch.compile(
                model, backend=make_tracking_backend(self._forkcert_compile_audit)
            )
        # The first compiled call for each batch is a discarded warm-up. The
        # next two calls are measured and provide an online self control.
        before_warm = self._forkcert_compile_audit.runtime_invocations
        score(self._forkcert_compiled_model)
        warm_invocations = self._forkcert_compile_audit.runtime_invocations - before_warm
        before_first = self._forkcert_compile_audit.runtime_invocations
        alt_first = score(self._forkcert_compiled_model)
        first_invocations = self._forkcert_compile_audit.runtime_invocations - before_first
        before_second = self._forkcert_compile_audit.runtime_invocations
        alt_second = score(self._forkcert_compiled_model)
        second_invocations = self._forkcert_compile_audit.runtime_invocations - before_second

        for batch_index in range(completion_ids.size(0)):
            prompt_ids = inputs["prompt_ids"]
            prompt_mask = inputs["prompt_mask"]
            prompt_tokens = prompt_ids[batch_index][prompt_mask[batch_index].bool()].tolist()
            valid = completion_mask[batch_index].bool()
            response_tokens = completion_ids[batch_index][valid].tolist()
            cid = case_id(rollout_batch, prompt_tokens, response_tokens)
            advantage = float(advantages[batch_index].item())
            for token_index, token_id in enumerate(response_tokens):
                logp_ref = float(ref_first[batch_index, token_index].item())
                logp_alt = float(alt_first[batch_index, token_index].item())
                row = {
                    "case_id": cid,
                    "token_index": token_index,
                    "token_id": int(token_id),
                    "token_text": self.processing_class.decode([token_id], skip_special_tokens=False),
                    "path_ref": f"hf-eager-{compute_dtype}-sdpa-math-online",
                    "path_alt": f"hf-compile-{compute_dtype}-sdpa-math-online",
                    "training_compute_dtype": compute_dtype,
                    "logp_ref": logp_ref,
                    "logp_alt": logp_alt,
                    "logprob_delta": abs(logp_alt - logp_ref),
                    "delta_self_ref": abs(float(ref_second[batch_index, token_index].item()) - logp_ref),
                    "delta_self_alt": abs(float(alt_second[batch_index, token_index].item()) - logp_alt),
                    "old_logp": float(old_logps[batch_index, token_index].item()),
                    "advantage": advantage,
                    "advantage_sign": 1 if advantage > 0 else -1 if advantage < 0 else 0,
                    "optimizer_step": int(self.state.global_step),
                    "rollout_batch": rollout_batch,
                    "policy_iteration": policy_iteration,
                    "state": "pre_minibatch",
                    "online_state_aligned": True,
                    "attention_backend_locked": "MATH",
                    "compile_warmup_calls_discarded_per_batch": 1,
                    "compiled_warmup_runtime_invocations": warm_invocations,
                    "compiled_first_runtime_invocations": first_invocations,
                    "compiled_second_runtime_invocations": second_invocations,
                    "candidate_identity_valid": first_invocations > 0 and second_invocations > 0,
                }
                self._forkcert_online_scan_fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._forkcert_online_scan_fh.flush()

    def _scan_grad_compile_pair(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
        input_ids: Any,
        attention_mask: Any,
        old_logps: Any,
        advantages: Any,
        rollout_batch: int,
        policy_iteration: int,
    ) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if not torch.is_grad_enabled():
            raise RuntimeError("grad-enabled compile scan entered with autograd disabled")
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        compute_dtype = trainer_compute_dtype(self.args)
        rng_before = rng_snapshot()
        gradients_before = gradient_signature(model)
        versions_before = tensor_version_signature(model)
        global_step_before = int(self.state.global_step)
        internal_step_before = int(self._step)

        def score(target_model: Any) -> tuple[Any, bool]:
            # Every paired and repeated call starts from the same RNG state.
            # The original state is restored after the whole probe so the
            # surrounding training trajectory does not consume probe RNG.
            restore_rng(rng_before)
            with sdpa_kernel(SDPBackend.MATH):
                values, _, _ = self._get_per_token_logps_and_entropies(
                    target_model,
                    input_ids,
                    attention_mask,
                    completion_ids.size(1),
                    compute_entropy=False,
                )
            requires_grad = bool(values.requires_grad)
            detached = values.detach().clone()
            del values
            return detached, requires_grad

        ref_first, ref_first_grad = score(model)
        ref_second, ref_second_grad = score(model)
        if self._forkcert_grad_compiled_model is None:
            self._forkcert_grad_compiled_model = torch.compile(
                model, backend=make_tracking_backend(self._forkcert_grad_compile_audit)
            )
        before_warm = self._forkcert_grad_compile_audit.runtime_invocations
        warm, warm_grad = score(self._forkcert_grad_compiled_model)
        warm_invocations = self._forkcert_grad_compile_audit.runtime_invocations - before_warm
        del warm
        before_first = self._forkcert_grad_compile_audit.runtime_invocations
        alt_first, alt_first_grad = score(self._forkcert_grad_compiled_model)
        first_invocations = self._forkcert_grad_compile_audit.runtime_invocations - before_first
        before_second = self._forkcert_grad_compile_audit.runtime_invocations
        alt_second, alt_second_grad = score(self._forkcert_grad_compiled_model)
        second_invocations = self._forkcert_grad_compile_audit.runtime_invocations - before_second
        rng_after_measurement = rng_snapshot()
        restore_rng(rng_before)
        rng_after_restore = rng_snapshot()

        gradients_after = gradient_signature(model)
        versions_after = tensor_version_signature(model)
        state_id = f"step-{int(self.state.global_step)}-rollout-{rollout_batch}-iteration-{policy_iteration}"
        state_record = {
            "state_id": state_id,
            "optimizer_step": int(self.state.global_step),
            "rollout_batch": rollout_batch,
            "policy_iteration": policy_iteration,
            "batch_size": int(completion_ids.size(0)),
            "completion_length": int(completion_ids.size(1)),
            "training_compute_dtype": compute_dtype,
            "autograd_enabled": True,
            "all_outputs_require_grad": all(
                (ref_first_grad, ref_second_grad, warm_grad, alt_first_grad, alt_second_grad)
            ),
            "accelerate_native_amp": bool(self.accelerator.native_amp),
            "accelerate_mixed_precision": str(self.accelerator.mixed_precision),
            "accelerate_forward_wrapped": bool(
                hasattr(model, "_original_forward") and hasattr(model.forward, "__wrapped__")
            ),
            "attention_backend_locked": "MATH",
            "ref_first_sha256": tensor_sha256(ref_first),
            "ref_second_sha256": tensor_sha256(ref_second),
            "alt_first_sha256": tensor_sha256(alt_first),
            "alt_second_sha256": tensor_sha256(alt_second),
            "ref_self_max_abs": float((ref_second.float() - ref_first.float()).abs().max()),
            "alt_self_max_abs": float((alt_second.float() - alt_first.float()).abs().max()),
            "compile_audit": {
                "backend_compiles_so_far": self._forkcert_grad_compile_audit.backend_compiles,
                "runtime_invocations_so_far": self._forkcert_grad_compile_audit.runtime_invocations,
                "graph_code_sha256_so_far": list(self._forkcert_grad_compile_audit.graph_code_sha256),
                "graph_node_counts_so_far": list(self._forkcert_grad_compile_audit.graph_node_counts),
                "warmup_runtime_invocations": warm_invocations,
                "first_runtime_invocations": first_invocations,
                "second_runtime_invocations": second_invocations,
            },
            "candidate_identity_valid": first_invocations > 0 and second_invocations > 0,
            "gradient_signature_before": gradients_before,
            "gradient_signature_after": gradients_after,
            "gradients_preserved": gradients_before == gradients_after,
            "tensor_versions_before": versions_before,
            "tensor_versions_after": versions_after,
            "tensor_versions_preserved": versions_before == versions_after,
            "trainer_steps_preserved": global_step_before == int(self.state.global_step)
            and internal_step_before == int(self._step),
            "rng_equal_after_measurement": rng_equal(rng_before, rng_after_measurement),
            "rng_restored_exactly": rng_equal(rng_before, rng_after_restore),
        }
        self._forkcert_grad_state_fh.write(
            json.dumps(state_record, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._forkcert_grad_state_fh.flush()

        for batch_index in range(completion_ids.size(0)):
            prompt_ids = inputs["prompt_ids"]
            prompt_mask = inputs["prompt_mask"]
            prompt_tokens = prompt_ids[batch_index][prompt_mask[batch_index].bool()].tolist()
            valid = completion_mask[batch_index].bool()
            response_tokens = completion_ids[batch_index][valid].tolist()
            cid = case_id(rollout_batch, prompt_tokens, response_tokens)
            advantage = float(advantages[batch_index].item())
            for token_index, token_id in enumerate(response_tokens):
                row = {
                    "state_id": state_id,
                    "case_id": cid,
                    "batch_index": batch_index,
                    "flat_index": batch_index * int(completion_ids.size(1)) + token_index,
                    "token_index": token_index,
                    "token_id": int(token_id),
                    "token_text": self.processing_class.decode([token_id], skip_special_tokens=False),
                    "logp_ref_first": float(ref_first[batch_index, token_index]),
                    "logp_ref_second": float(ref_second[batch_index, token_index]),
                    "logp_alt_first": float(alt_first[batch_index, token_index]),
                    "logp_alt_second": float(alt_second[batch_index, token_index]),
                    "old_logp": float(old_logps[batch_index, token_index]),
                    "advantage": advantage,
                    "advantage_sign": 1 if advantage > 0 else -1 if advantage < 0 else 0,
                    "optimizer_step": int(self.state.global_step),
                    "rollout_batch": rollout_batch,
                    "policy_iteration": policy_iteration,
                    "state": "pre_minibatch",
                    "path_ref": f"hf-eager-{compute_dtype}-sdpa-math-grad-enabled",
                    "path_alt": f"hf-compile-{compute_dtype}-sdpa-math-grad-enabled",
                }
                self._forkcert_grad_scan_fh.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        self._forkcert_grad_scan_fh.flush()
        del ref_first, ref_second, alt_first, alt_second

    @torch.no_grad()
    def _scan_logsoftmax_input_pair(
        self,
        *,
        model: Any,
        inputs: dict[str, Any],
        input_ids: Any,
        attention_mask: Any,
        current_logps: Any,
        old_logps: Any,
        advantages: Any,
        rollout_batch: int,
        policy_iteration: int,
    ) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from trl.trainer.utils import selective_log_softmax

        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        logits_to_keep = completion_ids.size(1)

        def score_pair():
            model_inputs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            with sdpa_kernel(SDPBackend.MATH):
                logits = model(**model_inputs).logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits.div_(self.temperature)
            half_input = selective_log_softmax(logits, completion_ids)
            float_input = selective_log_softmax(logits.float(), completion_ids)
            return half_input.detach(), float_input.detach(), str(logits.dtype), str(half_input.dtype), str(float_input.dtype)

        ref_first, alt_first, logits_dtype, ref_output_dtype, alt_output_dtype = score_pair()
        ref_second, alt_second, *_ = score_pair()
        training_match_max = float((ref_first.float() - current_logps.float()).abs().max().item())
        if training_match_max > 2e-6:
            raise ValueError(
                "online log-softmax half-input path does not match Trainer current_logps: "
                f"max_abs_error={training_match_max}"
            )

        for batch_index in range(completion_ids.size(0)):
            prompt_ids = inputs["prompt_ids"]
            prompt_mask = inputs["prompt_mask"]
            prompt_tokens = prompt_ids[batch_index][prompt_mask[batch_index].bool()].tolist()
            valid = completion_mask[batch_index].bool()
            response_tokens = completion_ids[batch_index][valid].tolist()
            cid = case_id(rollout_batch, prompt_tokens, response_tokens)
            advantage = float(advantages[batch_index].item())
            for token_index, token_id in enumerate(response_tokens):
                logp_ref = float(ref_first[batch_index, token_index].item())
                logp_alt = float(alt_first[batch_index, token_index].item())
                row = {
                    "case_id": cid,
                    "token_index": token_index,
                    "token_id": int(token_id),
                    "token_text": self.processing_class.decode([token_id], skip_special_tokens=False),
                    "path_ref": "hf-half-input-logsoftmax-autocast-online",
                    "path_alt": "hf-explicit-float-input-logsoftmax-online",
                    "logp_ref": logp_ref,
                    "logp_alt": logp_alt,
                    "logprob_delta": abs(logp_alt - logp_ref),
                    "delta_self_ref": abs(float(ref_second[batch_index, token_index].item()) - logp_ref),
                    "delta_self_alt": abs(float(alt_second[batch_index, token_index].item()) - logp_alt),
                    "old_logp": float(old_logps[batch_index, token_index].item()),
                    "advantage": advantage,
                    "advantage_sign": 1 if advantage > 0 else -1 if advantage < 0 else 0,
                    "optimizer_step": int(self.state.global_step),
                    "rollout_batch": rollout_batch,
                    "policy_iteration": policy_iteration,
                    "state": "pre_minibatch",
                    "online_state_aligned": True,
                    "attention_backend_locked": "MATH",
                    "shared_logits_within_each_score_pair": True,
                    "training_logp_match_max": training_match_max,
                    "logits_dtype": logits_dtype,
                    "ref_output_dtype": ref_output_dtype,
                    "alt_output_dtype": alt_output_dtype,
                }
                self._forkcert_logsoftmax_scan_fh.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        self._forkcert_logsoftmax_scan_fh.flush()

    def write_forkcert_artifacts(self, final_rollout_path: Path) -> None:
        self._dump_fh.flush()
        self._dump_fh.close()
        self._samples_fh.flush()
        self._samples_fh.close()
        if self._forkcert_online_scan_fh is not None:
            self._forkcert_online_scan_fh.flush()
            self._forkcert_online_scan_fh.close()
        if self._forkcert_logsoftmax_scan_fh is not None:
            self._forkcert_logsoftmax_scan_fh.flush()
            self._forkcert_logsoftmax_scan_fh.close()
        if self._forkcert_grad_scan_fh is not None:
            self._forkcert_grad_scan_fh.flush()
            self._forkcert_grad_scan_fh.close()
        if self._forkcert_grad_state_fh is not None:
            self._forkcert_grad_state_fh.flush()
            self._forkcert_grad_state_fh.close()
        samples_by_id = {str(row["case_id"]): row for row in read_jsonl(self.samples_path)}
        write_jsonl(self.samples_path, samples_by_id.values())

        rows = []
        with self.dump_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        latest: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["case_id"]), int(row["token_index"]))
            if key not in latest or int(row["policy_iteration"]) >= int(latest[key]["policy_iteration"]):
                latest[key] = row
        final_rows = []
        for row in latest.values():
            item = dict(row)
            item["state"] = "final"
            final_rows.append(item)
        write_jsonl(final_rollout_path, final_rows)


def builtin_arithmetic_dataset(count: int) -> Dataset:
    rows = []
    for index in range(count):
        start = 7 + index
        added = 3 + (index % 11)
        removed = 1 + (index % 5)
        result = start + added - removed
        rows.append(
            {
                "question": f"A box starts with {start} items, receives {added}, then gives away {removed}. How many remain?",
                "answer": f"Compute {start} + {added} - {removed}. #### {result}",
            }
        )
    return Dataset.from_list(rows)


def prepare_dataset(cfg: dict[str, Any]) -> tuple[Any, str]:
    dataset_cfg = cfg["dataset"]
    offset = int(dataset_cfg.get("offset", 0))
    max_prompts = int(dataset_cfg.get("max_prompts", 64))
    if offset < 0 or max_prompts <= 0:
        raise ValueError("dataset offset must be non-negative and max_prompts must be positive")
    source = f"{dataset_cfg['name']}:{dataset_cfg.get('config')}:{dataset_cfg.get('split', 'train')}"
    if dataset_cfg["name"] == "forkcert_builtin_arithmetic":
        dataset = builtin_arithmetic_dataset(offset + max_prompts)
        source = "forkcert_builtin_arithmetic"
    else:
        try:
            dataset = load_dataset(dataset_cfg["name"], dataset_cfg.get("config"), split=dataset_cfg.get("split", "train"))
        except Exception:
            if not dataset_cfg.get("fallback_builtin", False):
                raise
            dataset = builtin_arithmetic_dataset(offset + max_prompts)
            source = "forkcert_builtin_arithmetic_fallback"
    stop = min(offset + max_prompts, len(dataset))
    if offset >= stop:
        raise ValueError(f"dataset slice [{offset}:{offset + max_prompts}] is empty for {len(dataset)} rows")
    dataset = dataset.select(range(offset, stop))
    source = f"{source}[{offset}:{stop}]"

    def convert(row: dict[str, Any]) -> dict[str, str]:
        return {
            "prompt": "Solve the problem. Show concise reasoning and end with the numeric answer.\n\n" + row["question"],
            "answer": row["answer"],
        }

    return dataset.map(convert, remove_columns=dataset.column_names), source


def latest_checkpoint(output_dir: Path) -> tuple[Path | None, int]:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, path))
    if not candidates:
        return None, 0
    step, path = max(candidates)
    state_path = path / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        step = int(state.get("global_step", step))
    return path, step


def checkpoint_step(path: Path) -> int:
    state_path = path / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state["global_step"])
    return int(path.name.rsplit("-", 1)[1])


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_transition_capture_plan(path: Path) -> list[dict[str, Any]]:
    plan_path = path.resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "forkcert.multi-transition-capture-plan.v0.1":
        raise ValueError("unsupported multi-transition capture plan schema")
    root = Path(payload["capture_root"])
    if not root.is_absolute():
        root = (plan_path.parent / root).resolve()
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("transition capture plan targets must be a non-empty list")
    plan_digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_identity = payload.get("identity") or {}
    if not isinstance(plan_identity, dict):
        raise ValueError("transition capture plan identity must be an object")
    result: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        step = int(target["optimizer_step"])
        if step < 0:
            raise ValueError(f"capture target {index} has a negative optimizer step")
        state_id = str(target["state_id"])
        relative_dir = Path(target.get("relative_dir", f"step_{step:06d}"))
        if relative_dir.is_absolute() or ".." in relative_dir.parts:
            raise ValueError(f"capture target {index} relative_dir escapes capture_root")
        result.append(
            {
                **plan_identity,
                "optimizer_step": step,
                "state_id": state_id,
                "capture_dir": root / relative_dir,
                "plan_digest": plan_digest,
                "plan_path": str(plan_path),
                "history_selection": str(
                    target.get("history_selection", "FINAL_POLICY_ITERATION_ONLY")
                ),
                "phase": target.get("phase", plan_identity.get("phase")),
                "eligible_step_population": target.get(
                    "eligible_step_population",
                    plan_identity.get("eligible_step_population"),
                ),
            }
        )
        if result[-1]["history_selection"] not in {
            "FINAL_POLICY_ITERATION_ONLY",
            "EVERY_OPTIMIZER_PRE_STEP",
        }:
            raise ValueError(
                f"capture target {index} has unsupported history_selection"
            )
    steps = [row["optimizer_step"] for row in result]
    state_ids = [row["state_id"] for row in result]
    directories = [str(row["capture_dir"]) for row in result]
    if len(steps) != len(set(steps)):
        raise ValueError("transition capture plan optimizer steps must be unique")
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("transition capture plan state IDs must be unique")
    if len(directories) != len(set(directories)):
        raise ValueError("transition capture plan directories must be unique")
    return sorted(result, key=lambda row: row["optimizer_step"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run instrumented TRL GRPO for the canonical Phase 0 margin experiment.")
    parser.add_argument("--config", default="configs/phase0_grpo.example.yaml")
    parser.add_argument("--out-jsonl", default="data/phase0_grpo_dump.jsonl")
    parser.add_argument("--samples-jsonl", default="data/phase0_grpo_samples.jsonl")
    parser.add_argument("--final-rollout-jsonl", default="data/phase0_final_rollout.jsonl")
    parser.add_argument("--output-dir", default="data/phase0_policy_final")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--snapshot-step", type=int)
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--online-compile-scan-jsonl")
    parser.add_argument("--online-logsoftmax-scan-jsonl")
    parser.add_argument("--online-grad-compile-scan-jsonl")
    parser.add_argument("--online-grad-compile-state-jsonl")
    parser.add_argument("--transition-capture-step", type=int)
    parser.add_argument("--transition-capture-dir")
    parser.add_argument("--transition-capture-plan")
    parser.add_argument("--stop-after-snapshot", action="store_true")
    args = parser.parse_args()

    if (args.transition_capture_step is None) != (args.transition_capture_dir is None):
        raise ValueError("transition-capture-step and transition-capture-dir must be supplied together")
    if args.transition_capture_plan and args.transition_capture_step is not None:
        raise ValueError("transition-capture-plan and legacy single-step capture are exclusive")
    transition_capture_targets = (
        load_transition_capture_plan(Path(args.transition_capture_plan))
        if args.transition_capture_plan
        else None
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    if args.online_compile_scan_jsonl or args.online_grad_compile_scan_jsonl:
        # The confirmation trajectory intentionally exposes multiple static
        # sequence shapes. The default limit of eight silently falls back to
        # eager after the eighth specialization, which invalidates candidate
        # identity rather than constituting a zero-disagreement result.
        torch._dynamo.config.recompile_limit = max(
            int(torch._dynamo.config.recompile_limit), 64
        )

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    seed = int(train_cfg.get("seed", 0))
    model_dtype = str(model_cfg.get("dtype", "bfloat16")).lower()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Phase 0 requires exactly one visible CUDA device, got {torch.cuda.device_count()}; "
            "set CUDA_VISIBLE_DEVICES to one physical GPU to prevent implicit DataParallel"
        )
    use_bf16 = model_dtype in {"bf16", "bfloat16"}
    use_fp16 = model_dtype in {"fp16", "float16"}
    model_parameter_dtype = "bfloat16" if use_bf16 else "float32"
    if not (use_bf16 or use_fp16):
        raise ValueError(f"Phase 0 training dtype must be bf16 or fp16, got {model_dtype}")
    if use_bf16 and (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8):
        raise ValueError("BF16 Phase 0 requires an Ampere-or-newer GPU; use the resolved FP16 runtime config on T4")
    num_iterations = int(train_cfg.get("num_iterations", 3))
    save_steps = int(train_cfg.get("save_steps", 30))
    if save_steps % num_iterations != 0:
        raise ValueError(
            f"save_steps={save_steps} must be divisible by num_iterations={num_iterations}; "
            "otherwise resume would regenerate a rollout in the middle of its policy-iteration group"
        )
    torch.manual_seed(seed)
    dataset, dataset_source = prepare_dataset(cfg)
    processing_class = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
    )
    if processing_class.pad_token is None and processing_class.eos_token is not None:
        processing_class.pad_token = processing_class.eos_token
    output_dir = Path(args.output_dir)
    explicit_resume = Path(args.resume_from_checkpoint).resolve() if args.resume_from_checkpoint else None
    if explicit_resume is not None:
        if not explicit_resume.exists():
            raise FileNotFoundError(explicit_resume)
        resume_checkpoint = explicit_resume
        resume_step = checkpoint_step(explicit_resume)
    else:
        resume_checkpoint, resume_step = latest_checkpoint(output_dir)
    append_artifacts = (
        explicit_resume is None
        and resume_checkpoint is not None
        and Path(args.out_jsonl).exists()
        and Path(args.samples_jsonl).exists()
    )
    if explicit_resume is None and resume_checkpoint is not None and not append_artifacts:
        # A checkpoint without its aligned instrumentation cannot prove Phase 0 coverage.
        resume_checkpoint, resume_step = None, 0
    if resume_checkpoint is not None and resume_step % num_iterations != 0:
        raise ValueError(
            f"checkpoint step {resume_step} is not aligned to num_iterations={num_iterations}; "
            "refuse to mix a regenerated rollout with a partial old-logprob group"
        )
    grpo_args = GRPOConfig(
        output_dir=str(output_dir),
        model_init_kwargs={
            # FP16 training must keep FP32 master parameters; Accelerate then
            # applies autocast + GradScaler. Loading parameters directly as
            # FP16 makes GradScaler.unscale_ reject the gradients.
            "dtype": model_parameter_dtype,
            "attn_implementation": model_cfg.get("attention_backend", "sdpa"),
        },
        max_steps=int(train_cfg.get("max_steps", 300)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=1,
        num_generations=int(train_cfg.get("num_generations", 4)),
        num_iterations=num_iterations,
        max_completion_length=int(train_cfg.get("max_completion_length", 128)),
        generation_kwargs=dict(train_cfg.get("generation_kwargs") or {}),
        epsilon=float(train_cfg.get("epsilon", 0.2)),
        loss_type="grpo",
        importance_sampling_level="token",
        beta=0.0,
        temperature=1.0,
        use_vllm=False,
        disable_dropout=True,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=use_fp16,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = InstrumentedGRPOTrainer(
            model=model_cfg["model_name_or_path"],
            reward_funcs=numeric_reward,
            args=grpo_args,
            train_dataset=dataset,
            processing_class=processing_class,
            dump_path=Path(args.out_jsonl),
            samples_path=Path(args.samples_jsonl),
            append=append_artifacts,
            snapshot_step=args.snapshot_step,
            snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
            online_compile_scan_path=(
                Path(args.online_compile_scan_jsonl) if args.online_compile_scan_jsonl else None
            ),
            online_logsoftmax_scan_path=(
                Path(args.online_logsoftmax_scan_jsonl) if args.online_logsoftmax_scan_jsonl else None
            ),
            online_grad_compile_scan_path=(
                Path(args.online_grad_compile_scan_jsonl)
                if args.online_grad_compile_scan_jsonl
                else None
            ),
            online_grad_compile_state_path=(
                Path(args.online_grad_compile_state_jsonl)
                if args.online_grad_compile_state_jsonl
                else None
            ),
            transition_capture_step=args.transition_capture_step,
            transition_capture_dir=(
                Path(args.transition_capture_dir) if args.transition_capture_dir else None
            ),
            transition_capture_targets=transition_capture_targets,
            stop_after_snapshot=args.stop_after_snapshot,
        )
        trainer._step = resume_step
        trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
        trainer.save_model(str(output_dir))
        trainer.write_forkcert_artifacts(Path(args.final_rollout_jsonl))
    warning_messages = sorted({str(item.message) for item in caught})
    metadata = {
        "training_kind": "trl_grpo",
        "advantage_source": "trl_group_normalized_rewards",
        "old_logp_source": "trl_old_per_token_logps",
        "dataset_source": dataset_source,
        "config": cfg,
        "training_compute_dtype": "bf16" if use_bf16 else "fp16",
        "model_parameter_dtype": model_parameter_dtype,
        "trl_version": __import__("trl").__version__,
        "torch_version": torch.__version__,
        "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_step": resume_step,
        "snapshot_step": args.snapshot_step,
        "snapshot_dir": args.snapshot_dir,
        "transition_capture_step": args.transition_capture_step,
        "transition_capture_dir": args.transition_capture_dir,
        "transition_capture_plan": args.transition_capture_plan,
        "transition_capture_targets": (
            [
                {
                    "optimizer_step": row["optimizer_step"],
                    "state_id": row["state_id"],
                    "capture_dir": str(row["capture_dir"]),
                    "plan_digest": row["plan_digest"],
                    "history_selection": row["history_selection"],
                }
                for row in transition_capture_targets
            ]
            if transition_capture_targets is not None
            else None
        ),
        "online_compile_scan_jsonl": args.online_compile_scan_jsonl,
        "online_compile_scan_protocol": (
            "policy_iteration=2 pre-minibatch; eager MATH twice; tracked compiled MATH warm-up once then measured twice"
            if args.online_compile_scan_jsonl
            else None
        ),
        "online_logsoftmax_scan_jsonl": args.online_logsoftmax_scan_jsonl,
        "online_logsoftmax_scan_protocol": (
            "policy_iteration=2 pre-minibatch; shared logits per call; half-input autocast and explicit-float-input "
            "selective log-softmax; both paths measured twice"
            if args.online_logsoftmax_scan_jsonl
            else None
        ),
        "online_grad_compile_scan_jsonl": args.online_grad_compile_scan_jsonl,
        "online_grad_compile_state_jsonl": args.online_grad_compile_state_jsonl,
        "online_grad_compile_scan_protocol": (
            "policy_iteration=2 pre-minibatch; autograd explicitly enabled inside the no-grad dump hook; "
            "actual Accelerate-wrapped eager Trainer scorer twice; separately tracked compiled scorer warm-up once "
            "then measured twice; identical RNG restored before every call and after the probe"
            if args.online_grad_compile_scan_jsonl
            else None
        ),
        "stop_after_snapshot": args.stop_after_snapshot,
        "compile_audit": {
            "backend_compiles": trainer._forkcert_compile_audit.backend_compiles,
            "runtime_invocations": trainer._forkcert_compile_audit.runtime_invocations,
            "graph_code_sha256": trainer._forkcert_compile_audit.graph_code_sha256,
            "graph_node_counts": trainer._forkcert_compile_audit.graph_node_counts,
        },
        "grad_compile_audit": {
            "backend_compiles": trainer._forkcert_grad_compile_audit.backend_compiles,
            "runtime_invocations": trainer._forkcert_grad_compile_audit.runtime_invocations,
            "graph_code_sha256": trainer._forkcert_grad_compile_audit.graph_code_sha256,
            "graph_node_counts": trainer._forkcert_grad_compile_audit.graph_node_counts,
        },
        "deterministic_warn_messages": warning_messages,
        "environment": {
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "transformers_version": importlib.metadata.version("transformers"),
            "datasets_version": importlib.metadata.version("datasets"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "dynamo_recompile_limit": int(torch._dynamo.config.recompile_limit),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "git_commit": git_commit(),
        },
    }
    Path(args.out_jsonl).with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
