#!/usr/bin/env python3
"""Paired q/k-weight trajectory for the exact layer-27 saved-P softmax repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import torch
from torch._inductor.codecache import PyCodeCache

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "archive/round1_code/src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.generated_contrast_observer import _source_identity  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_frozen_candidate_fp32_screen import wrapper_modules  # noqa: E402
from scripts.run_generated_fp32_screen import file_digest, load_model  # noqa: E402
from scripts.run_targeted_full_coordinate import validate_release  # noqa: E402
from scripts.screen_qwen128_softmax_fb import (  # noqa: E402
    ALPHA, FORWARD_SHA, FORWARD_SYMBOL, TARGET_SHA, SYMBOL,
)
from kernel_analyzer.seup import (  # noqa: E402
    SEUPCalibrator,
    SymmetricSEUPEvaluator,
    adamw_update,
    adamw_effective_update_delta,
)
from kernel_analyzer.short_persistence import SharedShortPersistenceScreen  # noqa: E402

CANDIDATE_ID = "qwen_seq128_layer27_attention_softmax_fb"
CARRIERS = (
    "model.layers.27.self_attn.q_proj.weight",
    "model.layers.27.self_attn.k_proj.weight",
)


def state_tokens(state: dict[str, Any]) -> list[int]:
    return state.get("token_ids", state.get("input_ids"))


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def digest(values: dict[str, torch.Tensor]) -> str:
    state = hashlib.sha256()
    for name in CARRIERS:
        state.update(name.encode())
        state.update(values[name].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return state.hexdigest()


def joint_norm(values: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(value * value) for value in values.values()))


def joint_dot(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(torch.sum(left[name] * right[name]) for name in CARRIERS)


class SavedProbabilityRepair:
    """Replace only the closed dS semantic boundary with the true saved-P VJP."""

    def __init__(self, modules: list[Any], mode: str) -> None:
        backward = []
        forward = []
        for module in modules:
            b_kernel = getattr(module, SYMBOL, None)
            if b_kernel is not None and all(id(b_kernel) != id(item) for item in backward):
                backward.append(b_kernel)
            f_kernel = getattr(module, FORWARD_SYMBOL, None)
            if f_kernel is not None and all(id(f_kernel) != id(item) for item in forward):
                forward.append(f_kernel)
        if len(backward) != 1 or len(forward) != 1:
            raise RuntimeError(f"softmax kernel identity drift: F={len(forward)} B={len(backward)}")
        if mode not in ("SHAM", "REPAIR_SAVED_P", "PERMUTE_SAVED_P_RESIDUAL"):
            raise ValueError(mode)
        self.backward = backward[0]
        self.forward = forward[0]
        self.mode = mode
        self.b_had = "run" in vars(self.backward); self.b_previous = vars(self.backward).get("run")
        self.f_had = "run" in vars(self.forward); self.f_previous = vars(self.forward).get("run")
        self.b_original = self.backward.run; self.f_original = self.forward.run
        self.probability: torch.Tensor | None = None
        self.forward_calls = 0; self.backward_calls = 0
        self.changed_coordinates = 0; self.correction_l2 = 0.0
        self.correction_vector: Any | None = None
        self.natural_residual_l2 = 0.0
        self.permuted_residual_l2 = 0.0
        self.delivered_residual_l2 = 0.0
        self.residual_sum = 0.0
        self.permuted_residual_sum = 0.0
        self.head_shift = 1

    def __enter__(self) -> "SavedProbabilityRepair":
        def forward_wrapped(*args: Any, **kwargs: Any) -> Any:
            _, _, source = _source_identity()
            if source != FORWARD_SHA:
                return self.f_original(*args, **kwargs)
            scores, token_ids = args[:2]
            if tuple(scores.shape) != (1, 16, 128, 128):
                raise RuntimeError("forward score shape drift")
            raw = scores.detach().float()
            ids = token_ids.detach().reshape(-1)
            positions = torch.arange(128, device=scores.device)
            valid = ((positions[None, :] <= positions[:, None])
                     & (ids[None, :] == ids[:, None]))
            mask = torch.where(
                valid, torch.zeros((), device=scores.device, dtype=torch.float32),
                torch.full((), -3.3895313892515355e38,
                           device=scores.device, dtype=torch.float32),
            )
            self.probability = torch.softmax(
                raw * ALPHA + mask.reshape(1, 1, 128, 128), dim=-1
            )
            result = self.f_original(*args, **kwargs)
            self.forward_calls += 1
            return result

        def backward_wrapped(*args: Any, **kwargs: Any) -> Any:
            _, _, source = _source_identity()
            if source != TARGET_SHA:
                return self.b_original(*args, **kwargs)
            destination, upstream = args[:2]
            result = self.b_original(*args, **kwargs)
            actual = destination.detach().clone()
            if self.probability is None:
                raise RuntimeError("backward ran without its bound forward probability")
            if self.mode == "SHAM":
                destination.copy_(actual)
            else:
                gradient = upstream.detach().float().reshape(1, 16, 128, 128)
                inner = (gradient * self.probability).sum(dim=-1, keepdim=True)
                repaired = (self.probability * (gradient - inner) * ALPHA).to(destination.dtype)
                if self.mode == "REPAIR_SAVED_P":
                    destination.copy_(repaired)
                else:
                    # A fixed head derangement preserves the complete local dS
                    # residual multiset, support, and L2 norm while breaking its
                    # pairing with the head-specific downstream Q/K transport.
                    # This is a causal intervention, not a candidate kernel.
                    natural_residual = actual.float() - repaired.float()
                    permuted_residual = torch.roll(
                        natural_residual, shifts=self.head_shift, dims=1
                    )
                    delivered = (repaired.float() + permuted_residual).to(
                        destination.dtype
                    )
                    destination.copy_(delivered)
                    delivered_residual = destination.detach().float() - repaired.float()
                    self.natural_residual_l2 = float(
                        torch.linalg.vector_norm(natural_residual).item()
                    )
                    self.permuted_residual_l2 = float(
                        torch.linalg.vector_norm(permuted_residual).item()
                    )
                    self.delivered_residual_l2 = float(
                        torch.linalg.vector_norm(delivered_residual).item()
                    )
                    self.residual_sum = float(natural_residual.double().sum().item())
                    self.permuted_residual_sum = float(
                        permuted_residual.double().sum().item()
                    )
                correction = destination.detach().float() - actual.float()
                self.changed_coordinates = int(torch.count_nonzero(correction).item())
                self.correction_l2 = float(torch.linalg.vector_norm(correction).item())
                # Formation capture consumes this one endpoint residual.  The
                # large tensor is copied to a temporary disk spool by the
                # caller and is never retained in the certificate JSON.
                self.correction_vector = correction.detach().cpu().numpy().reshape(-1).copy()
            self.backward_calls += 1
            return result

        self.forward.run = forward_wrapped
        self.backward.run = backward_wrapped
        return self

    def __exit__(self, *unused: Any) -> None:
        del unused
        if self.f_had: self.forward.run = self.f_previous
        else: delattr(self.forward, "run")
        if self.b_had: self.backward.run = self.b_previous
        else: delattr(self.backward, "run")
        if self.forward_calls != 1 or self.backward_calls != 1:
            raise RuntimeError(
                f"saved-P boundary executed F={self.forward_calls}, B={self.backward_calls}"
            )


def adam_step(master: torch.Tensor, grad: torch.Tensor, first: torch.Tensor,
              second: torch.Tensor, step: int, learning_rate: float) -> None:
    beta1, beta2, epsilon = 0.9, 0.95, 1e-8
    first.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    second.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    master.addcdiv_(
        first / (1.0 - beta1**step),
        (second / (1.0 - beta2**step)).sqrt().add_(epsilon),
        value=-learning_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json",
    )
    parser.add_argument("--seup-output", type=Path)
    parser.add_argument("--geometry-spool", type=Path,
                        help="optional CPU vector spool for exploratory SEUP geometry")
    parser.add_argument(
        "--short-screen-output", type=Path,
        help="optional compact CountSketch output; requires --seup-output and avoids raw vector spooling",
    )
    parser.add_argument("--short-screen-steps", type=int, default=8)
    parser.add_argument("--short-screen-projection-dim", type=int, default=64)
    parser.add_argument("--short-screen-null-draws", type=int, default=2000)
    args = parser.parse_args()
    if not 1 <= args.steps <= 32:
        raise ValueError("steps must be in [1, 32]")
    if args.short_screen_output is not None:
        if args.seup_output is None:
            raise ValueError("--short-screen-output requires --seup-output")
        if not 4 <= args.short_screen_steps <= 16:
            raise ValueError("--short-screen-steps must be in [4, 16]")

    proof_path = ROOT / "results/coverage/cases/qwen128_softmax_fb.json"
    proof = json.loads(proof_path.read_text())
    if proof["candidate_id"] != CANDIDATE_ID or not all(
        proof["concrete_program_proof"].values()
    ):
        raise RuntimeError("softmax F+B proof is incomplete")
    bank_path = ROOT / "results/coverage/qwen_seq128_input_bank.json"
    states = json.loads(bank_path.read_text())["states"]
    release = ROOT / "results/coverage/runtime_releases/qwen_seq128_r1"
    capture = json.loads((release / "capture.json").read_text())
    if file_digest(bank_path) != capture["input"]["input_bank_sha256"]:
        raise RuntimeError("input bank differs from frozen release")

    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model("qwen", Path("/data1/tzh/models/Qwen/Qwen3-1.7B"), device)
    parameters = dict(model.named_parameters())
    targets = {name: parameters[name] for name in CARRIERS}
    start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([state_tokens(states[0])], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True); candidate(warm).backward(); torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[start:])
    if args.seup_output is None:
        validate_release(wrapper_modules(modules), capture)

    initial = {name: value.detach().float().clone() for name, value in targets.items()}
    candidate_master = {name: value.clone() for name, value in initial.items()}
    repair_master = {name: value.clone() for name, value in initial.items()}
    candidate_m = {name: torch.zeros_like(value) for name, value in initial.items()}
    candidate_v = {name: torch.zeros_like(value) for name, value in initial.items()}
    repair_m = {name: torch.zeros_like(value) for name, value in initial.items()}
    repair_v = {name: torch.zeros_like(value) for name, value in initial.items()}

    def gradient(master: dict[str, torch.Tensor], tokens: list[int], mode: str | None):
        with torch.no_grad():
            for name in CARRIERS:
                targets[name].copy_(master[name].to(targets[name].dtype))
        values = torch.tensor([tokens], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        observer = SavedProbabilityRepair(modules, mode) if mode else None
        if observer is None:
            loss = candidate(values); loss.backward()
        else:
            with observer:
                loss = candidate(values); loss.backward()
        torch.cuda.synchronize(device)
        gradients = {}
        for name in CARRIERS:
            if targets[name].grad is None:
                raise RuntimeError(f"missing carrier gradient: {name}")
            gradients[name] = targets[name].grad.detach().float().clone()
            targets[name].grad = None
        summary = None if observer is None else {
            "changed_coordinates": observer.changed_coordinates,
            "correction_l2": observer.correction_l2,
        }
        return loss.detach(), gradients, summary

    if args.seup_output is not None:
        if len(states) < 32:
            raise RuntimeError("SEUP requires 16 calibration and 16 evaluation states")
        calibrator = SEUPCalibrator(16, 0.5)
        calibration_master = {name: value.clone() for name, value in initial.items()}
        calibration_m = {name: torch.zeros_like(value) for name, value in initial.items()}
        calibration_v = {name: torch.zeros_like(value) for name, value in initial.items()}
        calibration_rows = []
        geometry_rows = []
        short_screen = (
            SharedShortPersistenceScreen(
                projection_dim=args.short_screen_projection_dim,
                projection_seed=20260822,
                expected_steps=args.short_screen_steps,
                null_draws=args.short_screen_null_draws,
                prefix_growth_mode="after_warmup",
            )
            if args.short_screen_output is not None else None
        )

        def add_short_screen(phase: str, tree: dict[str, torch.Tensor], step: int) -> None:
            if short_screen is None or step > args.short_screen_steps:
                return
            # The carrier order is fixed by CARRIERS and is part of the input
            # certificate. Stream one parameter block at a time; do not
            # concatenate a multi-billion-coordinate carrier before hashing.
            chunks = (
                tree[name].detach().float().reshape(-1).cpu().numpy()
                for name in CARRIERS
            )
            short_screen.add_chunks(f"{CANDIDATE_ID}::{phase}", chunks)

        def cpu_tree(tree: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: value.detach().float().cpu().clone() for key, value in tree.items()}

        def tree_sub(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: left[key] - right[key] for key in CARRIERS}

        def tree_avg(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {key: (left[key] + right[key]) * 0.5 for key in CARRIERS}

        for index, state in enumerate(states[:16]):
            tokens = state_tokens(state)
            loss_c, grad_c, _ = gradient(calibration_master, tokens, None)
            loss_r, grad_r, boundary = gradient(calibration_master, tokens, "REPAIR_SAVED_P")
            calibration_forward_repair_exact = torch.equal(loss_c, loss_r)
            delta = adamw_effective_update_delta(
                grad_c, grad_r, calibration_m, calibration_v, calibration_master,
                step=index + 1, learning_rate=args.learning_rate,
                betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0,
            )
            calibrator.add(str(state.get("sequence_id", index)), delta)
            if args.geometry_spool is not None:
                zero = {name: torch.zeros_like(delta[name]) for name in CARRIERS}
                geometry_rows.append({
                    "phase": "calibration",
                    "step": index + 1,
                    "state_id": str(state.get("sequence_id", index)),
                    "local": cpu_tree(delta),
                    "local_candidate": cpu_tree(delta),
                    "feedback": cpu_tree(zero),
                    "actual": cpu_tree(delta),
                    "gradient_delta": cpu_tree({name: grad_c[name] - grad_r[name] for name in CARRIERS}),
                    "effective_update": cpu_tree(delta),
                })
            add_short_screen("calibration", delta, index + 1)
            for name in CARRIERS:
                adam_step(calibration_master[name], grad_c[name], calibration_m[name], calibration_v[name], index + 1, args.learning_rate)
            calibration_rows.append({"step": index + 1, "state_id": str(state.get("sequence_id", index)),
                                     "repair_changed_coordinates": boundary["changed_coordinates"],
                                     "forward_repair_exact": bool(calibration_forward_repair_exact)})
            del grad_c, grad_r, delta
            torch.cuda.empty_cache()
        carrier = calibrator.freeze()
        evaluator = SymmetricSEUPEvaluator(carrier, 16)
        candidate_master = {name: value.clone() for name, value in initial.items()}
        repair_master = {name: value.clone() for name, value in initial.items()}
        candidate_m = {name: torch.zeros_like(value) for name, value in initial.items()}
        candidate_v = {name: torch.zeros_like(value) for name, value in initial.items()}
        repair_m = {name: torch.zeros_like(value) for name, value in initial.items()}
        repair_v = {name: torch.zeros_like(value) for name, value in initial.items()}
        rows = []
        for offset, state in enumerate(states[16:32]):
            index = offset + 1
            tokens = state_tokens(state)
            loss_cc, grad_cc, _ = gradient(candidate_master, tokens, None)
            loss_cr, grad_cr, boundary_cr = gradient(candidate_master, tokens, "REPAIR_SAVED_P")
            loss_rc, grad_rc, _ = gradient(repair_master, tokens, None)
            loss_rr, grad_rr, boundary_rr = gradient(repair_master, tokens, "REPAIR_SAVED_P")
            candidate_forward_repair_exact = torch.equal(loss_cc, loss_cr)
            repair_forward_repair_exact = torch.equal(loss_rc, loss_rr)
            candidate_before = {name: candidate_master[name].clone() for name in CARRIERS}
            repair_before = {name: repair_master[name].clone() for name in CARRIERS}
            uc_sc_planned = adamw_update(grad_cc, candidate_m, candidate_v, candidate_master, step=index, learning_rate=args.learning_rate,
                                         betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0)
            ur_sc = adamw_update(grad_cr, candidate_m, candidate_v, candidate_master, step=index, learning_rate=args.learning_rate,
                                 betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0)
            uc_sr = adamw_update(grad_rc, repair_m, repair_v, repair_master, step=index, learning_rate=args.learning_rate,
                                 betas=(0.9, 0.95), epsilon=1e-8, weight_decay=0.0)
            for name in CARRIERS:
                adam_step(candidate_master[name], grad_cc[name], candidate_m[name], candidate_v[name], index, args.learning_rate)
                adam_step(repair_master[name], grad_rr[name], repair_m[name], repair_v[name], index, args.learning_rate)
            uc_sc = {name: candidate_master[name] - candidate_before[name] for name in CARRIERS}
            ur_sr = {name: repair_master[name] - repair_before[name] for name in CARRIERS}
            before = {name: candidate_before[name] - repair_before[name] for name in CARRIERS}
            after = {name: candidate_master[name] - repair_master[name] for name in CARRIERS}
            evaluator.add(str(state.get("sequence_id", 16 + offset)), uc_sc, ur_sc, uc_sr, ur_sr, before, after,
                           endpoint_repair_nonzero=bool(boundary_cr["changed_coordinates"] > 0 and boundary_rr["changed_coordinates"] > 0))
            local_candidate = tree_sub(uc_sc, ur_sc)
            if args.geometry_spool is not None:
                local_symmetric = tree_avg(tree_sub(uc_sc, ur_sc), tree_sub(uc_sr, ur_sr))
                feedback_symmetric = tree_avg(tree_sub(uc_sc, uc_sr), tree_sub(ur_sc, ur_sr))
                geometry_rows.append({
                    "phase": "evaluation",
                    "step": 16 + index,
                    "state_id": str(state.get("sequence_id", 16 + offset)),
                    "local": cpu_tree(local_symmetric),
                    "local_candidate": cpu_tree(local_candidate),
                    "feedback": cpu_tree(feedback_symmetric),
                    "actual": cpu_tree(after),
                    "gradient_delta": cpu_tree({name: grad_cc[name] - grad_cr[name] for name in CARRIERS}),
                    "effective_update": cpu_tree(local_candidate),
                })
            add_short_screen("evaluation", local_candidate, index)
            rows.append({"step": index, "state_id": str(state.get("sequence_id", 16 + offset)),
                         "forward_repair_exact": bool(candidate_forward_repair_exact),
                         "repair_arm_forward_repair_exact": bool(repair_forward_repair_exact),
                         "candidate_repair_changed_coordinates": boundary_cr["changed_coordinates"],
                         "repair_arm_changed_coordinates": boundary_rr["changed_coordinates"],
                         "drift_l2": float(joint_norm(after).item())})
            print(json.dumps({"event": "SEUP_STEP_COMPLETE", **rows[-1]}), flush=True)
            del grad_cc, grad_cr, grad_rc, grad_rr, uc_sc_planned, uc_sc, ur_sc, uc_sr, ur_sr
            torch.cuda.empty_cache()
        certificate = evaluator.finalize()
        persistence = certificate.get("signed_persistence", 0.0)
        positive = bool(carrier.stable and persistence >= 0.80 and
                        certificate.get("local_fraction_of_projected_accumulation", 0.0) >= 0.50 and
                        certificate.get("local_and_final_carrier_same_sign", False))
        gates = {
            "stable_calibration_carrier": carrier.stable,
            "sixteen_evaluation_steps": len(rows) == 16,
            "endpoint_repair_nonzero_every_step": all(
                row["candidate_repair_changed_coordinates"] > 0 and row["repair_arm_changed_coordinates"] > 0 for row in rows),
            "endpoint_repair_nonzero_any_step": any(
                row["candidate_repair_changed_coordinates"] > 0 or row["repair_arm_changed_coordinates"] > 0 for row in rows),
            "recurrence_closed": certificate["max_recurrence_relative_residual"] <= 1e-6,
            "signed_persistence_ge_0_80": persistence >= 0.80,
            "local_effect_fraction_ge_0_50": certificate.get("local_fraction_of_projected_accumulation", 0.0) >= 0.50,
        }
        payload = {
            "schema": "kernel-analyzer-softmax-saved-p-seup-mainline-v2",
            "status": "PASS_SEUP_POSITIVE" if positive else "PASS_SEUP_NEGATIVE_CONTROL" if all(
                gates[key] for key in ("sixteen_evaluation_steps", "endpoint_repair_nonzero_any_step", "recurrence_closed")
            ) else "MEASURED_WITH_FAILED_GATE",
            "case_id": CANDIDATE_ID,
            "mechanism": "SAVED_PROBABILITY_RECONSTRUCTION_TO_D_S_VJP",
            "mechanism_level": "CLOSED_SEMANTIC_REGION",
            "carrier_parameters": list(CARRIERS),
            "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "betas": [0.9, 0.95], "epsilon": 1e-8},
            "protocol": {"calibration_states": 16, "evaluation_states": 16, "evaluation_state_ids_disjoint": True,
                         "state_feedback_decomposition": "symmetric_candidate_and_repair_state_counterfactuals"},
            "calibration": {"steps": calibration_rows, "carrier": carrier.certificate},
            "evaluation": certificate,
            "steps": rows,
            "gates": gates,
            "claim_boundary": "Closed layer-27 softmax dS semantic region; not a claim that one Triton instruction is uniquely responsible.",
        }
        payload["result_sha256"] = canonical(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        args.seup_output.parent.mkdir(parents=True, exist_ok=True)
        args.seup_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.geometry_spool is not None:
            args.geometry_spool.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "schema": "kernel-analyzer-seup-geometry-spool-v1",
                "case_id": payload["case_id"],
                "task_id": CANDIDATE_ID,
                "carrier_parameters": list(CARRIERS),
                "fields": ["local", "local_candidate", "feedback", "actual", "gradient_delta", "effective_update"],
                "calibration_steps": 16,
                "evaluation_steps": 16,
                "protocol": "carrier-local-feedback-only",
                "rows": geometry_rows,
            }, args.geometry_spool)
        if short_screen is not None:
            short_payload = short_screen.finalize()
            short_payload["input"] = {
                "kind": "LIVE_PAIRED_TRAJECTORY_EFFECTIVE_UPDATE",
                "case_id": CANDIDATE_ID,
                "carrier_parameters": list(CARRIERS),
                "calibration_steps_used": args.short_screen_steps,
                "evaluation_steps_used": args.short_screen_steps,
                "raw_vectors_retained": False,
                "source_result_sha256": payload["result_sha256"],
            }
            args.short_screen_output.parent.mkdir(parents=True, exist_ok=True)
            args.short_screen_output.write_text(
                json.dumps(short_payload, indent=2, sort_keys=True) + "\n"
            )
        print(json.dumps({"event": "SEUP_COMPLETE", "status": payload["status"], "gates": gates}, sort_keys=True))
        return

    first_tokens = state_tokens(states[0])
    baseline_loss, baseline_grad, _ = gradient(initial, first_tokens, None)
    sham_loss, sham_grad, _ = gradient(initial, first_tokens, "SHAM")
    repair_loss, repair_grad, repair_summary = gradient(
        initial, first_tokens, "REPAIR_SAVED_P"
    )
    controls = {
        "matched_sham_exact": torch.equal(baseline_loss, sham_loss)
        and digest(baseline_grad) == digest(sham_grad),
        "forward_loss_unchanged_by_backward_repair": torch.equal(baseline_loss, repair_loss),
        "saved_p_boundary_repair_nonzero": digest(baseline_grad) != digest(repair_grad)
        and bool(repair_summary and repair_summary["changed_coordinates"] > 0),
    }
    del baseline_grad, sham_grad, repair_grad

    records = []
    direction: dict[str, torch.Tensor] | None = None
    for index in range(args.steps):
        tokens = state_tokens(states[index])
        state_id = str(states[index].get("sequence_id", index))
        cand_loss_c, cand_grad_c, _ = gradient(candidate_master, tokens, None)
        _, repair_grad_c, summary_c = gradient(candidate_master, tokens, "REPAIR_SAVED_P")
        _, cand_grad_r, _ = gradient(repair_master, tokens, None)
        repair_loss_r, repair_grad_r, summary_r = gradient(
            repair_master, tokens, "REPAIR_SAVED_P"
        )
        removal_c = {name: cand_grad_c[name] - repair_grad_c[name] for name in CARRIERS}
        removal_r = {name: cand_grad_r[name] - repair_grad_r[name] for name in CARRIERS}
        for name in CARRIERS:
            adam_step(candidate_master[name], cand_grad_c[name], candidate_m[name],
                      candidate_v[name], index + 1, args.learning_rate)
            adam_step(repair_master[name], repair_grad_r[name], repair_m[name],
                      repair_v[name], index + 1, args.learning_rate)
        delta = {name: candidate_master[name] - repair_master[name] for name in CARRIERS}
        if direction is None:
            norm = joint_norm(delta)
            if not bool(norm > 0):
                raise RuntimeError("step-1 joint master divergence is zero")
            direction = {name: value / norm for name, value in delta.items()}
        records.append({
            "step": index + 1, "state_id": state_id,
            "candidate_loss": float(cand_loss_c.cpu()),
            "repair_loss": float(repair_loss_r.cpu()),
            "candidate_removal_l2": float(joint_norm(removal_c).cpu()),
            "repair_removal_l2": float(joint_norm(removal_r).cpu()),
            "candidate_removal_nonzero": bool(joint_norm(removal_c) > 0),
            "repair_removal_nonzero": bool(joint_norm(removal_r) > 0),
            "semantic_boundary_changed_coordinates": summary_c["changed_coordinates"],
            "semantic_boundary_correction_l2": summary_c["correction_l2"],
            "repair_weight_boundary_changed_coordinates": summary_r["changed_coordinates"],
            "joint_master_l2": float(joint_norm(delta).cpu()),
            "joint_master_projection": float(joint_dot(delta, direction).cpu()),
            "bf16_materialized_nonzero": sum(int(torch.count_nonzero(
                candidate_master[name].to(torch.bfloat16)
                != repair_master[name].to(torch.bfloat16)).item()) for name in CARRIERS),
        })
        print(json.dumps({"event": "STEP_COMPLETE", **records[-1]}), flush=True)

    checkpoints = [step for step in (1, 8, 16, 32) if step <= args.steps]
    projections = [records[step - 1]["joint_master_projection"] for step in checkpoints]
    grows = args.steps == 32 and all(
        right > left for left, right in zip(projections, projections[1:])
    )
    gates = {
        **controls,
        "only_declared_qk_parameters_updated": True,
        "paired_same_weight_measurement": True,
        "all_steps_repair_nonzero": all(
            row["candidate_removal_nonzero"] and row["repair_removal_nonzero"]
            for row in records
        ),
        "directional_live_weight_accumulation": grows,
    }
    payload = {
        "schema": "kernel-analyzer-softmax-saved-p-trajectory-v1",
        "status": "PASS_STRICT_SEMANTIC_REGION_FLASH_STYLE_CASE"
        if all(gates.values()) else "COMPLETE_PILOT" if args.steps < 32
        else "FAIL_DIRECTIONAL_ACCUMULATION",
        "candidate_id": CANDIDATE_ID,
        "repair": (
            "At the exact generated dS boundary, replace probability reconstructed from "
            "BF16 logits/max/sum with the analytic VJP of the true FP32 forward probability; "
            "retain the original BF16 dS ABI and all downstream Q/K VJPs."
        ),
        "carrier_parameters": list(CARRIERS),
        "steps": args.steps,
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate,
                      "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.0},
        "initial_controls": controls, "records": records,
        "directional_projection_checkpoints": checkpoints,
        "directional_projections": projections, "gates": gates,
        "bindings": {
            "forward_source_line_sha256": FORWARD_SHA,
            "backward_source_line_sha256": TARGET_SHA,
            "release_capture_sha256": capture["result_sha256"],
            "input_bank_sha256": file_digest(bank_path),
            "proof_result_sha256": proof["result_sha256"],
        },
        "claim_boundary": (
            "This is a causal repair of the closed softmax dS semantic region, not a claim "
            "that one Triton arithmetic instruction is uniquely responsible. Only layer-27 "
            "q_proj/k_proj weights evolve; cross-state generalization remains separately negative."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "TRAJECTORY_COMPLETE", "status": payload["status"],
                      "projections": projections}, sort_keys=True))


if __name__ == "__main__":
    main()
