#!/usr/bin/env python3
"""Same-state warm AdamW replay at every completed language checkpoint.

This is post-confirmation mechanism analysis, not another unseen confirmation.
Frozen weights/moments across probe batches do not constitute continued training.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")

import numpy as np
import torch
from kernel_analyzer.update_write import adamw_parameter_write
from scripts import run_liger_language_pilot as pilot
from scripts.run_liger_language_confirmation import OUT as TRAINING, N
from scripts.run_liger_single_boundary_collapse import file_sha256
from scripts.run_training_numerical_v2 import BASE, ROOT, save_new

OUT = BASE / "language_checkpoint_direct_probe"
SOURCES = ("scripts/probe_liger_language_checkpoints.py", "src/kernel_analyzer/update_write.py",
           "scripts/run_liger_single_boundary_collapse.py", "scripts/run_liger_language_pilot.py")


def source_hashes():
    return {name: file_sha256(ROOT / name) for name in SOURCES}


def freeze():
    verified = json.loads((TRAINING / "execution_verification.json").read_text())
    if verified["status"] != "VERIFIED_RECORDED_EXECUTION":
        raise RuntimeError("Training execution not verified")
    checkpoints = {}
    for pair in range(N):
        for condition in ("candidate", "reference"):
            status = json.loads((TRAINING / f"pair{pair}" / condition / "status.json").read_text())
            checkpoints[f"pair{pair}/{condition}"] = status["final_checkpoint_sha256"]
    save_new(OUT / "plan.json", {"schema": "liger-language-checkpoint-direct-probe-v1",
        "data_use": "POST_CONFIRMATION_MECHANISM_ANALYSIS_ALL_PAIRS_NO_OUTCOME_SELECTION",
        "pairs": N, "batches_per_checkpoint": 32, "training_step_indices": list(range(1024, 1056)),
        "state_policy": "Restore the actual final master weights and moments; keep both fixed across probe batches",
        "parameter_scope": "transformer.wte.weight (tied language head); other parameters not certified",
        "contrast": "Candidate minus reference at each identical checkpoint and batch",
        "statistics": "Original-coordinate FP64 energies, mean-vector energies, adjacent and split-half inner products; fixed-suite descriptive only",
        "loss_relation": "Link by pair ID to the existing loss endpoint; no new causal sufficiency or persistence verdict",
        "checkpoint_sha256": checkpoints, "sources": source_hashes(),
        "training_summary_sha256": file_sha256(TRAINING / "summary.json"),
        "no_threshold_selection": True})


class Moments:
    """Streaming full-coordinate summaries, without a sketch or chosen direction."""
    def __init__(self):
        self.n = 0
        self.total = self.left = self.right = self.previous = None
        self.energy = self.repair_energy = self.aligned = self.lag = 0.0

    def add(self, effect, reference):
        u, r = effect.double(), reference.double()
        if not torch.isfinite(u).all() or not torch.isfinite(r).all():
            raise ValueError("Nonfinite replay")
        if self.total is None:
            self.total = torch.zeros_like(u)
            self.left = torch.zeros_like(u)
            self.right = torch.zeros_like(u)
        self.total.add_(u)
        (self.left if self.n < 16 else self.right).add_(u)
        if self.previous is not None:
            self.lag += float((self.previous * u).sum())
        self.previous = u
        x, b, a = float(u.square().sum()), float(r.square().sum()), float((u*r).sum())
        self.energy += x; self.repair_energy += b; self.aligned += a; self.n += 1
        return {"effect_energy": x, "repair_energy": b, "effect_repair_inner_product": a}

    def result(self):
        b = self.repair_energy
        return {"states": self.n, "effect_energy_sum": self.energy, "repair_energy_sum": b,
            "effect_mean_energy": float(self.total.square().sum()) / self.n**2,
            "mean_relative_to_repair_rms": (float(self.total.square().sum()) / (self.n*b))**0.5 if b else None,
            "total_relative_rms": (self.energy / b)**0.5 if b else None,
            "aligned_ratio_of_sums": self.aligned / b if b else None,
            "lag1_inner_product_sum": self.lag,
            "split_half_mean_inner_product": float((self.left*self.right).sum()) / (16*16),
            "scope": "Fixed 32 probe batches at one frozen checkpoint; not independent training steps or a persistence test"}


def measure(pair, condition, device):
    plan = json.loads((OUT / "plan.json").read_text())
    if source_hashes() != plan["sources"]:
        raise RuntimeError("Frozen probe sources changed")
    original = TRAINING / f"pair{pair}"
    protocol = json.loads((original / "protocol.json").read_text())
    path = original / condition / "final.pt"
    if file_sha256(path) != plan["checkpoint_sha256"][f"pair{pair}/{condition}"]:
        raise RuntimeError("Checkpoint changed")
    for filename, digest in protocol["liger_sources"].items():
        if file_sha256(pilot.PACKAGE / filename) != digest:
            raise RuntimeError("Liger source changed")
    if file_sha256(original / "train.npy") != protocol["data"]["train"]["encoded_sha256"]:
        raise RuntimeError("Training tokens changed")
    output = OUT / f"pair{pair}" / condition
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("liger_kernel", pilot.PACKAGE / "__init__.py",
                                                 submodule_search_locations=[str(pilot.PACKAGE)])
    module = importlib.util.module_from_spec(spec); sys.modules["liger_kernel"] = module
    spec.loader.exec_module(module)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["steps"] != 1024:
        raise RuntimeError("Wrong checkpoint horizon")
    model, target = pilot.build_model(protocol, device)
    names = list(dict(model.named_parameters()))
    if names != list(checkpoint["master"]):
        raise RuntimeError("Optimizer parameter order is not established")
    master = {name: tensor.to(device) for name, tensor in checkpoint["master"].items()}
    pilot.materialize(model, master)
    group = checkpoint["optimizer"]["param_groups"]
    if len(group) != 1 or len(group[0]["params"]) != len(names):
        raise RuntimeError("Unexpected optimizer parameter groups")
    state = checkpoint["optimizer"]["state"][group[0]["params"][names.index(target)]]
    first, second = state["exp_avg"].to(device), state["exp_avg_sq"].to(device)
    opt = protocol["optimizer"]
    for key in ("lr", "eps", "weight_decay", "foreach", "fused"):
        if group[0][key] != opt[key]:
            raise RuntimeError("Optimizer hyperparameters changed")
    if tuple(group[0]["betas"]) != tuple(opt["betas"]):
        raise RuntimeError("Optimizer betas changed")
    losses = pilot.make_loss_modules(protocol, device)
    tokens = np.load(original / "train.npy")
    gradient, update = Moments(), Moments()
    with (output / "states.jsonl").open("x") as log:
        for index in plan["training_step_indices"]:
            x, y, offsets = pilot.batch_from_stream(tokens, batch_size=2, sequence_length=128,
                stream=0, step=index, seed=20260906, repeat_within_batch=False, device=device)
            gs, writes, loss = {}, {}, {}
            for label, key in (("candidate", "CANDIDATE"), ("reference", "REPAIR")):
                loss[label], gs[label] = pilot.backward_pass(model, losses[key], x, y, target_name=target)
                writes[label] = adamw_parameter_write(master[target], gs[label], first=first, second=second,
                    prior_step=int(state["step"]), learning_rate=opt["lr"], beta1=opt["betas"][0],
                    beta2=opt["betas"][1], epsilon=opt["eps"], weight_decay=opt["weight_decay"])
            row = {"probe_step_index": index, "offsets": offsets, "loss": loss,
                "gradient": gradient.add(gs["candidate"]-gs["reference"], gs["reference"]),
                "update": update.add(writes["candidate"]-writes["reference"], writes["reference"])}
            log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
    save_new(output / "summary.json", {"status": "COMPLETE_FIXED_CHECKPOINT_PROBE", "pair": pair,
        "trajectory_state": condition, "checkpoint_sha256": file_sha256(path),
        "plan_sha256": file_sha256(OUT / "plan.json"), "gradient": gradient.result(), "update": update.result(),
        "bias_persistence": "NOT_ASSESSED_FROZEN_CHECKPOINT_NOT_TEMPORAL_TRAJECTORY"})
    print(json.dumps({"pair": pair, "state": condition, "status": "COMPLETE"}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("command", choices=("freeze", "measure"))
    p.add_argument("--pairs", nargs="+", type=int, choices=range(N)); p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.command == "freeze": freeze(); return
    if not args.pairs: p.error("measure requires --pairs")
    for pair in args.pairs:
        for condition in ("candidate", "reference"):
            measure(pair, condition, torch.device(args.device))


if __name__ == "__main__": main()
