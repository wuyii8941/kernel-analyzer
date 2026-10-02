#!/usr/bin/env python3
"""Capture the Liger FP32 dW chunk accumulation as Triton launches (step 4 input).

Liger 0.7 accumulates ``grad_weight += torch.mm(grad_logits_chunk.t(), x_chunk).float()``.
The in-place FP32 add is replaced by a Triton kernel performing the same
operation (``scripts.reference_eval_kernels.accumulate``); every unit checks
that the resulting gradient, loss and hidden-state gradient are bitwise equal
to the unmodified torch path.  Each accumulate launch is captured with its
compiled artifacts and the operand values on the declared coordinate set D
(whole lm_head rows).  A hand-written FP64 accumulation of the same
contributions on D is saved as the manual reference.

Runs in an environment with liger_kernel and transformers (the ``liger`` env).
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch  # noqa: E402
from scripts.run_liger_fp32_chunk_order_length import DESIGN, MODEL, _order_values  # noqa: E402

SEED = 20260919  # draw seed of run_liger_fp32_order_population_mean.py
ROW_SEED = 20261002
CALIBRATION_COUNT, CONFIRMATION_COUNT = 32, 64
BLOCK = 2048

ORDER_MARKER = "    for chunk_id in range(num_chunks):"
ACCUM_MARKER = "            grad_weight += torch.mm(grad_logits_chunk.t(), _input_chunk).float()"


def install(fused, source: str, kind: str, accumulate_hook) -> None:
    if source.count(ORDER_MARKER) != 1 or source.count(ACCUM_MARKER) != 1:
        raise RuntimeError("Liger source markers were not uniquely identified")
    transformed = source.replace(ORDER_MARKER, "    for chunk_id in _declared_chunk_order(num_chunks):", 1)
    if accumulate_hook is not None:
        transformed = transformed.replace(
            ACCUM_MARKER,
            "            _declared_accumulate(grad_weight, torch.mm(grad_logits_chunk.t(), _input_chunk).float())", 1)
    namespace = dict(fused.__dict__)
    namespace["_declared_chunk_order"] = lambda count: _order_values(kind, count)
    namespace["_declared_accumulate"] = accumulate_hook
    exec(compile(transformed, f"<liger-{kind}-{'triton' if accumulate_hook else 'torch'}>", "exec"), namespace)
    fused.fused_linear_cross_entropy_forward = namespace["fused_linear_cross_entropy_forward"]


def gradients(module, hidden, weight, labels):
    w = torch.nn.Parameter(weight.detach().clone())
    h = hidden.detach().clone().reshape(-1, hidden.shape[-1]).requires_grad_(True)
    loss = module(w, h, labels)
    grad_hidden, grad_weight = torch.autograd.grad(loss, (h, w), retain_graph=False)
    return loss.detach(), grad_hidden.detach(), grad_weight.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", required=True, help="e.g. 0-7 (indices into the 96 draws)")
    parser.add_argument("--variants", default="original,reverse")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    lo, hi = (int(x) for x in args.units.split("-")) if "-" in args.units else (int(args.units),) * 2
    units = list(range(lo, hi + 1))

    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    import liger_kernel.ops.fused_linear_cross_entropy as fused
    from transformers import AutoModelForCausalLM
    from scripts.reference_eval_kernels import accumulate

    records = [r for r in json.loads(DESIGN.read_text())["records"] if int(r["length"]) == 64]
    draw_indices = np.random.default_rng(SEED).integers(0, len(records), size=CALIBRATION_COUNT + CONFIRMATION_COUNT)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager",
                                                 local_files_only=True).to(device).eval()
    model.config.use_cache = False
    weight0 = model.lm_head.weight.detach().clone()
    V, H = weight0.shape
    if H != BLOCK:
        raise RuntimeError("one program per lm_head row requires BLOCK == hidden size")
    rows = np.sort(np.random.default_rng(ROW_SEED).choice(V, size=args.rows, replace=False))
    window = (rows[:, None] * H + np.arange(H)[None, :]).reshape(-1)
    window_t = torch.as_tensor(window, device=device)
    source = inspect.getsource(fused.fused_linear_cross_entropy_forward)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "design.json").write_text(json.dumps({
        "rows": rows.tolist(), "row_seed": ROW_SEED, "block": BLOCK, "draw_seed": SEED,
        "draw_indices": [int(x) for x in draw_indices], "calibration": CALIBRATION_COUNT,
        "confirmation": CONFIRMATION_COUNT, "model": str(MODEL), "length": 64}) + "\n")

    for unit in units:
        record = records[int(draw_indices[unit])]
        ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.detach()
        labels = torch.nn.functional.pad(ids, (0, 1), value=-100)[..., 1:].contiguous().reshape(-1)
        for kind in args.variants.split(","):
            t0 = time.time()
            install(fused, source, kind, None)
            module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean",
                                                      accum_dtype=torch.float32).to(device)
            loss_t, hidden_t, grad_t = gradients(module, hidden, weight0, labels)

            manual = torch.zeros(window.size, dtype=torch.float64, device=device)
            abs_sum = torch.zeros(window.size, dtype=torch.float64, device=device)
            count = [0]

            def accumulate_hook(acc, contribution):
                flat_c = contribution.reshape(-1)
                picked = flat_c[window_t].double()
                manual.add_(picked)
                abs_sum.add_(picked.abs())
                count[0] += 1
                n = acc.numel()
                accumulate[(triton_cdiv(n, BLOCK),)](acc.view(-1), flat_c, n, BLOCK=BLOCK)

            install(fused, source, kind, accumulate_hook)
            module = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction="mean",
                                                      accum_dtype=torch.float32).to(device)
            recorder = TritonLaunchRecorder(select=lambda name, i: name == "accumulate",
                                            window=lambda kernel, name, t: window)
            with recorder:
                loss_k, hidden_k, grad_k = gradients(module, hidden, weight0, labels)
                torch.cuda.synchronize()
            equal = {"loss": bool(torch.equal(loss_t, loss_k)), "hidden_grad": bool(torch.equal(hidden_t, hidden_k)),
                     "weight_grad": bool(torch.equal(grad_t, grad_k))}
            if not all(equal.values()):
                raise RuntimeError(f"Triton accumulation changed the computation: {equal}")
            unit_dir = args.out / f"unit{unit:03d}" / kind
            for j, launch in enumerate(recorder.launches):
                save_launch(launch, unit_dir / f"launch{j:03d}")
            np.savez(unit_dir / "arrays.npz",
                     grad=grad_k.reshape(-1)[window_t].double().cpu().numpy(),
                     weight=weight0.reshape(-1)[window_t].double().cpu().numpy(),
                     manual=manual.cpu().numpy(), abs_sum=abs_sum.cpu().numpy())
            (unit_dir / "unit.json").write_text(json.dumps({
                "unit": unit, "bank_index": int(draw_indices[unit]), "state_id": str(record["sequence_id"]),
                "variant": kind, "launches": len(recorder.launches), "accumulate_calls": count[0],
                "bitwise_equal_to_torch_path": equal, "capture_seconds": round(time.time() - t0, 2)}) + "\n")
            print(json.dumps({"unit": unit, "variant": kind, "launches": len(recorder.launches),
                              "equal": equal, "seconds": round(time.time() - t0, 1)}), flush=True)
            del recorder, grad_t, grad_k, hidden_t, hidden_k
            torch.cuda.empty_cache()


def triton_cdiv(a, b):
    return (a + b - 1) // b


if __name__ == "__main__":
    main()
