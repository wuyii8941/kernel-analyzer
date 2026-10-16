"""Post-hoc P6 (protocol.md): what one unit costs -- the exact program value G, the emulator (C-exec), the predictor
(C-model), and the kernel itself on the device (recompiled from its source with the captured configuration, run on the
captured inputs, code unchanged).  One JSON line per unit, each unit in its own process with its own Triton cache.

    python cost.py units.jsonl cost.jsonl [--n 100] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SEED = 20261016


def device_times(unit: str) -> dict:
    """First launch (JIT compile included) and a later launch, timed on the device."""
    import hashlib
    import importlib.util

    import torch

    from kernel_analyzer.reference_eval.capture import load_launch
    from patch_replay import _module_text, kernel_file

    L = load_launch(Path(unit))
    text, _ = _module_text(kernel_file(L), L.kernel_name, [])
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    mod_path = Path(os.environ["TRITON_CACHE_DIR"]) / f"{L.kernel_name[:80]}_{digest}.py"
    mod_path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"cost_{digest}", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, L.kernel_name)
    info = json.loads((Path(unit) / "launch.json").read_text())["src_info"]
    constexprs = {info["arg_names"][k[0] if isinstance(k, list) else k]: v for k, v in info["constants"]}
    storages, call = {}, {}
    for arg in L.args:
        if arg.name in constexprs or arg.constexpr:
            continue
        if arg.kind == "tensor":
            if arg.storage_ptr not in storages:
                storages[arg.storage_ptr] = arg.before.to("cuda").clone()
            raw = storages[arg.storage_ptr]
            dtype = getattr(torch, arg.dtype)
            offset = (arg.data_ptr - arg.storage_ptr) // arg.element_size
            flat = torch.empty(0, dtype=dtype, device="cuda").set_(raw.untyped_storage(), 0,
                                                                   (raw.numel() // arg.element_size,))
            call[arg.name] = flat.as_strided(arg.shape, arg.stride, offset)
        elif arg.kind in ("int", "float", "bool"):
            call[arg.name] = arg.value
        elif arg.kind == "none":
            call[arg.name] = None
    opts = {k: L.metadata[k] for k in ("num_warps", "num_stages") if k in L.metadata}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn[tuple(L.grid)](**call, **constexprs, **opts)
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn[tuple(L.grid)](**call, **constexprs, **opts)
    end.record()
    torch.cuda.synchronize()
    return {"device_first_launch_s": first, "device_launch_s": start.elapsed_time(end) / 1000}


def measure(unit: str) -> dict:
    import numpy as np

    from analyze_units import programs_for
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval import emulate as E
    from kernel_analyzer.reference_eval import intervals as iv
    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.premises import textbook_emulate
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    L = load_launch(Path(unit))
    progs = programs_for(L)
    row = {"unit": unit, "kernel": L.kernel_name, "programs": len(progs), "programs_total": int(np.prod(L.grid))}
    t0 = time.perf_counter()
    module = parse_ttir(L.asm["ttir"])
    module.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(module, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    original = iv.elementary_bounds
    iv.elementary_bounds = BP._float64_elementary(original)
    try:
        ev.evaluate(L, programs=progs)
    finally:
        iv.elementary_bounds = original
    row["G_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, emu = textbook_emulate(L, premises=(), programs=progs)
    row["C_exec_s"] = time.perf_counter() - t0
    row["rounding_sites"] = len(E.rounding_nodes(emu))
    t0 = time.perf_counter()
    try:
        row["prediction_status"] = BP.summarize(BP.predict(L, programs=progs))["status"]
    except Exception as e:
        row["prediction_status"] = f"error {type(e).__name__}"
    row["C_model_s"] = time.perf_counter() - t0
    row.update(device_times(unit))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--single", action="store_true")
    a = ap.parse_args()
    if a.single:
        print(json.dumps(measure(a.units)))
        return
    pool = sorted(json.loads(l)["unit"] for l in open(a.units) if json.loads(l).get("verify") == "bit_identical")
    sample = random.Random(SEED).sample(pool, a.n)
    gpus = a.gpus.split(",")
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))

    def run(i, u):
        cache = tmp / f"p6_triton_{i}"
        cache.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)], TRITON_CACHE_DIR=str(cache))
        try:
            p = subprocess.run([sys.executable, __file__, u, "-", "--single"], env=env, capture_output=True, text=True,
                               timeout=3600)
            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
            return json.loads(lines[-1]) if lines else {"unit": u, "error": p.stderr[-600:]}
        except subprocess.TimeoutExpired:
            return {"unit": u, "error": "timeout"}
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    with open(a.out, "w") as f, ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(run, i, u) for i, u in enumerate(sample)]):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
