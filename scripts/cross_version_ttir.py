#!/usr/bin/env python3
"""Same TTIR, two Triton versions: lowering differences and cross-version device comparison.

capture  (ka_main) saves capture packages (operands before/after) for the
         exercise and Inductor workloads of the evaluated set; the Liger and
         torchao packages captured earlier are linked in.
run      (each environment) compiles every distinct TTIR -- from the capture
         packages and from the static corpus -- with that version's compiler
         and the options of the original compilation, writes the PTX, and runs
         the binaries compiled from captured TTIR on the captured operands
         (twice, to see nondeterminism).
activate (ka_main) adds packages whose inputs reach the lowerings that differ
         (subnormal operands of sqrt / div / pow).
analyze  (ka_main) compares the PTX (normalized text, floating-point
         instruction multiset) and the device outputs between the versions.
         The TTIR is the same, so the automatic reference K_R is the same and
         E[K2 - G] - E[K1 - G] = E[K2 - K1]: any difference between the
         versions is a lowering difference.

    python scripts/cross_version_ttir.py capture --out .cache/version_ptx
    python scripts/cross_version_ttir.py activate --out .cache/version_ptx
    python scripts/cross_version_ttir.py run --out .cache/version_ptx      # in ka_main and in the 3.5.1 env
    python scripts/cross_version_ttir.py analyze --out .cache/version_ptx \
        --report results/reference_eval/version_migration/same_ttir_351_vs_360.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "results/reference_eval/ttir_corpus"
LINKED = {"liger": ROOT / ".cache/liger_kernels", "torchao": ROOT / ".cache/torchao_capture"}
OPTION_KEYS = ("num_warps", "num_stages", "num_ctas", "enable_fp_fusion", "maxnreg")
DEFAULT_OPTIONS = {"num_warps": 4, "num_stages": 3}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# capture (ka_main)
# ----------------------------------------------------------------------


def capture(out: Path, seed: int = 20261006, workload_seed: int = 0, linked: Optional[dict] = None,
            target_name: str = "captures"):
    import torch

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch
    from scripts import build_ttir_corpus, validate_inductor_reference, validate_ttir_reference

    target = out / target_name
    target.mkdir(parents=True, exist_ok=True)
    seen = set()
    linked = LINKED if linked is None else linked

    def keep(group, launches):
        for launch in launches:
            key = (launch.kernel_name, _digest(launch.asm["ttir"]))
            if key in seen or any(a.kind == "tensor" and a.window is not None for a in launch.args):
                continue
            seen.add(key)
            save_launch(launch, target / f"{group}__{launch.kernel_name[:80]}__{key[1]}")

    torch.manual_seed(seed)
    for name, fn in validate_ttir_reference.workloads(workload_seed):
        recorder = TritonLaunchRecorder()
        with recorder:
            fn()
            torch.cuda.synchronize()
        keep("exercise", recorder.launches)
    recorder = TritonLaunchRecorder()
    with recorder:
        build_ttir_corpus.exercise_kernels()
        torch.cuda.synchronize()
    keep("exercise", recorder.launches)
    from scripts import reference_eval_kernels as k

    recorder = TritonLaunchRecorder()
    with recorder:  # the remaining evaluated exercise kernels and the Liger dW accumulation kernel
        x = torch.full((16,), 2.0 ** 24, device="cuda")
        k.branch_after_rounding[(1,)](x, torch.empty_like(x), 2.0 ** 24, BLOCK=16)
        k.masked_copy[(1,)](torch.randn(128, device="cuda"), torch.empty(128, device="cuda"),
                            torch.empty(128, device="cuda"), 100, BLOCK=128)
        acc = torch.randn(1 << 16, device="cuda") * 100.0
        k.accumulate[(32,)](acc, torch.randn(1 << 16, device="cuda"), 1 << 16, BLOCK=2048)
        torch.cuda.synchronize()
    keep("exercise", recorder.launches)
    for name, make in validate_inductor_reference.workloads().items():
        torch._dynamo.reset()
        recorder = TritonLaunchRecorder()
        with recorder:
            make()()
            torch.cuda.synchronize()
        keep("inductor", recorder.launches)
    for group, root in linked.items():
        if not root.exists():
            continue
        for pkg in sorted(p for p in root.iterdir() if (p / "launch.json").exists()):
            manifest = json.loads((pkg / "launch.json").read_text())
            key = (manifest["kernel_name"], manifest["ir_sha256"]["ttir"][:16])
            link = target / f"{group}__{manifest['kernel_name'][:80]}__{pkg.name}"
            if not link.exists():
                link.symlink_to(pkg.resolve())
            seen.add(key)
    print(len(list(target.iterdir())), "capture packages")


# ----------------------------------------------------------------------
# activate (ka_main): inputs that reach the differing lowerings
# ----------------------------------------------------------------------

# Captured kernels whose inputs are rescaled so that the operand of the square root is subnormal:
# the Inductor AdamW kernel loads grad (in_ptr0), exp_avg (in_ptr1) and exp_avg_sq (in_ptr2).
ACTIVATIONS = {"inductor__triton_poi_fused_add_addcdiv_copy__div_lerp_mul_neg_pow_reciprocal_rsub_sqrt_0":
               {"in_ptr0": 2.0 ** -70, "in_ptr1": 2.0 ** -70, "in_ptr2": 2.0 ** -140}}


def activate(out: Path):
    import torch

    from kernel_analyzer.reference_eval.capture import (TritonLaunchRecorder, load_compiled, load_launch, replay,
                                                         save_launch)
    from scripts import reference_eval_kernels as k

    target = out / "captures_activated"
    target.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device="cuda").manual_seed(20261007)
    n = 4096
    normal = torch.randn(n, device="cuda", generator=g)
    tiny = torch.sign(torch.randn(n, device="cuda", generator=g)) * torch.exp2(
        -torch.randint(120, 150, (n,), device="cuda", generator=g).float())
    x = torch.where(torch.arange(n, device="cuda") % 2 == 0, normal, tiny)  # half subnormal or tiny normal
    recorder = TritonLaunchRecorder()
    with recorder:
        k.ftz_lowering[(n // 256,)](x, torch.empty_like(x), torch.empty_like(x), torch.empty_like(x), n, BLOCK=256)
        torch.cuda.synchronize()
    save_launch(recorder.launches[-1], target / f"exercise__ftz_lowering__{_digest(recorder.launches[-1].asm['ttir'])}")
    for prefix, scales in ACTIVATIONS.items():
        for pkg in sorted((out / "captures").glob(prefix + "__*")):
            launch = load_launch(pkg)
            launch.kernel = load_compiled(pkg)
            for arg in launch.args:
                if arg.name in scales:
                    vals = arg.before.view(getattr(torch, arg.dtype)) * scales[arg.name]
                    arg.before = vals.view(torch.uint8)
            after = replay(launch)  # the captured 3.6.0 binary on the rescaled operands
            for arg in launch.args:
                if arg.kind == "tensor":
                    arg.after = after[arg.storage_ptr]
            save_launch(launch, target / f"{pkg.name}__subnormal")


# ----------------------------------------------------------------------
# run (each environment)
# ----------------------------------------------------------------------

CAPTURE_DIRS = ("captures", "captures_activated")


def _items(out: Path):
    """Distinct TTIR texts with options, and the capture packages that use each."""

    items = {}
    pkgs = [p for d in CAPTURE_DIRS if (out / d).exists() for p in sorted((out / d).iterdir())]
    for pkg in pkgs:
        manifest = json.loads((pkg / "launch.json").read_text())
        ttir = (pkg / "ir/kernel.ttir").read_text()
        d = _digest(ttir)
        meta = manifest["metadata"]
        entry = items.setdefault(d, {"ttir": ttir, "source": "capture", "kernel": manifest["kernel_name"],
                                     "options": {k: meta[k] for k in OPTION_KEYS if meta.get(k) is not None},
                                     "captures": []})
        entry["captures"].append(f"{pkg.parent.name}/{pkg.name}")
    for path in sorted(CORPUS.glob("*/*.ttir")):
        ttir = path.read_text()
        d = _digest(ttir)
        if d not in items:
            items[d] = {"ttir": ttir, "source": f"corpus:{path.parent.name}/{path.name}",
                        "kernel": path.name.split("__")[1], "options": dict(DEFAULT_OPTIONS), "captures": []}
    return items


def _launch_args(launch, device="cuda"):
    import torch

    storages = {}
    for arg in launch.args:
        if arg.kind == "tensor" and arg.storage_ptr not in storages:
            storages[arg.storage_ptr] = arg.before.to(device).clone()
    args = []
    for arg in launch.ttir_params():
        if arg.kind == "tensor":
            raw = storages[arg.storage_ptr]
            dtype = getattr(torch, arg.dtype)
            offset = (arg.data_ptr - arg.storage_ptr) // arg.element_size
            flat = torch.empty(0, dtype=dtype, device=device).set_(raw.untyped_storage(), 0,
                                                                   (raw.numel() // arg.element_size,))
            args.append(flat.as_strided(arg.shape, arg.stride, offset))
        elif arg.kind in ("int", "float", "bool"):
            args.append(arg.value)
        elif arg.kind == "none":
            args.append(None)
        else:
            raise ValueError(f"cannot pass argument kind {arg.kind}")
    return storages, args


def run(out: Path):
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import CUDAOptions

    from kernel_analyzer.reference_eval.capture import load_launch

    version = triton.__version__
    target_dir = out / "versions" / version
    target_dir.mkdir(parents=True, exist_ok=True)
    allowed = set(CUDAOptions.__dataclass_fields__)
    summary = {"triton": version, "torch": torch.__version__, "items": {}}
    work = out / "work" / version
    work.mkdir(parents=True, exist_ok=True)
    for d, item in _items(out).items():
        path = work / f"{d}.ttir"
        path.write_text(item["ttir"])
        options = {k: v for k, v in item["options"].items() if k in allowed}
        rec = {"source": item["source"], "kernel": item["kernel"], "options": options, "captures": {}}
        try:
            ck = triton.compile(str(path), target=GPUTarget("cuda", 86, 32), options=options)
        except Exception as exc:
            rec["compile_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
            summary["items"][d] = rec
            print(version, d, item["kernel"][:50], "compile error", rec["compile_error"][:120], flush=True)
            continue
        (target_dir / f"{d}.ptx").write_text(ck.asm["ptx"])
        (target_dir / f"{d}.ttgir").write_text(ck.asm["ttgir"])
        rec["num_warps"] = ck.metadata.num_warps
        rec["shared"] = ck.metadata.shared
        for name in item["captures"]:
            launch = load_launch(out / name)
            results = []
            for _ in range(2):
                storages, args = _launch_args(launch)
                ck[tuple(launch.grid)](*args)
                torch.cuda.synchronize()
                results.append({ptr: raw.cpu() for ptr, raw in storages.items()})
            deterministic = all(torch.equal(results[0][p], results[1][p]) for p in results[0])
            torch.save({str(p): v for p, v in results[0].items()}, target_dir / f"{name.replace('/', '__')}.outputs.pt")
            matches_capture = all(torch.equal(results[0][a.storage_ptr], a.after)
                                  for a in launch.args if a.kind == "tensor")
            rec["captures"][name] = {"deterministic": deterministic, "matches_original_binary": matches_capture}
        summary["items"][d] = rec
        print(version, d, item["kernel"][:50], {k: v for k, v in rec["captures"].items()} or "", flush=True)
    (target_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


# ----------------------------------------------------------------------
# analyze (ka_main)
# ----------------------------------------------------------------------

FLOAT_TYPES = ("f16", "f32", "f64", "bf16", "f16x2", "bf16x2", "tf32", "e4m3", "e5m2", "e4m3x2", "e5m2x2")
FLOAT_OPS = ("ex2", "lg2", "rcp", "sqrt", "rsqrt", "sin", "cos", "tanh", "fma", "div", "mma", "wmma")


def normalize_ptx(text: str) -> list:
    lines = []
    for line in text.splitlines():
        s = line.split("//")[0].strip()
        if s.startswith(".section"):  # debug sections follow the code
            break
        if s and not s.startswith((".version", ".target", ".address_size", ".file", ".loc")):
            lines.append(s)
    return lines


def float_instructions(lines: list) -> Counter:
    out = Counter()
    for s in lines:
        if s.startswith((".", "{", "}", "$")) or s.endswith(":"):
            continue
        tok = s.split()[0]
        if tok.startswith("@"):
            tok = s.split()[1]
        parts = tok.split(".")
        if parts[0] in FLOAT_OPS or any(p in FLOAT_TYPES for p in parts[1:]):
            out[tok] += 1
    return out


def canonical_registers(lines: list) -> list:
    """Rename registers in order of first appearance, so that allocation differences vanish."""

    names = {}

    def sub(m):
        return names.setdefault(m.group(0), f"%v{len(names)}")
    return [re.sub(r"%[a-z]+[0-9]+", sub, s) for s in lines]


def reference_check(launch, outputs: dict) -> dict:
    """Each version's float outputs against the (shared) K_R: mean residual bounds E[K_v - K_R] in
    numerical-difference mode, bitwise RN(K_R) in rounding-check mode where the reference is a single
    rounded value."""

    import numpy as np
    import torch

    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    module = parse_ttir(launch.asm["ttir"])
    nd = KernelReferenceEvaluator(module).evaluate(launch)
    rc = KernelReferenceEvaluator(module, mode=NumericMode.ROUNDING_CHECK).evaluate(launch)
    out = {}
    for ptr, b in nd.buffers.items():
        if b.kind != "f" or not b.written.any():
            continue
        rb = rc.buffers.get(ptr)
        arg = next(a for a in launch.args if a.kind == "tensor" and a.storage_ptr == ptr)
        entry = {}
        for v, o in outputs.items():
            vals = o[str(ptr)].view(getattr(torch, arg.dtype)).double().numpy()[b.global_indices()]
            m = b.written & (b.st == 0) & np.isfinite(vals)
            e = {"elements": int(m.sum()),
                 "mean_residual_bounds": [float(np.mean((vals - b.hi)[m])), float(np.mean((vals - b.lo)[m]))]
                 if m.any() else None,
                 "subnormal_values": int(((np.abs(vals) < 2.0 ** -126) & (vals != 0))[b.written].sum())
                 if arg.dtype == "float32" else None,
                 "zero_values": int((vals == 0)[b.written].sum())}
            if rb is not None:
                mm = rb.written & (rb.st == 0) & (rb.lo == rb.hi)
                e["rounding_check_elements"] = int(mm.sum())
                e["differs_from_RN_K_R"] = int((vals != rb.lo)[mm].sum())
            entry[v] = e
        out[b.name] = entry
    return out


def analyze(out: Path, report_path: Path):
    import numpy as np
    import torch

    from kernel_analyzer.reference_eval.capture import load_launch

    versions = sorted(p.name for p in (out / "versions").iterdir() if (p / "summary.json").exists())
    if len(versions) != 2:
        raise ValueError(f"expected two versions, found {versions}")
    v1, v2 = versions
    s1, s2 = (json.loads((out / "versions" / v / "summary.json").read_text()) for v in versions)
    rows = []
    for d, r1 in s1["items"].items():
        r2 = s2["items"].get(d)
        row = {"ttir": d, "kernel": r1["kernel"], "source": r1["source"], "options": r1["options"]}
        if r2 is None or "compile_error" in r1 or "compile_error" in r2:
            row["compile_error"] = {v1: r1.get("compile_error"), v2: (r2 or {}).get("compile_error", "missing")}
            rows.append(row)
            continue
        p1 = normalize_ptx((out / "versions" / v1 / f"{d}.ptx").read_text())
        p2 = normalize_ptx((out / "versions" / v2 / f"{d}.ptx").read_text())
        f1, f2 = float_instructions(p1), float_instructions(p2)
        row["ptx_identical"] = p1 == p2
        row["ptx_identical_up_to_registers"] = canonical_registers(p1) == canonical_registers(p2)
        row["ptx_lines"] = {v1: len(p1), v2: len(p2)}
        row["float_instructions_identical"] = f1 == f2
        if f1 != f2:
            row["float_instruction_diff"] = {k: {v1: f1.get(k, 0), v2: f2.get(k, 0)}
                                             for k in sorted(set(f1) | set(f2)) if f1.get(k, 0) != f2.get(k, 0)}
        row["ttgir_identical"] = ((out / "versions" / v1 / f"{d}.ttgir").read_text()
                                  == (out / "versions" / v2 / f"{d}.ttgir").read_text())
        caps = {}
        for name, c1 in r1["captures"].items():
            c2 = r2["captures"][name]
            o1 = torch.load(out / "versions" / v1 / f"{name.replace('/', '__')}.outputs.pt")
            o2 = torch.load(out / "versions" / v2 / f"{name.replace('/', '__')}.outputs.pt")
            launch = load_launch(out / name)
            differ = {}
            for arg in launch.args:
                if arg.kind != "tensor" or not arg.dtype.startswith(("float", "bfloat")):
                    continue
                a = o1[str(arg.storage_ptr)].view(getattr(torch, arg.dtype)).double().numpy()
                b = o2[str(arg.storage_ptr)].view(getattr(torch, arg.dtype)).double().numpy()
                n = int((a != b).sum() - np.sum(np.isnan(a) & np.isnan(b)))
                if n:
                    diff = (b - a)[a != b]
                    differ[arg.name] = {"elements_differ": n, "elements": int(a.size),
                                        "mean_diff_v2_minus_v1": float(np.nanmean(diff)),
                                        "max_abs_diff": float(np.nanmax(np.abs(diff)))}
            ref_check = None
            if name.startswith("captures_activated/") or differ:
                ref_check = reference_check(launch, {v1: o1, v2: o2})
            caps[name] = {"bytes_identical": all(torch.equal(o1[k], o2[k]) for k in o1),
                          "reference_check": ref_check,
                          f"deterministic_{v1}": c1["deterministic"], f"deterministic_{v2}": c2["deterministic"],
                          f"{v1}_matches_original_binary": c1["matches_original_binary"],
                          f"{v2}_matches_original_binary": c2["matches_original_binary"],
                          "float_outputs_differ": differ}
        row["captures"] = caps
        rows.append(row)
        print(d, r1["kernel"][:50], "ptx", row["ptx_identical"], "regs", row["ptx_identical_up_to_registers"],
              "fp", row["float_instructions_identical"],
              {k: v["bytes_identical"] for k, v in caps.items()} if caps else "", flush=True)
    compiled = [r for r in rows if "compile_error" not in r]
    summary = {
        "distinct_ttir": len(rows), "compiled_by_both": len(compiled),
        "compile_errors": {r["ttir"]: r["compile_error"] for r in rows if "compile_error" in r},
        "ptx_identical": sum(r["ptx_identical"] for r in compiled),
        "ptx_identical_up_to_registers": sum(r["ptx_identical_up_to_registers"] for r in compiled),
        "float_instructions_identical": sum(r["float_instructions_identical"] for r in compiled),
        "float_instructions_differ": [r["kernel"] for r in compiled if not r["float_instructions_identical"]],
        "captures_run": sum(len(r["captures"]) for r in compiled),
        "captures_bytes_identical": sum(c["bytes_identical"] for r in compiled for c in r["captures"].values()),
        "captures_differ": [n for r in compiled for n, c in r["captures"].items() if not c["bytes_identical"]],
    }
    report = {"schema": "kernel-analyzer-same-ttir-cross-version-v1", "versions": [v1, v2],
              "compilers": {v1: {"torch": s1["torch"]}, v2: {"torch": s2["torch"]}}, "summary": summary,
              "rows": rows,
              "notes": ["the same TTIR text is compiled by both versions with the options of the original "
                        "compilation (corpus files without a capture: num_warps 4, num_stages 3)",
                        "PTX is compared after dropping comments, version/target headers and debug sections",
                        "same TTIR means the same K_R, so E[K2 - G] - E[K1 - G] = E[K2 - K1]"]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("capture", "activate", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    {"capture": lambda: capture(args.out), "activate": lambda: activate(args.out), "run": lambda: run(args.out),
     "analyze": lambda: analyze(args.out, args.report)}[args.stage]()


if __name__ == "__main__":
    main()
