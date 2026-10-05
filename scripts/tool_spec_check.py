#!/usr/bin/env python3
"""Kernel against its specification with the tool: e_num = K - K_R and e_sem = K_R - f.

For each case the kernel call runs under ``TritonLaunchRecorder``; the automatic reference K_R of the measured
buffer comes from the captured TTIR (``evaluate_sequence``, composed over all launches of the call), and the
specification f is the documented math of the function the kernel implements or replaces, evaluated on the same
inputs as a rigorous float64 enclosure (``reference_eval.intervals``).  Both residuals enter as directed intervals
and go through the unified decision layer (``analysis.assess_units``: rules R1, R2, R3, R5, endpoint-conservative
inference on the confirmation seeds, default detector 2.1).  Seeds 0-31 are development, 32-95 confirmation.

Reading: e_num detected / e_sem not -> numerical (class 1); e_sem detected with |e_sem| far above the K_R width
and the rounding scale -> the kernel's own semantics differ from f (class 4).  The per-coordinate profile of
|e_sem| (``sem_profile``) shows where in the output the deviation sits.

    python scripts/tool_spec_check.py --group flex --case all --out results/tool_spec/flex
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import assess_units, residual_interval  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_NINF, ST_OK, evaluate_sequence, ptx_zero_fills  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

DEV = list(range(0, 32))
CONF = list(range(32, 96))
RULES = ["R1", "R2", "R3", "R5"]


def f64(t):
    return t.detach().double().cpu().numpy()


def exact_mul(a, b):
    """Products of float32 values are exact in float64 (24 + 24 significant bits)."""
    p = a * b
    return p, p


def to_storage_order(out: torch.Tensor, lo: np.ndarray, hi: np.ndarray):
    """Spec arrays in the logical shape of ``out`` -> arrays over its storage (element offsets)."""
    n = out.untyped_storage().nbytes() // out.element_size()
    pos = torch.arange(n).as_strided(out.shape, out.stride(), out.storage_offset()).reshape(-1).numpy()
    s_lo, s_hi = np.full(n, np.nan), np.full(n, np.nan)
    s_lo[pos], s_hi[pos] = lo.reshape(-1), hi.reshape(-1)
    return s_lo, s_hi, pos


# ---------------------------------------------------------------------------------------------------------------
# Cases: Case subclasses live in tool_spec_cases_<group>.py; launch() returns {output name: tensor} and spec()
# returns {output name: (lo, hi)} in the tensor's logical shape.
# ---------------------------------------------------------------------------------------------------------------


class Case:
    name = ""
    implementation = ""   # what runs (the kernel)
    specification = ""    # what f is
    spec_bound = "rigorous float64 enclosure"

    def setup(self):
        """Once before the seeds (compilation, warm-up); runs outside the recorder."""

    def inputs(self, seed):
        raise NotImplementedError

    def launch(self, inp):
        raise NotImplementedError

    def spec(self, inp):
        raise NotImplementedError


def f64_point_spec(value, rel=2.0 ** -40):
    """A float64 evaluation of the specification taken as f with a declared bound rel * max|f| (screening use;
    a confirmed finding gets a rigorous enclosure)."""
    value = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(value)
    b = rel * float(np.max(np.abs(value[finite]))) if finite.any() else 0.0
    return (np.where(finite, iv.down(value - b), value), np.where(finite, iv.up(value + b), value))


GROUPS = {"liger": "tool_spec_cases_liger", "flex": "tool_spec_cases_flex", "inductor": "tool_spec_cases_inductor",
          "tridao": "tool_spec_cases_tridao", "fla": "tool_spec_cases_fla", "inductor2": "tool_spec_cases_inductor2",
          "inductor3": "tool_spec_cases_inductor3", "tutorials": "tool_spec_cases_triton_tutorials",
          "inductor4": "tool_spec_cases_inductor4", "vllm": "tool_spec_cases_vllm", "opinfo": "tool_spec_cases_opinfo", "optim2": "tool_spec_cases_optim2", "scatter1": "tool_spec_cases_scatter1", "vllm2": "tool_spec_cases_vllm2"}


def load_cases(group):
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return {c.name: c for c in importlib.import_module(GROUPS[group]).CASES}


# ---------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------


def torch_intermediates(launches, seq, inp):
    """Per written storage: the float buffers upstream of it (through the recorded launches) that the reference
    loaded but that are neither inputs of the case nor written by an earlier recorded launch, i.e. produced by a
    torch / ATen op in between.  K_R treats their captured values as exact inputs, so an output depending on them
    carries K's upstream numerical error in K_R, and its e_sem is mixed rather than purely semantic."""
    import hashlib

    def digest(a):
        return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes()).hexdigest()

    def tensors(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from tensors(v)

    # raw bytes (bf16 / fp8 have no NumPy dtype)
    inputs = {digest(v.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()) for v in tensors(inp)
              if v.is_floating_point()}
    deps = {}  # storage -> set of foreign buffer labels it depends on
    last_after = {}  # storage -> digest of its bytes after the last recorded launch that wrote it
    for i, (l, ref) in enumerate(zip(launches, seq.launches)):
        tensors = [a for a in l.args if a.kind == "tensor"]
        upstream = set()
        for a in tensors:
            if a.storage_ptr not in ref.loaded_any:
                continue
            raw = a.before.numpy() if hasattr(a.before, "numpy") else a.before
            if a.storage_ptr in deps and last_after.get(a.storage_ptr) == digest(raw):
                upstream |= deps[a.storage_ptr]  # unchanged since a recorded launch wrote it
            elif a.storage_ptr in ref.loaded and str(a.dtype).startswith(("float", "bfloat")):
                if digest(raw) not in inputs and np.asarray(raw).view(np.uint8).any():  # all-zero = exact constant
                    upstream.add(f"L{i}:{l.kernel_name[:40]}:{a.name}")
        for a in tensors:
            before = np.asarray(a.before.numpy() if hasattr(a.before, "numpy") else a.before)
            after = np.asarray(a.after.numpy() if hasattr(a.after, "numpy") else a.after)
            if before.shape != after.shape or not np.array_equal(before.view(np.uint8), after.view(np.uint8)):
                deps[a.storage_ptr] = set(upstream)
                last_after[a.storage_ptr] = digest(after)
    return {k: sorted(v) for k, v in deps.items()}


def run(case, dev=DEV, conf=CONF, zero_fill_mode="auto"):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    TritonLaunchRecorder.install_hook()
    # every case starts from an empty Dynamo cache, as PyTorch's own tests do: cases run several to a process, and a
    # cached graph can be reused for a different op when its guards do not pin the callable (pytorch#197811; B017)
    torch._dynamo.reset()
    case.setup()
    per, coverage, launch_info = {}, None, None
    not_triton, modified_after, aborted_outputs, mixed_by_output = set(), set(), {}, {}
    external = None
    mixed_sources = None
    special = {}
    t0 = time.time()
    for seed in list(dev) + list(conf):
        inp = case.inputs(seed)
        rec = TritonLaunchRecorder()
        with rec:
            outs = case.launch(inp)
            torch.cuda.synchronize()
        if coverage is None:
            coverage = [kernel_coverage(parse_ttir(l.asm["ttir"]))["complete"] for l in rec.launches]
            launch_info = [{"kernel": l.kernel_name, "grid": list(l.grid), "triton": l.environment.get("triton")}
                           for l in rec.launches]
            zero_fill = [ptx_zero_fills(l.asm.get("ptx", "")) for l in rec.launches]
            fill = zero_fill_mode == "auto" and all(zero_fill)
        seq = evaluate_sequence(rec.launches, masked_fill_zero=fill)
        if mixed_sources is None:
            mixed_sources = torch_intermediates(rec.launches, seq, inp)
        if external is None:
            # a buffer changed between recorded launches (a torch op in between): from there on its captured value
            # re-enters as an exact input, so K_R downstream carries the upstream numerical error of K
            external = [{"launch": e["launch"], "kernel": rec.launches[e["launch"]].kernel_name, "buffer": e["buffer"]}
                        for e in seq.external_writes]
        specs = case.spec(inp)
        reasons, aborted = {}, {}
        for r in seq.launches:
            for key, n in getattr(r, "reasons", {}).items():
                reasons[key] = reasons.get(key, 0) + n
            for why in (r.aborted.values() if isinstance(r.aborted, dict) else r.aborted):
                aborted[str(why)[:200]] = aborted.get(str(why)[:200], 0) + 1
        for name, out in outs.items():
            buf = seq.memory.get(out.untyped_storage().data_ptr())
            if buf is None or not buf.written.any():
                if aborted and buf is not None:
                    aborted_outputs.setdefault(name, sorted(aborted)[:3])  # the writing programs aborted
                else:
                    not_triton.add(name)  # produced by a non-Triton op (cuBLAS, ATen reduction): nothing to evaluate
                continue
            m = buf.written
            idx = buf.global_indices()[m]
            final = torch.empty(0, dtype=out.dtype, device=out.device).set_(out.untyped_storage()).reshape(-1)
            final = final.detach().cpu()
            final = (final.float() if final.dtype in (torch.bfloat16,) or final.dtype.itemsize == 1 and final.is_floating_point()
                     else final).numpy()  # bf16 / fp8 -> float32 is exact
            final = final[idx] if idx.size else final[:0]
            actual = np.asarray(buf.actual_after[m], dtype=np.float64)
            if buf.actual_after_st is not None:  # the decoder stores special values as 0 plus a status
                a_st = buf.actual_after_st[m]
                actual = np.where(a_st == 1, np.nan, np.where(a_st == 2, np.inf, np.where(a_st == 3, -np.inf, actual)))
            if not np.array_equal(final.astype(np.float64), actual, equal_nan=True):
                modified_after.add(name)  # a non-Triton op changed it after the last recorded launch
                continue
            f_lo_s, f_hi_s, pos = to_storage_order(out, *specs[name])
            f_lo, f_hi = f_lo_s[idx], f_hi_s[idx]
            k, r_lo = buf.actual_after[m], buf.lo[m].astype(np.float64)
            r_hi = buf.hi[m] if buf.hi is not None else r_lo  # integer buffers are exact
            if name not in mixed_by_output:
                mixed_by_output[name] = mixed_sources.get(out.untyped_storage().data_ptr(), [])
            n_lo, n_hi = residual_interval(k, r_lo, r_hi)
            s_lo, s_hi = iv.isub(r_lo, r_hi, f_lo, f_hi)
            # special values (NaN / +-inf) are excluded from the residuals; compare their classes separately
            f_cls = np.where(np.isnan(f_lo), 1, np.where(f_lo == np.inf, 2, np.where(f_hi == -np.inf, 3, 0)))
            kr_cls = np.where(buf.st[m] <= ST_NINF, buf.st[m], -1)  # -1: not established
            k_arr = actual
            k_cls = np.where(np.isnan(k_arr), 1, np.where(k_arr == np.inf, 2, np.where(k_arr == -np.inf, 3, 0)))
            # only the storage elements of this tensor view (outputs can be views of one shared buffer, e.g. the
            # gradients AOTAutograd returns; the other views' elements have no f here)
            in_view = np.zeros(max(int(pos.max(initial=-1)), int(idx.max(initial=-1))) + 1, dtype=bool)
            in_view[pos] = True
            inside = in_view[idx]
            decided = (kr_cls >= 0) & ~buf.cond[m] & inside
            mism = decided & (kr_cls != f_cls)
            k_mism = inside & (k_cls != f_cls)
            sv = special.setdefault(name, {"elements_with_special_f": 0, "kr_vs_f_class_mismatch": 0,
                                            "k_vs_f_class_mismatch": 0, "examples": []})
            sv["elements_with_special_f"] += int(((f_cls > 0) & inside).sum())
            sv["kr_vs_f_class_mismatch"] += int(mism.sum())
            sv["k_vs_f_class_mismatch"] += int(k_mism.sum())
            names_ = {0: "finite", 1: "nan", 2: "+inf", 3: "-inf", -1: "?"}
            for j in np.flatnonzero(mism | k_mism)[: max(0, 5 - len(sv["examples"]))]:
                sv["examples"].append({"index": int(idx[j]), "f": names_[int(f_cls[j])], "K_R": names_[int(kr_cls[j])],
                                       "K": names_[int(k_cls[j])], "K_value": float(k_arr[j])})
            # one coordinate frame per output: the elements of its view (the written set can change between seeds,
            # e.g. atomic scatters whose targets follow the data); unwritten elements are not ok
            nv = pos.size
            inv_pos = np.full(max(int(pos.max(initial=-1)), int(idx.max(initial=-1))) + 1, -1, dtype=np.int64)
            inv_pos[pos] = np.arange(nv)
            w = inv_pos[idx]
            sel = w >= 0

            def frame(a, fill=0.0):
                a = np.asarray(a)
                o = np.full(nv, fill, dtype=a.dtype if a.dtype != bool else bool)
                o[w[sel]] = a[sel]
                return o

            ok_e = (buf.st[m] == ST_OK) & ~buf.cond[m] & np.isfinite(f_lo)
            special_agree = decided & (f_cls > 0) & (kr_cls == f_cls)
            per.setdefault(name, []).append({
                "n": (frame(n_lo), frame(n_hi)), "s": (frame(s_lo), frame(s_hi)),
                "kr": frame(0.5 * (r_lo + r_hi)), "k": frame(np.asarray(k, dtype=np.float64)),
                "ok": frame(ok_e, False), "written": frame(np.ones(idx.size, dtype=bool), False),
                "resolved": frame(ok_e | special_agree, False), "inside": np.ones(nv, dtype=bool), "idx": pos,
                "pos": pos, "shape": tuple(out.shape), "width": frame(r_hi - r_lo), "reasons": reasons,
                "aborted": aborted})
    seconds = time.time() - t0
    n_dev = len(list(dev))
    report = {"case": case.name, "implementation": case.implementation, "specification": case.specification,
              "spec_bound": case.spec_bound, "launches": launch_info, "ttir_coverage_complete": coverage,
              "versions": {k: _version(k) for k in ("liger-kernel", "transformers", "torch", "triton")},
              "seeds": {"development": [list(dev)[0], list(dev)[-1]], "confirmation": [list(conf)[0], list(conf)[-1]]},
              "seconds": round(seconds, 1), "outputs": {},
              "outputs_not_written_by_triton": sorted(not_triton),
              "outputs_modified_after_last_triton_write": sorted(modified_after),
              "outputs_whose_writing_programs_aborted": aborted_outputs,
              "external_reentries_seed0": external,
              "masked_lane_assumption": {"applied": fill, "ptx_zero_fills_per_launch": zero_fill,
                                         "meaning": "masked-off lanes of loads without `other` taken as 0 (the "
                                                    "lowering zero-initializes them; TTIR leaves them undefined)"}}
    for name, rows in per.items():
        ok = np.stack([p["ok"] for p in rows])
        kr = np.stack([p["kr"] for p in rows])
        written = np.stack([p["written"] for p in rows])
        resolved = np.stack([p["resolved"] for p in rows])
        entry = {"elements_per_seed": int(written[0].sum()), "shape": list(rows[0]["shape"]),
                 "depends_on_non_triton_intermediates": mixed_by_output.get(name, []),
                 # written elements whose reference is complete and finite, or a special value of f's class
                 "reference_classes": {"complete_fraction": float(resolved[written].mean()) if written.any() else 0.0,
                                       "finite_complete_fraction": float(ok[written].mean()) if written.any() else 0.0,
                                       "written_fraction": float(written.mean())},
                 "special_values": special.get(name),
                 "not_established_reasons_seed0": rows[0]["reasons"],
                 "aborted_programs_seed0": rows[0]["aborted"]}
        for key, label in (("n", "e_num = K - K_R"), ("s", "e_sem = K_R - f")):
            lo = np.stack([p[key][0] for p in rows])
            hi = np.stack([p[key][1] for p in rows])
            rec, _ = assess_units(f"{name}: {label}", lo, hi, kr, ok, n_dev, RULES, alignment_reference=kr,
                                  unit_ids=list(dev) + list(conf))
            mid = 0.5 * (lo + hi)
            if ok.any():
                rec["scale"] = {"mean_abs_residual": float(np.abs(mid[ok]).mean()),
                                "max_abs_residual": float(np.abs(mid[ok]).max()),
                                "rms_reference": float(np.sqrt((kr[ok] ** 2).mean())),
                                "relative_rms": float(np.sqrt((mid[ok] ** 2).mean() / max((kr[ok] ** 2).mean(), 1e-300))),
                                "max_K_R_width": float(np.stack([p["width"] for p in rows])[ok].max())}
            entry["numerical" if key == "n" else "semantic"] = rec
        # element-wise view of e_sem over every (seed, element) with a complete finite reference, independent of
        # the coordinate set fixed on the development units (outputs whose finite positions move with the data,
        # e.g. masks that set data-dependent entries to -inf, leave that set nearly empty)
        if ok.any():
            s_lo_all = np.stack([p["s"][0] for p in rows])[ok]
            s_hi_all = np.stack([p["s"][1] for p in rows])[ok]
            mid_all = 0.5 * (s_lo_all + s_hi_all)
            entry["semantic_elementwise"] = {
                "elements": int(ok.sum()),
                "certified_frac": float(((s_lo_all > 0) | (s_hi_all < 0)).mean()),
                "relative_rms": float(np.sqrt((mid_all ** 2).mean() / max((kr[ok] ** 2).mean(), 1e-300)))}
        # the total K - f = e_num + e_sem: a semantic deviation that the rounded execution undoes (a branch that
        # K_R decides on real values while the device decides on rounded ones, e.g. an equality between a
        # compile-time rounded constant and a value cast at run time) shows as large e_sem and e_num of
        # opposite sign with a small total
        if ok.any():
            tot = 0.5 * (np.stack([p["n"][0] for p in rows]) + np.stack([p["n"][1] for p in rows])) + \
                0.5 * (np.stack([p["s"][0] for p in rows]) + np.stack([p["s"][1] for p in rows]))
            entry["total"] = {"comparison": "K - f = e_num + e_sem (midpoints)",
                              "relative_rms": float(np.sqrt((tot[ok] ** 2).mean() / max((kr[ok] ** 2).mean(), 1e-300))),
                              "max_abs": float(np.abs(tot[ok]).max())}
        # where e_sem sits in seed 0: mean |e_sem| per index of the last logical axis
        r0 = rows[0]
        inv = np.full(int(max(r0["pos"].max(), r0["idx"].max(initial=0))) + 1, -1)
        inv[r0["pos"]] = np.arange(r0["pos"].size)
        full = np.zeros(r0["pos"].size)
        where = inv[r0["idx"]]
        inside = where >= 0  # written elements of the storage that belong to this tensor view
        full[where[inside]] = np.abs(0.5 * (r0["s"][0] + r0["s"][1]))[inside]
        last = r0["shape"][-1] if len(r0["shape"]) else 1  # 0-dim outputs: a single column
        entry["sem_profile_last_axis"] = [round(float(x), 8) for x in full.reshape(-1, last).mean(0)]
        report["outputs"][name] = entry
    return report


def _version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def verdicts(rec):
    out = {r["rule"]: r.get("verdict") for r in rec.get("rules", [])}
    det = rec.get("default_detector") or {}
    out["detector_vector_mean"] = (det.get("vector_mean") or {}).get("verdict")
    out["detector_alignment"] = (det.get("alignment") or {}).get("verdict")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--case", required=True, help="case name, or 'all'")
    parser.add_argument("--out", type=Path, required=True, help="directory; one JSON per case")
    parser.add_argument("--seeds", type=int, default=96, help="development = first third")
    parser.add_argument("--zero-fill", default="auto", choices=("auto", "off"),
                        help="auto: take masked lanes without `other` as 0 when the PTX of every launch zero-fills them")
    args = parser.parse_args()
    import torch._inductor.config as inductor_config

    inductor_config.use_static_cuda_launcher = False  # Inductor's static launcher bypasses the launch hook
    cases = load_cases(args.group)
    names = sorted(cases) if args.case == "all" else args.case.split(",")
    n_dev = args.seeds // 3
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            report = run(cases[name], dev=range(0, n_dev), conf=range(n_dev, args.seeds), zero_fill_mode=args.zero_fill)
        except Exception as exc:  # noqa: BLE001
            import traceback

            report = {"case": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]}
            print(name, "ERROR", report["error"][:300], flush=True)
        (args.out / f"{name}.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
        for oname, entry in report.get("outputs", {}).items():
            sem, num = entry["semantic"], entry["numerical"]
            print(f"{name:34s} {oname:6s} complete={entry['reference_classes']['complete_fraction']:.2f} "
                  f"num={verdicts(num)} sem={verdicts(sem)} "
                  f"sem_rel_rms={sem.get('scale', {}).get('relative_rms', float('nan')):.1e} "
                  f"num_rel_rms={num.get('scale', {}).get('relative_rms', float('nan')):.1e}", flush=True)


if __name__ == "__main__":
    main()
