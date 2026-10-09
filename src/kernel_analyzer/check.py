"""Kernel against its declared semantics (mode A) and, when a specification is given, against the task (mode B).

One engine for both modes.  The kernel call runs under ``TritonLaunchRecorder``; the automatic reference K_R of every
measured output comes from the captured TTIR (``evaluate_sequence``, composed over all launches of the call).

- mode A (no specification): e_num = K - K_R, through the decision layer (``analysis.assess_units``: rules R1, R2,
  R3, R5, endpoint-conservative inference on the confirmation seeds, default detector 2.1).  The report says that the
  task semantics were not checked.
- mode B (``spec`` returns f): in addition e_sem = K_R - f and the total K - f = e_num + e_sem, reported together.

A case (``Case``) gives ``setup`` (compilation, warm-up; outside the recorder), ``inputs(seed)``, ``launch(inputs)``
-> {output name: tensor} and optionally ``spec(inputs)`` -> {output name: (lo, hi)} in the tensors' logical shapes.
``BindingCase`` adapts a binding module with ``make_inputs(seed)``, ``run(inputs)`` and optional ``spec(inputs)``.
"""

from __future__ import annotations

import importlib.metadata as md
import time

import numpy as np
import torch

from .reference_eval import intervals as iv
from .reference_eval.analysis import assess_units, residual_interval
from .reference_eval.capture import TritonLaunchRecorder
from .reference_eval.ttir_eval import ST_NINF, ST_OK, evaluate_sequence, ptx_zero_fills
from .reference_eval.ttir_mapping import kernel_coverage
from .reference_eval.ttir_parser import parse_ttir

TOOL_VERSION = "4.0"   # 4.0 (dsl-v2 branch, unreleased): generic combine regions along the lowering order; execution
#                        validity (cross-program read/write, same-program cross-thread write -> read, barrier phases);
#                        3.1: audit fixes -- SumK final pass adds p_n last (the 3.0 bound was not rigorous), Welford
#                        unguarded ratio / zero-weight rows / rounding-check, upstream copies of the inputs tagged;
#                        3.0: Welford multi-value reduction; SumK / DotK accumulation; x - x, x / x one variable;
#                        2.3: upstream (non-Triton) sources tracked through the launches' own stores, not byte changes,
#                        and case inputs identified by their bytes before the launch (in-place ops inside it);
#                        2.2: output binding by storage identity (2.1: by address)
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
        """{output name: (lo, hi)} enclosing f, or None: no specification (mode A, task semantics not checked)."""
        return None


class BindingCase(Case):
    """A binding module as a case: ``make_inputs(seed)``, ``run(inputs)`` -> {output name: tensor}, optional
    ``spec(inputs)``, ``setup()``, ``NAME``, ``IMPLEMENTATION``, ``SPECIFICATION``, ``SPEC_BOUND``."""

    def __init__(self, module, name=None):
        self.module = module
        self.name = name or getattr(module, "NAME", module.__name__)
        self.implementation = getattr(module, "IMPLEMENTATION", "")
        self.specification = getattr(module, "SPECIFICATION", "" if not hasattr(module, "spec") else "given")
        self.spec_bound = getattr(module, "SPEC_BOUND", "rigorous float64 enclosure")

    def setup(self):
        if hasattr(self.module, "setup"):
            self.module.setup()

    def inputs(self, seed):
        return self.module.make_inputs(seed)

    def launch(self, inp):
        out = self.module.run(inp)
        if not isinstance(out, dict):
            raise TypeError("run(inputs) must return {output name: tensor}")
        return out

    def spec(self, inp):
        return self.module.spec(inp) if hasattr(self.module, "spec") else None


def storage_positions(out: torch.Tensor):
    """Element offsets in the storage of ``out`` for its logical elements (row-major)."""
    n = out.untyped_storage().nbytes() // out.element_size()
    return torch.arange(n).as_strided(out.shape, out.stride(), out.storage_offset()).reshape(-1).numpy()


def f64_point_spec(value, rel=2.0 ** -40):
    """A float64 evaluation of the specification taken as f with a declared bound rel * max|f| (screening use;
    a confirmed finding gets a rigorous enclosure)."""
    value = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(value)
    b = rel * float(np.max(np.abs(value[finite]))) if finite.any() else 0.0
    return (np.where(finite, iv.down(value - b), value), np.where(finite, iv.up(value + b), value))


def input_digests(inp):
    """Digests of the case's floating inputs, taken BEFORE the launch: an op inside the launch may modify an input in
    place (e.g. ATen's embedding renorm on the weight), and that modified buffer is an intermediate, not an input."""
    import hashlib

    def tensors(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from tensors(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from tensors(v)
    return {hashlib.sha1(v.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            for v in tensors(inp) if v.is_floating_point()}


def torch_intermediates(launches, seq, inp, digests_before=None):
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

    # raw bytes (bf16 / fp8 have no NumPy dtype); digests taken before the launch when the caller has them
    inputs = digests_before if digests_before is not None else \
        {digest(v.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()) for v in tensors(inp)
         if v.is_floating_point()}
    # element bit patterns of the floating inputs, per element size: an upstream buffer whose every element is one of
    # them is pure data movement (copy / gather / layout change) and its captured values are the declared inputs
    patterns = {}
    for v in tensors(inp):
        if v.is_floating_point():
            n = v.element_size()
            u = v.detach().contiguous().cpu().reshape(-1).view({2: torch.int16, 4: torch.int32, 8: torch.int64}[n]).numpy()
            patterns.setdefault(n, []).append(np.unique(u))
    patterns = {n: np.unique(np.concatenate(p)) for n, p in patterns.items()}
    sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "fp32": 4, "float64": 8, "fp64": 8, "fp16": 2, "bf16": 2}

    def data_movement(raw, dtype):
        n = sizes.get(str(dtype).replace("torch.", ""))
        if n is None or n not in patterns:
            return False
        b = np.ascontiguousarray(np.asarray(raw)).view(np.uint8)
        if b.size % n:
            return False
        el = b.view({2: np.int16, 4: np.int32, 8: np.int64}[n])
        return bool(el.size) and bool(np.isin(el, patterns[n]).all())
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
                    tag = " [copy of inputs]" if data_movement(raw, a.dtype) else ""
                    upstream.add(f"L{i}:{l.kernel_name[:40]}:{a.name}{tag}")
        stored = getattr(ref, "stored", set())
        for a in tensors:
            before = np.asarray(a.before.numpy() if hasattr(a.before, "numpy") else a.before)
            after = np.asarray(a.after.numpy() if hasattr(a.after, "numpy") else a.after)
            # written by this launch: the reference's own stores decide; a byte change only adds to them.  Comparing
            # bytes alone misses a store that rewrites identical values (an output buffer that the caching allocator
            # hands back still holding the same result from a warm-up on the same inputs)
            if a.storage_ptr in stored or before.shape != after.shape or \
                    not np.array_equal(before.view(np.uint8), after.view(np.uint8)):
                deps[a.storage_ptr] = set(upstream)
                last_after[a.storage_ptr] = digest(after)
    return {k: sorted(v) for k, v in deps.items()}


def run(case, dev=DEV, conf=CONF, zero_fill_mode="auto", keep=None, equivalence_rel=None, repeats=None,
        magnitude_bound=None):
    """One case through mode A or B.  ``keep``: a dict that receives, per output and seed, the reference interval
    and K in the output's logical element order (for independent recomputation of K_R); ``equivalence_rel``: passed
    to the decision layer (the equivalence axis next to each nonzero verdict).

    ``repeats`` (DSL v2 rc3 02 8.7): launches per input, the recorded one included; default 2, or 8 when a launch has
    float atomics.  The extra launches regenerate the input from the seed (digests checked) and keep K only (the
    reference does not depend on the schedule).  Per output: bitwise identical -> as before; an execution race found by
    the reference -> statistics withheld; different and the launch has float atomics (order-free fold admitted) ->
    residual intervals averaged within the input over the launches (outward rounding), the input is the unit;
    different for no identified reason -> execution validity not established, statistics withheld (diagnosis).

    ``magnitude_bound``: {"elementwise": M, "basis": text}, a bound on |K - G| per element over the declared population
    (from a derivation such as a per-input error budget, never from the sample); e_num rules then also report the
    bounded route (Hoeffding on truncated endpoints).  Every rule reports the approximate route's sensitivity."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    import torch._inductor.config as inductor_config

    inductor_config.use_static_cuda_launcher = False  # Inductor's static launcher bypasses the launch hook
    TritonLaunchRecorder.install_hook()
    # every case starts from an empty Dynamo cache, as PyTorch's own tests do: cases run several to a process, and a
    # cached graph can be reused for a different op when its guards do not pin the callable (pytorch#197811; B017)
    torch._dynamo.reset()
    timing = {"setup_compile_warmup": 0.0, "inputs": 0.0, "capture_first_seed": 0.0, "capture": 0.0, "reference": 0.0,
              "specification": 0.0, "statistics": 0.0, "repeat_launches": 0.0}
    r_exec = None
    t_phase = time.time()
    case.setup()
    timing["setup_compile_warmup"] = time.time() - t_phase
    per, coverage, launch_info = {}, None, None
    not_triton, modified_after, aborted_outputs, mixed_by_output = set(), set(), {}, {}
    binding_unconfirmed, reused_address = set(), set()  # detector 2.2: output-to-producer binding by storage identity
    external = None
    mixed_sources = None
    special = {}
    mode = None
    t0 = time.time()
    for seed in list(dev) + list(conf):
        t_phase = time.time()
        inp = case.inputs(seed)
        timing["inputs"] += time.time() - t_phase
        t_phase = time.time()
        digest0 = input_digests(inp)
        before = digest0 if mixed_sources is None else None
        rec = TritonLaunchRecorder()
        with rec:
            outs = case.launch(inp)
            torch.cuda.synchronize()
        # the first seed's capture also pays any JIT compilation the case's setup did not warm up
        timing["capture_first_seed" if seed == list(dev)[0] else "capture"] += time.time() - t_phase
        if coverage is None:
            coverage = [kernel_coverage(parse_ttir(l.asm["ttir"]))["complete"] for l in rec.launches]
            # float atomics make K depend on the run-time order of the atomic updates: e_num (and verdicts that
            # hinge on it) can differ between two runs of the same code; K_R and e_sem cannot
            launch_info = [{"kernel": l.kernel_name, "grid": list(l.grid), "triton": l.environment.get("triton"),
                            "float_atomics": _float_atomics(l.asm["ttir"])} for l in rec.launches]
            zero_fill = [ptx_zero_fills(l.asm.get("ptx", "")) for l in rec.launches]
            fill = zero_fill_mode == "auto" and all(zero_fill)
        t_phase = time.time()
        seq = evaluate_sequence(rec.launches, masked_fill_zero=fill)
        timing["reference"] += time.time() - t_phase
        if mixed_sources is None:
            mixed_sources = torch_intermediates(rec.launches, seq, inp, before)
        if external is None:
            # a buffer changed between recorded launches (a torch op in between): from there on its captured value
            # re-enters as an exact input, so K_R downstream carries the upstream numerical error of K
            external = [{"launch": e["launch"], "kernel": rec.launches[e["launch"]].kernel_name, "buffer": e["buffer"]}
                        for e in seq.external_writes]
        t_phase = time.time()
        specs = case.spec(inp)
        timing["specification"] += time.time() - t_phase
        this_mode = "A" if specs is None else "B"
        if mode not in (None, this_mode):
            raise ValueError("spec() must return f for every seed or for none")
        mode = this_mode
        has_f = mode == "B"
        reasons, aborted = {}, {}
        for r in seq.launches:
            for key, n in getattr(r, "reasons", {}).items():
                reasons[key] = reasons.get(key, 0) + n
            for why in (r.aborted.values() if isinstance(r.aborted, dict) else r.aborted):
                aborted[str(why)[:200]] = aborted.get(str(why)[:200], 0) + 1
        for name, out in outs.items():
            ptr = out.untyped_storage().data_ptr()
            buf = seq.memory.get(ptr)
            if buf is not None and buf.written.any():
                # an address identifies a storage only while that storage lives: bind the output to the recorded
                # writes only if it IS the recorded storage instance (detector 2.2)
                ids = rec.storage_ids(ptr)
                # identities (data address, StorageImpl address) are unique only among live storages: without the
                # recorder's keep-alive a freed intermediate's addresses can both be reused, so nothing is provable
                if not getattr(rec, "keep_storages", False) or not ids or None in ids:
                    binding_unconfirmed.add(name)  # no binding claimed
                    continue
                if out.untyped_storage()._cdata not in ids:
                    reused_address.add(name)       # another storage at a recorded address: not written by Triton
                    not_triton.add(name)
                    continue
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
            if has_f:
                f_lo_s, f_hi_s, pos = to_storage_order(out, *specs[name])
                f_lo, f_hi = f_lo_s[idx], f_hi_s[idx]
            else:  # mode A: no f; the special-value classes are compared between K and K_R
                pos = storage_positions(out)
                f_lo = f_hi = np.zeros(idx.size)
            k, r_lo = buf.actual_after[m], buf.lo[m].astype(np.float64)
            r_hi = buf.hi[m] if buf.hi is not None else r_lo  # integer buffers are exact
            if name not in mixed_by_output:
                mixed_by_output[name] = mixed_sources.get(out.untyped_storage().data_ptr(), [])
            n_lo, n_hi = residual_interval(k, r_lo, r_hi)
            s_lo, s_hi = iv.isub(r_lo, r_hi, f_lo, f_hi) if has_f else (None, None)
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
            names_ = {0: "finite", 1: "nan", 2: "+inf", 3: "-inf", -1: "?"}
            if has_f:
                mism = decided & (kr_cls != f_cls)
                k_mism = inside & (k_cls != f_cls)
                sv = special.setdefault(name, {"elements_with_special_f": 0, "kr_vs_f_class_mismatch": 0,
                                                "k_vs_f_class_mismatch": 0, "examples": []})
                sv["elements_with_special_f"] += int(((f_cls > 0) & inside).sum())
                sv["kr_vs_f_class_mismatch"] += int(mism.sum())
                sv["k_vs_f_class_mismatch"] += int(k_mism.sum())
                for j in np.flatnonzero(mism | k_mism)[: max(0, 5 - len(sv["examples"]))]:
                    sv["examples"].append({"index": int(idx[j]), "f": names_[int(f_cls[j])],
                                           "K_R": names_[int(kr_cls[j])], "K": names_[int(k_cls[j])],
                                           "K_value": float(k_arr[j])})
            else:
                k_mism = decided & (k_cls != kr_cls)
                sv = special.setdefault(name, {"elements_with_special_K_R": 0, "k_vs_kr_class_mismatch": 0,
                                                "examples": []})
                sv["elements_with_special_K_R"] += int((decided & (kr_cls > 0)).sum())
                sv["k_vs_kr_class_mismatch"] += int(k_mism.sum())
                for j in np.flatnonzero(k_mism)[: max(0, 5 - len(sv["examples"]))]:
                    sv["examples"].append({"index": int(idx[j]), "K_R": names_[int(kr_cls[j])],
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
            if has_f:
                special_agree = decided & (f_cls > 0) & (kr_cls == f_cls)
            else:
                special_agree = decided & (kr_cls > 0) & (k_cls == kr_cls)
            if keep is not None:
                keep.setdefault(name, []).append({"seed": seed, "r_lo": frame(r_lo), "r_hi": frame(r_hi),
                                                  "k": frame(np.asarray(k, dtype=np.float64)), "ok": frame(ok_e, False),
                                                  "shape": tuple(out.shape)})
            per.setdefault(name, []).append({
                "seed": seed, "r_lo": frame(r_lo), "r_hi": frame(r_hi), "k_reps": [], "repeat_inputs_differ": False,
                "n": (frame(n_lo), frame(n_hi)), "s": (frame(s_lo), frame(s_hi)) if has_f else None,
                "kr": frame(0.5 * (r_lo + r_hi)), "k": frame(np.asarray(k, dtype=np.float64)),
                "ok": frame(ok_e, False), "written": frame(np.ones(idx.size, dtype=bool), False),
                "resolved": frame(ok_e | special_agree, False), "inside": np.ones(nv, dtype=bool), "idx": pos,
                "pos": pos, "shape": tuple(out.shape), "width": frame(r_hi - r_lo), "reasons": reasons,
                "aborted": aborted})
        # repeated launches of the same input, outside the recorder: K only
        if r_exec is None:
            r_exec = int(repeats) if repeats is not None else (
                8 if any(li.get("float_atomics") for li in (launch_info or [])) else 2)
        mine = {name: rows[-1] for name, rows in per.items() if rows and rows[-1]["seed"] == seed}
        t_phase = time.time()
        for _ in range(max(0, r_exec - 1)):
            inp_r = case.inputs(seed)
            same_input = input_digests(inp_r) == digest0
            outs_r = case.launch(inp_r)
            torch.cuda.synchronize()
            for name, row in mine.items():
                if not same_input:
                    row["repeat_inputs_differ"] = True
                o = outs_r.get(name)
                if o is None or tuple(o.shape) != row["shape"]:
                    row["repeat_inputs_differ"] = True
                    continue
                v = o.detach()
                v = v.float() if v.dtype in (torch.bfloat16, torch.float16) or (
                    v.dtype.itemsize == 1 and v.is_floating_point()) else v
                row["k_reps"].append(v.cpu().double().numpy().reshape(-1))
        timing["repeat_launches"] += time.time() - t_phase
    seconds = time.time() - t0
    n_dev = len(list(dev))
    report = {"case": case.name, "mode": mode, "implementation": case.implementation,
              "specification": case.specification if mode == "B" else
              "none: task semantics not checked (mode A, e_num = K - K_R only)",
              "spec_bound": case.spec_bound, "launches": launch_info, "ttir_coverage_complete": coverage,
              "versions": {k: _version(k) for k in ("liger-kernel", "transformers", "torch", "triton")},
              "seeds": {"development": [list(dev)[0], list(dev)[-1]], "confirmation": [list(conf)[0], list(conf)[-1]]},
              "seconds": round(seconds, 1), "outputs": {},
              "tool_version": TOOL_VERSION, "launches_per_input": r_exec,
              "outputs_not_written_by_triton": sorted(not_triton),
              "outputs_binding_not_established": sorted(binding_unconfirmed),
              "outputs_at_address_of_another_recorded_storage": sorted(reused_address),
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
        execution = _execution_status(rows, launch_info, r_exec)
        entry["execution"] = execution
        for key, label in (("n", "e_num = K - K_R"), ("s", "e_sem = K_R - f"))[: 2 if mode == "B" else 1]:
            lo = np.stack([p[key][0] for p in rows])
            hi = np.stack([p[key][1] for p in rows])
            if key == "n" and execution["statistics"] == "within-input mean":
                lo, hi = _within_input_mean_residual(rows)
            t_phase = time.time()
            if execution["statistics"] == "withheld":
                rec = {"comparison": f"{name}: {label}", "verdict": "NOT_ESTABLISHED", "reason": execution["status"]}
            else:
                rec, _ = assess_units(f"{name}: {label}", lo, hi, kr, ok, n_dev, RULES, alignment_reference=kr,
                                      unit_ids=list(dev) + list(conf), equivalence_rel=equivalence_rel,
                                      magnitude_bound=magnitude_bound if key == "n" else None)
            timing["statistics"] += time.time() - t_phase
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
        if ok.any() and mode == "B":
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
        if ok.any() and mode == "B":
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
        prof = r0["s"] if mode == "B" else r0["n"]  # mode A: where e_num sits
        full[where[inside]] = np.abs(0.5 * (prof[0] + prof[1]))[inside]
        last = r0["shape"][-1] if len(r0["shape"]) else 1  # 0-dim outputs: a single column
        entry["sem_profile_last_axis" if mode == "B" else "num_profile_last_axis"] = [
            round(float(x), 8) for x in full.reshape(-1, last).mean(0)]
        if mode == "A":
            entry["task_semantics"] = "not checked (no specification)"
        report["outputs"][name] = entry
    # cost by phase (evaluation plan, work item H): compile / warm-up, input generation, capture (the launches under
    # the recorder), reference (TTIR evaluation), specification, statistics; "seconds" keeps its old meaning
    report["timing_seconds"] = {k: round(v, 3) for k, v in timing.items()}
    return report


def _execution_status(rows, launch_info, r_exec) -> dict:
    """Execution validity of one output over the units (DSL v2 rc3 02 8.7): repeated launches and race findings."""
    race = sorted({r.split("@")[0] for p in rows for r in (p["reasons"] or {}) if "execution race" in r})
    unknown = sorted({r.split("@")[0] for p in rows for r in (p["reasons"] or {}) if "execution validity" in r})
    differ, reps_done, inputs_differ = [], 0, False
    for p in rows:
        w = p["written"]
        reps_done = max(reps_done, len(p["k_reps"]))
        inputs_differ |= p["repeat_inputs_differ"]
        if any(not np.array_equal(k[w], p["k"][w], equal_nan=True) for k in p["k_reps"] if k.shape == p["k"].shape):
            differ.append(p["seed"])
    atomics = any(li.get("float_atomics") for li in (launch_info or []))
    out = {"launches_per_input": 1 + reps_done, "units_with_different_repeats": len(differ),
           "units": len(rows), "repeat_inputs_not_reproducible": inputs_differ, "race_findings": race,
           "unknown_validity_findings": unknown, "float_atomics": atomics}
    if race:
        out.update(status="execution race found by the reference: statistics withheld", statistics="withheld")
    elif differ and atomics and not inputs_differ:
        out.update(status=f"atomic execution randomness: residuals averaged within the input over {1 + reps_done} "
                          "launches", statistics="within-input mean")
    elif differ:
        out.update(status="repeated launches differ without an identified cause: execution validity not established "
                          "(diagnosis needed)" + ("; the regenerated inputs differ" if inputs_differ else ""),
                   statistics="withheld")
    elif reps_done == 0:
        out.update(status="not repeated", statistics="per launch")
    else:
        out.update(status="repeated launches bitwise identical", statistics="per launch")
    return out


def _within_input_mean_residual(rows):
    """Per unit: outward enclosure of the mean of K_j - K_R over the launches j (exact sums one ulp outward, then a
    directed division by the number of launches); elements without a complete reference keep 0 (not ok)."""
    lo_u, hi_u = [], []
    for p in rows:
        ks = [p["k"]] + [k for k in p["k_reps"] if k.shape == p["k"].shape]
        los, his = zip(*(residual_interval(k, p["r_lo"], p["r_hi"]) for k in ks))
        okm = p["ok"]
        L = np.where(okm, np.stack(los), 0.0)
        H = np.where(okm, np.stack(his), 0.0)
        s_lo, s_hi = iv.fsum_bounds(L, H, axis=0)
        n = float(len(ks))
        lo_u.append(iv.div_bounds(s_lo, n)[0])
        hi_u.append(iv.div_bounds(s_hi, n)[1])
    return np.stack(lo_u), np.stack(hi_u)


def run_black_box(case, dev=DEV, conf=CONF, equivalence_rel=None):
    """Black-box mode (evaluation plan, work item C) for implementations without TTIR (NumPy, CUDA, cuBLAS): the
    specification is the only reference, so only the total K - f is measured, through the same decision layer.
    No decomposition into e_num and e_sem; reported separately from the decomposed results."""
    timing = {"setup_compile_warmup": 0.0, "inputs": 0.0, "call": 0.0, "specification": 0.0, "statistics": 0.0}
    t_phase = time.time()
    case.setup()
    timing["setup_compile_warmup"] = time.time() - t_phase
    rows, special = {}, {}
    t0 = time.time()
    for seed in list(dev) + list(conf):
        t_phase = time.time()
        inp = case.inputs(seed)
        timing["inputs"] += time.time() - t_phase
        t_phase = time.time()
        outs = case.launch(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["call"] += time.time() - t_phase
        t_phase = time.time()
        specs = case.spec(inp)
        timing["specification"] += time.time() - t_phase
        if specs is None:
            raise ValueError("black-box mode needs spec(inputs): the specification is the only reference")
        for name, out in outs.items():
            k = (out.detach().cpu().double().numpy() if torch.is_tensor(out) else np.asarray(out, np.float64)).reshape(-1)
            f_lo, f_hi = (np.asarray(a, dtype=np.float64).reshape(-1) for a in specs[name])
            lo, hi = residual_interval(k, f_lo, f_hi)
            f_fin = np.isfinite(f_lo) & np.isfinite(f_hi)
            ok = f_fin & np.isfinite(k)
            sv = special.setdefault(name, {"elements_with_special_f": 0, "k_vs_f_class_mismatch": 0})
            sv["elements_with_special_f"] += int((~f_fin).sum())
            f_cls = np.where(np.isnan(f_lo), 1, np.where(f_lo == np.inf, 2, np.where(f_hi == -np.inf, 3, 0)))
            k_cls = np.where(np.isnan(k), 1, np.where(k == np.inf, 2, np.where(k == -np.inf, 3, 0)))
            sv["k_vs_f_class_mismatch"] += int((f_cls != k_cls).sum())
            rows.setdefault(name, []).append((np.where(ok, lo, 0.0), np.where(ok, hi, 0.0),
                                              np.where(f_fin, 0.5 * (f_lo + f_hi), 0.0), ok, tuple(np.shape(out))))
    n_dev = len(list(dev))
    report = {"case": case.name, "mode": "black-box", "implementation": case.implementation,
              "specification": case.specification, "spec_bound": case.spec_bound,
              "comparison": "K - f only (no TTIR: no reference K_R, no decomposition)",
              "seeds": {"development": [list(dev)[0], list(dev)[-1]], "confirmation": [list(conf)[0], list(conf)[-1]]},
              "seconds": round(time.time() - t0, 1), "outputs": {}}
    for name, r in rows.items():
        lo, hi, fm, ok = (np.stack([x[i] for x in r]) for i in range(4))
        t_phase = time.time()
        rec, _ = assess_units(f"{name}: K - f (black box)", lo, hi, fm, ok, n_dev, RULES, alignment_reference=fm,
                              unit_ids=list(dev) + list(conf), equivalence_rel=equivalence_rel)
        timing["statistics"] += time.time() - t_phase
        mid = 0.5 * (lo + hi)
        if ok.any():
            rec["scale"] = {"mean_abs_residual": float(np.abs(mid[ok]).mean()),
                            "max_abs_residual": float(np.abs(mid[ok]).max()),
                            "rms_reference": float(np.sqrt((fm[ok] ** 2).mean())),
                            "relative_rms": float(np.sqrt((mid[ok] ** 2).mean() / max((fm[ok] ** 2).mean(), 1e-300)))}
        report["outputs"][name] = {"elements_per_seed": int(ok.shape[1]), "shape": list(r[0][4]),
                                   "finite_fraction": float(ok.mean()), "total_black_box": rec,
                                   "special_values": special[name]}
    report["timing_seconds"] = {k: round(v, 3) for k, v in timing.items()}
    return report


def _float_atomics(ttir):
    import re

    return bool(re.search(r"tt\.atomic_rmw\s+fadd|atomic_rmw\s+fadd|tt\.atomic_rmw.*: .*f(16|32|64)", ttir))


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



def load_binding(path):
    """A binding file as a list of cases: its ``CASES`` if it defines them, else one ``BindingCase``."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    if hasattr(module, "CASES"):
        return list(module.CASES)
    return [BindingCase(module, name=getattr(module, "NAME", path.stem))]


def summary(report):
    """The decision-relevant part of a report, for regression comparison: per output the verdicts of every rule and
    detector, the reference classes, the special-value counts and the relative sizes."""
    if "error" in report:
        return {"error": report["error"].splitlines()[0]}
    out = {"mode": report.get("mode", "B"), "outputs": {}}
    for name, e in report.get("outputs", {}).items():
        row = {"reference_classes": e.get("reference_classes"),
               "special_values": {k: v for k, v in (e.get("special_values") or {}).items() if k != "examples"}}
        for key in ("numerical", "semantic"):
            if key in e:
                row[key] = {"verdicts": verdicts(e[key]), "relative_rms": (e[key].get("scale") or {}).get("relative_rms")}
        if "total" in e:
            row["total_relative_rms"] = e["total"]["relative_rms"]
        out["outputs"][name] = row
    for key in ("outputs_not_written_by_triton", "outputs_modified_after_last_triton_write"):
        out[key] = report.get(key)
    return out


def compare_reports(old, new, rtol=1e-6):
    """Differences between two reports of the same case: verdicts, classes and counts must match exactly; relative
    sizes within ``rtol`` (non-deterministic kernels, e.g. atomics, can move the last bits)."""
    a, b = summary(old), summary(new)
    diffs = []
    atomics = any(l.get("float_atomics") for r in (old, new) for l in (r.get("launches") or []))

    def walk(x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    diffs.append((path + "/" + k, x.get(k, "<absent>"), y.get(k, "<absent>")))
                else:
                    walk(x[k], y[k], path + "/" + k)
        elif isinstance(x, float) and isinstance(y, float):
            if not (x == y or abs(x - y) <= rtol * max(abs(x), abs(y))):
                diffs.append((path, x, y))
        elif x != y:
            diffs.append((path, x, y))

    walk(a, b, "")
    if atomics:  # K is order-dependent: e_num differences are expected between runs, not a regression
        diffs = [d if "/numerical/" not in d[0] else (d[0] + " [tolerated: float atomics]", d[1], d[2]) for d in diffs]
    return diffs
