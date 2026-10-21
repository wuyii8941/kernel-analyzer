"""Kernel against its declared semantics and, when supplied, an external numerical reference.

One engine for both modes.  The kernel call runs under ``TritonLaunchRecorder``; the automatic reference K_R of every
measured output comes from the captured TTIR (``evaluate_sequence``, composed over all launches of the call).

- mode A (no specification): e_num = K - K_R, through the decision layer (``analysis.assess_units``: rules R1, R2,
  R3, R5, endpoint-conservative inference on the confirmation seeds, default detector 2.1).  The report says that the
  task semantics were not checked.
- mode B (``spec`` returns a reference r): additionally report K_R - r and K - r.  This interface does not establish
  the reference's source, implementation validity or binding to the captured kernel, so these differences are not
  semantic-error verdicts.  Certified use requires the separate contract admission and binding path.

A case (``Case``) gives ``setup`` (compilation, warm-up; outside the recorder), ``inputs(seed)``, ``launch(inputs)``
-> {output name: tensor} and optionally ``spec(inputs)`` -> {output name: (lo, hi)} in the tensors' logical shapes.
``BindingCase`` adapts a binding module with ``make_inputs(seed)``, ``run(inputs)`` and optional ``spec(inputs)``.
"""

from __future__ import annotations

import collections
import importlib.metadata as md
import time

import numpy as np
import torch

from .reference_eval import intervals as iv
from .reference_eval.analysis import assess_units, residual_interval
from .reference_eval.capture import TritonLaunchRecorder
from .reference_eval.storage_effects import InputSnapshot
from .reference_eval.ttir_eval import (ST_NINF, ST_OK, NumericMode, evaluate_sequence, launch_ir, ptx_zero_fills,
                                       unproven_premise)
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
    specification = ""    # what the supplied reference is
    spec_bound = "unspecified numerical reference; no rigorous guarantee recorded"

    def setup(self):
        """Once before the seeds (compilation, warm-up); runs outside the recorder."""

    def inputs(self, seed):
        raise NotImplementedError

    def launch(self, inp):
        raise NotImplementedError

    def spec(self, inp):
        """{output name: (lo, hi)} for a supplied reference, or None for mode A."""
        return None


class BindingCase(Case):
    """A binding module as a case: ``make_inputs(seed)``, ``run(inputs)`` -> {output name: tensor}, optional
    ``spec(inputs)``, ``setup()``, ``NAME``, ``IMPLEMENTATION``, ``SPECIFICATION``, ``SPEC_BOUND``."""

    def __init__(self, module, name=None):
        self.module = module
        self.name = name or getattr(module, "NAME", module.__name__)
        self.implementation = getattr(module, "IMPLEMENTATION", "")
        self.specification = getattr(module, "SPECIFICATION", "" if not hasattr(module, "spec") else "given")
        self.spec_bound = getattr(module, "SPEC_BOUND",
                                  "unspecified numerical reference; no rigorous guarantee recorded")

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
    """A float64 point evaluation with a heuristic relative band for screening.

    The band is not a proved error enclosure and this function does not establish a task specification.
    """
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


def input_storages(inp):
    """Storage addresses of the case's floating input tensors, taken before the launch (audit F04: a buffer counts as
    a declared input only if it IS an input's storage, unchanged; equal bytes elsewhere prove nothing)."""
    out = set()

    def walk(obj):
        if torch.is_tensor(obj):
            if obj.is_floating_point():
                out.add(obj.untyped_storage().data_ptr())
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
    walk(inp)
    return out


PRODUCER_RECORD = " [producer record: "   # label suffix of an upstream buffer with a recorded copy / constant chain


def torch_intermediates(launches, seq, inp, snapshot=None, producers=None, decisions=None,
                        include_non_floating=False):
    """Per written storage: the float buffers upstream of it (through the recorded launches) that the reference
    loaded but that are neither declared inputs of the case nor carried over from an earlier recorded launch, i.e.
    produced by a torch / ATen op in between.  K_R treats their captured values as exact inputs, so an output
    depending on them carries K's upstream numerical error in K_R, and its e_sem is mixed rather than purely semantic.

    The memory-effect rules are those of ``reference_eval.storage_effects`` (M1-M5).  A read of a buffer's captured
    initial values is a declared-input read only when (M1) the storage is a declared input's storage instance, held
    alive since before the call, (M2, M3) the bytes the reference read lie inside the declared input tensors on that
    storage and equal *that storage's own* bytes before the call (``snapshot``: ``InputSnapshot`` taken before the
    call; equal bytes of another input, or zeros, are not provenance), and (M3) there is no write evidence: no
    unannounced ATen-visible version increment on the storage (``CapturedArg.mutations``), and no ATen write of the
    bytes read in the aligned producer trace (``producers``).  A buffer written by a recorded launch carries over
    only when its bytes are unchanged and no write evidence lies in between (M4).  ATen arithmetic can round back to
    the input bits or to the bits of another input (x + 2^-25 -> x for x in [1, 2)), and an underflow gives zeros
    whose real value is not zero; such buffers stay upstream and the reference kernel-level.

    ``producers`` ({(launch, argument): (provenance, chain)} from ``provenance.producer_records``): a buffer whose
    recorded producers are only copies of declared inputs or exact constants gets the label suffix
    ``PRODUCER_RECORD``; it is an upstream buffer whose captured values are the call's own values, not a mixed one.
    ``decisions``: a list that receives one entry per relevant read (launch, argument, outcome, reason) for the report.
    By default only floating reads are relevant to the numerical decomposition.  Contract binding sets
    ``include_non_floating`` because an undeclared integer index or boolean mask is also an omitted task input.
    Without ``snapshot`` one is taken now (after the call): callers that can should take it before the call."""
    import hashlib

    from .reference_eval.storage_effects import InputSnapshot, carry_over, read_runs_of

    def digest(a):
        return hashlib.sha1(np.ascontiguousarray(np.asarray(a)).view(np.uint8).tobytes()).hexdigest()

    snap = snapshot if snapshot is not None else InputSnapshot(inp)
    from .provenance import ATEN_WRITES
    aten = (producers or {}).get(ATEN_WRITES) or {}
    deps = {}        # storage -> set of foreign buffer labels it depends on
    last_after = {}  # storage -> digest of its bytes after the last recorded launch that wrote it
    last_mut = {}    # storage -> write-evidence count at that launch
    last_writer = {}  # storage -> index of that launch
    for i, (l, ref) in enumerate(zip(launches, seq.launches)):
        tensors = [a for a in l.args if a.kind == "tensor"]
        upstream = set()
        for a in tensors:
            if a.storage_ptr not in ref.loaded_any:
                continue
            raw = a.before.numpy() if hasattr(a.before, "numpy") else a.before
            mut = getattr(a, "mutations", None)
            if a.storage_ptr in deps:
                keep, why = carry_over(last_after.get(a.storage_ptr) == digest(raw), mut, last_mut.get(a.storage_ptr))
                n_now = aten.get(i, {}).get(a.storage_ptr)
                n_then = aten.get(last_writer.get(a.storage_ptr), {}).get(a.storage_ptr)
                if keep and n_now is not None and n_then is not None and n_now > n_then:
                    # M4: the producer trace saw an ATen write that bytes and version counters miss (tensor.data)
                    keep, why = False, f"ATen write between the launches in the producer trace ({n_now - n_then})"
                if keep:
                    upstream |= deps[a.storage_ptr]  # unchanged since a recorded launch wrote it (M4)
                    continue
            else:
                why = None
            if a.storage_ptr in ref.loaded and (include_non_floating or
                                                str(a.dtype).startswith(("float", "bfloat"))):
                rec_ = (producers or {}).get((i, a.name))
                if why is None:
                    ok, why = snap.declared_read(a, read_runs_of(ref, a))
                    if ok:
                        # write evidence (M3): every source that saw a write is reported
                        seen = []
                        if mut:
                            seen.append(f"ATen-visible write before the launch ({mut} version increments)")
                        if rec_ is not None and rec_[0] != "input":
                            seen.append(f"ATen write in the producer trace ({rec_[0]}: {str(rec_[1])[:80]})")
                        if not seen and mut is None and (rec_ is None or rec_[0] != "input"):
                            seen.append("no write evidence (no version counter, no aligned producer trace)")
                        if seen:
                            ok, why = False, "; ".join(seen)
                    if ok:
                        if decisions is not None:
                            decisions.append({"launch": i, "argument": a.name, "outcome": "declared input",
                                              "evidence": "identity, coverage, own bytes, " + (
                                                  "version counters" if mut == 0 else "producer trace")})
                        continue
                tag = f"{PRODUCER_RECORD}{rec_[0]}: {rec_[1][:120]}]" if rec_ and rec_[0] in ("input", "const",
                                                                                                "copy") else ""
                upstream.add(f"L{i}:{l.kernel_name[:40]}:{a.name}{tag}")
                if decisions is not None:
                    decisions.append({"launch": i, "argument": a.name, "outcome": "upstream", "reason": why,
                                      "producer": list(rec_) if rec_ else None})
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
                last_mut[a.storage_ptr] = getattr(a, "mutations", None)
                last_writer[a.storage_ptr] = i
    return {k: sorted(v) for k, v in deps.items()}


def contract_binding_context(outs, mixed_sources, producers=None):
    """Unaccounted tensor sources that reach each public output.

    A reviewed task mapping accounts for the call inputs, but a Python call can
    also close over a module tensor or global tensor.  The reference evaluator
    already follows those sources through recorded launches.  Copies of a
    declared input and exact constants with an aligned producer record are
    admitted; every other source must stop contract binding rather than merely
    downgrade the K-G decomposition.
    """
    hidden = {}
    for name, out in outs.items():
        if not torch.is_tensor(out):
            continue
        storage = out.untyped_storage().data_ptr()
        sources = sorted(source for source in mixed_sources.get(storage, ()) if PRODUCER_RECORD not in source)
        if sources:
            hidden[name] = sources
    context = {"unaccounted_sources_by_output": hidden}
    from .provenance import ENCLOSED_LAUNCHES
    enclosed = (producers or {}).get(ENCLOSED_LAUNCHES, [])
    if enclosed:
        context["dispatch_calls_with_untraced_internals"] = enclosed
    return context


CAPTURE_STORE_BYTES = int(__import__("os").environ.get("KA_CAPTURE_STORE_BYTES", str(8 << 30)))


def precision_environment() -> dict:
    """The precision switches the candidate runs under (recorded, never changed by the analysis)."""
    import os
    m = torch.backends.cuda.matmul
    return {"torch.backends.cuda.matmul.allow_tf32": bool(m.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch.backends.cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "allow_fp16_reduced_precision_reduction": bool(getattr(m, "allow_fp16_reduced_precision_reduction", False)),
            "allow_bf16_reduced_precision_reduction": bool(getattr(m, "allow_bf16_reduced_precision_reduction", False)),
            "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT")}


ANALYSIS_INTERVENTIONS = [
    "torch._inductor.config.use_static_cuda_launcher = False: Inductor launches its compiled kernels through the Python "
    "launcher, where the recorder's hook sits (same kernel binaries, same arguments)",
    "torch._dynamo.reset() before the case setup (each case starts from an empty Dynamo cache)",
    "the producer trace: one unmeasured call per seed under an ATen dispatch mode and force_eager stance, from the "
    "measured run's random state; the random state is restored afterwards",
    "no precision switch (TF32, float32 matmul precision, reduced-precision reductions) is set by the analysis"]


class _OutputSnapshot:
    """A measured output as it was at the end of the measured call (T1: later calls -- the unmeasured producer trace,
    repeated launches, the next unit -- can write the same buffers): its storage identity and a host copy of its storage
    viewed with the output's dtype, offset, sizes and strides."""

    def __init__(self, t):
        from .reference_eval.storage_effects import _host_bytes
        st = t.untyped_storage()
        self.ptr, self.cdata = st.data_ptr(), st._cdata
        host = _host_bytes(st)
        self.view = torch.empty(0, dtype=t.dtype).set_(host.untyped_storage(), t.storage_offset(), tuple(t.shape),
                                                       tuple(t.stride()))


def _contract_candidate_values(t, name):
    """A public result as exact binary64 values, or a refusal when an integer cannot be represented exactly."""
    if not t.is_floating_point() and t.dtype != torch.bool and any(
            abs(int(value)) > 2 ** 53 for value in t.reshape(-1).tolist()):
        raise ValueError(f"contract output {name!r} contains an integer outside exact binary64 range")
    return t.double().numpy().reshape(-1)


def _contract_special_values(rows):
    """Special-value class comparison over the complete public output used by K-f."""
    special_f = mismatches = unbounded = 0
    for row in rows:
        lo, hi, k = row["f_lo"], row["f_hi"], row["k"]
        f_cls = np.where(np.isnan(lo), 1, np.where(lo == np.inf, 2, np.where(hi == -np.inf, 3, 0)))
        k_cls = np.where(np.isnan(k), 1, np.where(k == np.inf, 2, np.where(k == -np.inf, 3, 0)))
        point_special = f_cls > 0
        special_f += int(point_special.sum())
        mismatches += int(((f_cls != k_cls) & (point_special | (k_cls > 0))).sum())
        unbounded += int(((~np.isfinite(lo) | ~np.isfinite(hi)) & ~point_special).sum())
    return {"elements_with_special_f": special_f, "k_vs_f_class_mismatch": mismatches,
            "unbounded_reference_elements": unbounded,
            "scope": "complete public output against admitted task specification"}


def _boolean_task_comparison(rows):
    """Exact complete-output Boolean comparison in the canonical False=0, True=1 frame."""
    coordinates = matches = false_positive = false_negative = 0
    per_unit = []
    for row in rows:
        valid = np.asarray(row["ok_total"], dtype=bool)
        candidate = np.asarray(row["k"], dtype=np.float64)
        target_lo = np.asarray(row["f_lo"], dtype=np.float64)
        target_hi = np.asarray(row["f_hi"], dtype=np.float64)
        if np.any(valid & ((candidate != 0.0) & (candidate != 1.0))):
            raise ValueError("a Boolean call output did not remain in the exact 0/1 comparison frame")
        if np.any(valid & ((target_lo != target_hi) | ((target_lo != 0.0) & (target_lo != 1.0)))):
            raise ValueError("an exact Boolean task reference did not remain in the exact 0/1 comparison frame")
        same = valid & (candidate == target_lo)
        fp = valid & (candidate == 1.0) & (target_lo == 0.0)
        fn = valid & (candidate == 0.0) & (target_lo == 1.0)
        n = int(valid.sum())
        miss = int((fp | fn).sum())
        coordinates += n
        matches += int(same.sum())
        false_positive += int(fp.sum())
        false_negative += int(fn.sum())
        per_unit.append({"seed": row["seed"], "coordinates": n, "mismatches": miss,
                         "mismatch_rate": miss / n if n else None})
    mismatches = false_positive + false_negative
    return {"encoding": "False=0, True=1 after strict Boolean validation",
            "coordinates": coordinates, "matches": matches, "mismatches": mismatches,
            "mismatch_rate": mismatches / coordinates if coordinates else None,
            "false_positive": false_positive, "false_negative": false_negative,
            "signed_truth_rate_difference": ((false_positive - false_negative) / coordinates
                                               if coordinates else None),
            "per_unit": per_unit,
            "meaning": "exact Boolean disagreements; signed differences may cancel, so mismatch counts are primary"}


def _capture_bytes(launches) -> int:
    n = 0
    for l in launches:
        for a in list(l.args) + list(getattr(l, "implicit", []) or []):
            for t in (a.before, a.after):
                if t is not None:
                    n += int(t.numel() * t.element_size()) if hasattr(t, "numel") else int(np.asarray(t).nbytes)
    return n


def _trace_conflicts(launches, seq, prods) -> set:
    """(launch, storage) pairs the reference memory carried over although the aligned producer trace saw an ATen
    write to the storage after the last recorded launch that touched it (storage_effects M4)."""
    from .provenance import ATEN_WRITES
    aten = (prods or {}).get(ATEN_WRITES) or {}
    out = set()
    for c in seq.carry_overs:
        i, ptr = c["launch"], c["storage"]
        prev = [j for j in range(i) if any(a.kind == "tensor" and a.storage_ptr == ptr for a in launches[j].args)]
        if not prev:
            continue
        n_now, n_then = aten.get(i, {}).get(ptr), aten.get(prev[-1], {}).get(ptr)
        if n_now is not None and n_then is not None and n_now > n_then:
            out.add((i, ptr))
    return out


def run(case, dev=DEV, conf=CONF, zero_fill_mode="auto", keep=None, equivalence_rel=None, repeats=None,
        magnitude_bound=None, alpha=0.05, producer_records=True, precision_level=1, captures=None, observe=None):
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
    bounded route (Hoeffding on truncated endpoints).  Every rule reports the approximate route's sensitivity.
    ``alpha``: the level of every rule test (the declared significance level of the unified entry).
    ``producer_records``: when a kernel reads the initial values of a float buffer, run the call once more per seed,
    unmeasured, under an ATen trace (``provenance.producer_records``): the trace is write evidence for declared inputs
    (an ATen write the version counters miss, e.g. through ``tensor.data``) and gives the producers of upstream
    buffers; buffers produced only by copies of the inputs or exact constants are then not mixed sources (audit F04
    follow-up; storage_effects M3).
    ``precision_level``: working precision of the reference (``intervals.working_precision``: 1, 2 or 3).
    ``captures``: a dict.  Empty: every seed's capture (launches, outputs, input snapshot, producer records,
    specification, repeated launches) is stored in it.  Filled by an earlier call: nothing is executed; the reference and
    the statistics are recomputed on the stored capture (same inputs, initial state and actual outputs), e.g. at a
    higher working precision.  When the stored launches exceed ``CAPTURE_STORE_BYTES`` the store is dropped and
    ``captures["overflow"]`` set.

    The analysis never changes the candidate's precision switches; they are recorded (``precision_environment``),
    with the analysis' own interventions (``analysis_interventions``).

    ``observe``: {output: {"layer": "increment", "old": input path, "floor": {...}}} -- the observation layer of an
    output (``reference_eval.observation``).  For the increment layer the old value is the input at ``old`` before the
    call; the decision layer's reference-dependent directions follow the reference increment, and the per-unit
    increment quantities (dropped updates, sign flips, retention, error energy) are reported under ``observation``.

    Proof status (audit F03): elements whose reference holds only under an unproven premise are reported apart
    (``complete_under_premise_fraction``) and are not ok for the statistics.  IR (audit F05): each launch is analysed
    on the IR it was captured with (TTIR, or TTGIR for Gluon kernels), the same choice for coverage, atomic
    detection and evaluation, and its kind is reported."""
    from .provenance import rng_state

    replay = captures is not None and bool(captures.get("seeds"))
    store = captures is not None and not replay and not captures.get("overflow")
    if store:
        captures["seeds"], captures["bytes"] = {}, 0
    import torch._inductor.config as inductor_config

    inductor_config.use_static_cuda_launcher = False  # Inductor's static launcher bypasses the launch hook
    TritonLaunchRecorder.install_hook()
    timing = {"setup_compile_warmup": 0.0, "inputs": 0.0, "capture_first_seed": 0.0, "capture": 0.0, "reference": 0.0,
              "specification": 0.0, "statistics": 0.0, "repeat_launches": 0.0}
    r_exec = None
    t_phase = time.time()
    if replay:
        precision_env = captures["precision_environment"]
    else:
        # every case starts from an empty Dynamo cache, as PyTorch's own tests do: cases run several to a process, and
        # a cached graph can be reused for a different op when its guards do not pin the callable (pytorch#197811)
        torch._dynamo.reset()
        case.setup()
        precision_env = precision_environment()
        if captures is not None:
            captures["precision_environment"] = precision_env
    timing["setup_compile_warmup"] = time.time() - t_phase
    per, task_totals, coverage, launch_info = {}, {}, None, None
    contract_bindings = []
    not_triton, modified_after, aborted_outputs, mixed_by_output = set(), set(), {}, {}
    binding_unconfirmed, reused_address = set(), set()  # detector 2.2: output-to-producer binding by storage identity
    trace_status = collections.Counter()
    provenance_seed0 = None
    precision_calls = 0   # calls of working-precision-dependent rules in the reference evaluations
    program_reference_mode = getattr(getattr(case, "contract_reference", None), "program_reference_mode",
                                     NumericMode.NUMERICAL_DIFFERENCE)
    if program_reference_mode not in (NumericMode.NUMERICAL_DIFFERENCE, NumericMode.ROUNDING_CHECK):
        raise ValueError(f"unknown program reference mode {program_reference_mode!r}")
    external = None
    special = {}
    mode = None
    t0 = time.time()
    for seed in list(dev) + list(conf):
        if replay:
            cap = captures["seeds"][seed]
            inp, snap, rec, out_snaps = cap["inp"], cap["snap"], cap["rec"], cap["out_snaps"]
        else:
            t_phase = time.time()
            inp = case.inputs(seed)
            timing["inputs"] += time.time() - t_phase
            t_phase = time.time()
            # the declared inputs before the call (storage_effects M1 / M3): storages held, own bytes, coverage, and an
            # ordered digest per input path (two inputs that swap contents are different inputs)
            snap = InputSnapshot(inp)
            rec = TritonLaunchRecorder()
            rec.watch_tensors(*(snap.all_tensors if getattr(case, "contract_reference", None) is not None
                                else snap.tensors))
            rng_before = rng_state()       # the trace replays the measured run's random draws (T1)
            with rec:
                outs = case.launch(inp)
                torch.cuda.synchronize()
            out_snaps = {name: _OutputSnapshot(o) for name, o in outs.items()}
            # the first seed's capture also pays any JIT compilation the case's setup did not warm up
            timing["capture_first_seed" if seed == list(dev)[0] else "capture"] += time.time() - t_phase
            if store:
                captures["bytes"] += _capture_bytes(rec.launches)
                if captures["bytes"] > CAPTURE_STORE_BYTES:
                    captures["seeds"], store, captures["overflow"] = None, False, True
                else:
                    cap = captures["seeds"][seed] = {"inp": inp, "snap": snap, "rec": rec, "out_snaps": out_snaps}
        if coverage is None:
            irs = [launch_ir(l) for l in rec.launches]   # one IR choice for coverage, atomics and evaluation
            coverage = [kernel_coverage(parse_ttir(text))["complete"] for _, text in irs]
            # float atomics make K depend on the run-time order of the atomic updates: e_num (and verdicts that
            # hinge on it) can differ between two runs of the same code; K_R and e_sem cannot
            launch_info = [{"kernel": l.kernel_name, "grid": list(l.grid), "triton": l.environment.get("triton"),
                            "ir_kind": kind, "float_atomics": _float_atomics(text)}
                           for l, (kind, text) in zip(rec.launches, irs)]
            zero_fill = [ptx_zero_fills(l.asm.get("ptx", "")) for l in rec.launches]
            fill = zero_fill_mode == "auto" and all(zero_fill)
        t_phase = time.time()
        calls0 = sum(iv.PRECISION_DEPENDENT_CALLS.values())
        with iv.working_precision(precision_level):
            seq = evaluate_sequence(rec.launches, mode=program_reference_mode, masked_fill_zero=fill)
        precision_calls += sum(iv.PRECISION_DEPENDENT_CALLS.values()) - calls0
        timing["reference"] += time.time() - t_phase
        # every seed (audit F04): the upstream sources of a storage can change with the data and the allocator
        if replay:
            prods, status = cap["prods"], cap["trace_status"]
        else:
            prods = None
            if producer_records and any(r.loaded for r in seq.launches):
                # an unmeasured traced run of the call for this seed: write evidence for the declared inputs and
                # producer records for the upstream buffers (None when it cannot be aligned with the measured launches)
                from .provenance import producer_records as _producers
                prods = _producers(case, seed, rec.launches, seq, rng_before=rng_before)
            status = ("aligned" if prods is not None else "not aligned") if producer_records and \
                any(r.loaded for r in seq.launches) else "not run"
            if store:
                cap.update(prods=prods, trace_status=status)
        trace_status[status] += 1
        # M4: a carry-over the producer trace contradicts (an ATen write the bytes and version counters miss) is
        # withdrawn and the reference re-evaluated with the buffer re-entering as an external input
        withdrawn = _trace_conflicts(rec.launches, seq, prods) if prods else set()
        if withdrawn:
            t_phase = time.time()
            calls0 = sum(iv.PRECISION_DEPENDENT_CALLS.values())
            with iv.working_precision(precision_level):
                seq = evaluate_sequence(rec.launches, mode=program_reference_mode, masked_fill_zero=fill,
                                        force_external=withdrawn)
            precision_calls += sum(iv.PRECISION_DEPENDENT_CALLS.values()) - calls0
            timing["reference"] += time.time() - t_phase
        decisions = [] if seed == list(dev)[0] else None
        mixed_sources = torch_intermediates(
            rec.launches, seq, inp, snap, prods, decisions,
            include_non_floating=getattr(case, "contract_reference", None) is not None)
        if decisions is not None:
            from .provenance import ENCLOSED_LAUNCHES
            provenance_seed0 = {"reads": decisions, "carry_overs": seq.carry_overs,
                                "external_reentries": seq.external_writes,
                                "carry_overs_withdrawn_by_trace": sorted([list(x) for x in withdrawn]),
                                "dispatch_calls_with_untraced_internals":
                                    (prods or {}).get(ENCLOSED_LAUNCHES, [])}
        if external is None:
            # a buffer changed between recorded launches (a torch op in between): from there on its captured value
            # re-enters as an exact input, so K_R downstream carries the upstream numerical error of K
            external = [{"launch": e["launch"], "kernel": rec.launches[e["launch"]].kernel_name, "buffer": e["buffer"]}
                        for e in seq.external_writes]
        t_phase = time.time()
        if replay:
            specs = cap["specs"]
            binding_summary = cap.get("contract_binding")
        else:
            if getattr(case, "contract_reference", None) is not None:
                specs, binding_summary = case.bound_spec(
                    inp, outs, rec.launches, snap,
                    binding_context=contract_binding_context(outs, mixed_sources, prods))
            else:
                with snap.pre_call_state():
                    specs = case.spec(inp)
                binding_summary = None
            if store:
                cap["specs"] = specs
                cap["contract_binding"] = binding_summary
        if binding_summary is not None:
            contract_bindings.append({"seed": seed, **binding_summary})
            # K-f is a statement about the public call result, so it covers the complete logical output.  The
            # K-G_pi decomposition below remains limited to coordinates whose Triton writes the interpreter can
            # establish.  Keeping the two coordinate sets apart prevents a partially written output from hiding a
            # task-level difference in an unwritten or non-Triton-produced coordinate.
            for name, osnap in out_snaps.items():
                candidate = osnap.view.detach()
                k_full = _contract_candidate_values(candidate, name)
                f_lo_full, f_hi_full = (np.asarray(a, dtype=np.float64).reshape(-1) for a in specs[name])
                if f_lo_full.shape != k_full.shape or f_hi_full.shape != k_full.shape:
                    raise ValueError(f"contract output {name!r} shape differs from its task reference")
                t_lo_full, t_hi_full = residual_interval(k_full, f_lo_full, f_hi_full)
                ok_full = np.isfinite(k_full) & np.isfinite(f_lo_full) & np.isfinite(f_hi_full)
                task_totals.setdefault(name, []).append({
                    "seed": seed, "t": (t_lo_full, t_hi_full), "f_lo": f_lo_full, "f_hi": f_hi_full,
                    "f": 0.5 * (f_lo_full + f_hi_full), "k": k_full, "k_reps": [], "ok_total": ok_full,
                    "shape": tuple(candidate.shape)})
                ob = (observe or {}).get(name)
                if ob and ob.get("layer") == "increment":
                    from .reference_eval.observation import floor_of, increment_unit
                    old = snap.pre_call_values(ob["old"])
                    if old.size != k_full.size:
                        raise ValueError(f"observe.{name}.old = {ob['old']}: {old.size} elements, the output has "
                                         f"{k_full.size}")
                    task_totals[name][-1]["increment_total"] = {
                        "seed": seed, **increment_unit(k_full, f_lo_full, f_hi_full, old, ok_full,
                                                       floor_of(ob.get("floor"), old))}
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
        for name, osnap in out_snaps.items():
            # values from the snapshot taken at the end of the measured call; identity from that moment too
            ptr, out = osnap.ptr, osnap.view
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
                if osnap.cdata not in ids:
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
            final = torch.empty(0, dtype=out.dtype).set_(out.untyped_storage()).reshape(-1)
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
            mixed_by_output.setdefault(name, set()).update(mixed_sources.get(ptr, []))
            n_lo, n_hi = residual_interval(k, r_lo, r_hi)
            s_lo, s_hi = iv.isub(r_lo, r_hi, f_lo, f_hi) if has_f else (None, None)
            t_lo, t_hi = residual_interval(k, f_lo, f_hi) if has_f else (None, None)
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

            # proof status (audit F03): an element that holds only under an unproven premise is not ok
            prem = np.asarray(buf.premise[m], dtype=bool) if buf.premise is not None else np.zeros(idx.size, bool)
            ok_e = (buf.st[m] == ST_OK) & ~buf.cond[m] & np.isfinite(f_lo) & ~prem
            ok_total = inside & np.isfinite(k_arr) & np.isfinite(f_lo) & np.isfinite(f_hi)
            if has_f:
                special_agree = decided & (f_cls > 0) & (kr_cls == f_cls) & ~prem
            else:
                special_agree = decided & (kr_cls > 0) & (k_cls == kr_cls) & ~prem
            under_premise = prem & (buf.st[m] <= ST_NINF) & ~buf.cond[m]
            # integer set targets (L_E): their width is the target's, no working precision shrinks it
            set_t = np.asarray(buf.iset[m], dtype=bool) if buf.iset is not None else np.zeros(idx.size, bool)
            set_w = np.where(set_t, (buf.iset_hi[m] - buf.iset_lo[m]).astype(np.float64), 0.0) if set_t.any() \
                else np.zeros(idx.size)
            # elements whose last write is a float atomic update: only there may repeated launches differ by the
            # order of the updates (audit F07)
            atomic_w = np.asarray(buf.float_atomic[m], dtype=bool) if buf.float_atomic is not None else \
                np.zeros(idx.size, bool)
            if keep is not None:
                keep.setdefault(name, []).append({"seed": seed, "r_lo": frame(r_lo), "r_hi": frame(r_hi),
                                                  "k": frame(np.asarray(k, dtype=np.float64)), "ok": frame(ok_e, False),
                                                  "premise": frame(under_premise, False), "shape": tuple(out.shape),
                                                  "set_target": frame(set_t, False), "set_width": frame(set_w)})
            per.setdefault(name, []).append({
                "seed": seed, "r_lo": frame(r_lo), "r_hi": frame(r_hi), "k_reps": [], "repeat_inputs_differ": False,
                "n": (frame(n_lo), frame(n_hi)), "s": (frame(s_lo), frame(s_hi)) if has_f else None,
                "t": (frame(t_lo), frame(t_hi)) if has_f else None,
                "f_lo": frame(f_lo), "f_hi": frame(f_hi), "f": frame(0.5 * (f_lo + f_hi)),
                "kr": frame(0.5 * (r_lo + r_hi)), "k": frame(np.asarray(k, dtype=np.float64)),
                "ok": frame(ok_e, False), "ok_total": frame(ok_total, False),
                "written": frame(np.ones(idx.size, dtype=bool), False),
                "resolved": frame(ok_e | special_agree, False), "inside": np.ones(nv, dtype=bool), "idx": pos,
                "pos": pos, "shape": tuple(out.shape), "width": frame(r_hi - r_lo), "reasons": reasons,
                "aborted": aborted, "premise": frame(under_premise, False), "atomic_written": frame(atomic_w, False)})
            ob = (observe or {}).get(name)
            if ob and ob.get("layer") == "increment":
                from .reference_eval.observation import floor_of, increment_unit
                old = snap.pre_call_values(ob["old"])
                if old.size != nv:
                    raise ValueError(f"observe.{name}.old = {ob['old']}: {old.size} elements, the output has {nv}")
                row = per[name][-1]
                row["old"] = old
                fl = floor_of(ob.get("floor"), old)
                row["increment"] = {"seed": seed, **increment_unit(row["k"], row["r_lo"], row["r_hi"], old, row["ok"],
                                                                   fl)}
                if has_f:
                    # the decomposition in the increment frame: total (K against f), semantic (K_R against f); the
                    # numerical part (K against K_R) is row["increment"]
                    fr_lo, fr_hi = frame(f_lo), frame(f_hi)
                    row["increment_total"] = {"seed": seed, **increment_unit(row["k"], fr_lo, fr_hi, old, row["ok"], fl)}
                    row["increment_semantic"] = {"seed": seed, **increment_unit(row["kr"], fr_lo, fr_hi, old, row["ok"],
                                                                                fl)}
        # repeated launches of the same input, outside the recorder: K only
        if r_exec is None:
            r_exec = int(repeats) if repeats is not None else (
                8 if any(li.get("float_atomics") for li in (launch_info or [])) else 2)
        mine = {name: rows[-1] for name, rows in per.items() if rows and rows[-1]["seed"] == seed}
        task_mine = {name: rows[-1] for name, rows in task_totals.items() if rows and rows[-1]["seed"] == seed}
        t_phase = time.time()
        if replay:
            reps = cap["reps"]
        else:
            reps = []
            for _ in range(max(0, r_exec - 1)):
                inp_r = case.inputs(seed)
                same_input = InputSnapshot.digests_of(inp_r) == snap.digest_map
                outs_r = case.launch(inp_r)
                torch.cuda.synchronize()
                reps.append((same_input, {k: v.detach().clone() for k, v in outs_r.items()
                                          if torch.is_tensor(v)}))
            if store:
                cap["reps"] = reps
        for same_input, outs_r in reps:
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
            for name, row in task_mine.items():
                o = outs_r.get(name)
                if o is None or tuple(o.shape) != row["shape"]:
                    continue
                v = o.detach()
                try:
                    row["k_reps"].append(_contract_candidate_values(v.cpu(), name))
                except ValueError:
                    continue
        timing["repeat_launches"] += time.time() - t_phase
    seconds = time.time() - t0
    n_dev = len(list(dev))
    admitted_contract = mode == "B" and len(contract_bindings) == len(list(dev) + list(conf)) and \
        all(row.get("reference_usable_for_bias_measurement") is True for row in contract_bindings)
    contract_output_kinds = dict(getattr(getattr(case, "contract_reference", None), "output_kinds", {}) or {})
    report = {"case": case.name, "mode": mode, "implementation": case.implementation,
              "specification": case.specification if mode == "B" else
              "none: task semantics not checked (mode A, e_num = K - K_R only)",
              "spec_bound": case.spec_bound, "launches": launch_info, "ir_coverage_complete": coverage,
              "reference_kind": ("admitted_task_specification" if admitted_contract else
                                 "implementation_reference" if mode == "B" else None),
              "semantic_verdict": "not_established",
              "semantic_reason": ("admitted task specification and reviewed call binding; bias decisions are in the "
                                  "per-output statistical queries" if admitted_contract else
                                  "supplied reference has no admitted contract and complete kernel binding"
                                  if mode == "B" else "no independent task specification was supplied"),
              "contract_bindings": contract_bindings,
              "program_reference_mode": program_reference_mode,
              "ir_kinds": [li["ir_kind"] for li in (launch_info or [])],
              # kept for the TTIR-only scripts: the coverage of TTIR launches, None for a launch analysed on TTGIR
              "ttir_coverage_complete": [c if li["ir_kind"] == "ttir" else None
                                         for c, li in zip(coverage or [], launch_info or [])],
              "versions": {k: _version(k) for k in ("liger-kernel", "transformers", "torch", "triton")},
              "seeds": {"development": [list(dev)[0], list(dev)[-1]], "confirmation": [list(conf)[0], list(conf)[-1]]},
              "seconds": round(seconds, 1), "outputs": {},
              "tool_version": TOOL_VERSION, "launches_per_input": r_exec, "precision_level": precision_level,
              "capture": "reused" if replay else "captured",
              "precision_environment": precision_env, "analysis_interventions": ANALYSIS_INTERVENTIONS,
              # 0: no rule on the reference path depends on the working-precision level (a higher level gives the
              # same enclosures exactly)
              "precision_dependent_calls": precision_calls,
              "outputs_not_written_by_triton": sorted(not_triton),
              "outputs_binding_not_established": sorted(binding_unconfirmed),
              "outputs_at_address_of_another_recorded_storage": sorted(reused_address),
              "outputs_modified_after_last_triton_write": sorted(modified_after),
              "outputs_whose_writing_programs_aborted": aborted_outputs,
              "external_reentries_seed0": external,
              # memory-effect evidence (storage_effects M1-M5): per float read of seed 0 the decision and its basis,
              # the carried-over buffers with their evidence, and the producer trace per seed
              "provenance_seed0": provenance_seed0,
              "producer_trace_per_seed": dict(trace_status),
              "masked_lane_assumption": {"applied": fill, "ptx_zero_fills_per_launch": zero_fill,
                                         "meaning": "masked-off lanes of loads without `other` taken as 0 (the "
                                                    "lowering zero-initializes them; TTIR leaves them undefined)"}}
    for name, rows in per.items():
        ok = np.stack([p["ok"] for p in rows])
        ok_total = np.stack([p["ok_total"] for p in rows])
        kr = np.stack([p["kr"] for p in rows])
        f_mid = np.stack([p["f"] for p in rows])
        written = np.stack([p["written"] for p in rows])
        resolved = np.stack([p["resolved"] for p in rows])
        premised = np.stack([p["premise"] for p in rows])
        premises = sorted({str(r).split("@")[0] for p in rows for r in (p["reasons"] or {}) if unproven_premise(r)})
        entry = {"elements_per_seed": int(written[0].sum()), "shape": list(rows[0]["shape"]),
                 "depends_on_non_triton_intermediates": sorted(d for d in mixed_by_output.get(name, ())
                                                               if PRODUCER_RECORD not in d),
                 # upstream buffers whose recorded producers are copies of the declared inputs or exact constants
                 "upstream_with_producer_record": sorted(d for d in mixed_by_output.get(name, ())
                                                         if PRODUCER_RECORD in d),
                 # written elements whose reference is complete and finite, or a special value of f's class
                 "reference_classes": {"complete_fraction": float(resolved[written].mean()) if written.any() else 0.0,
                                       "finite_complete_fraction": float(ok[written].mean()) if written.any() else 0.0,
                                       "complete_under_premise_fraction":
                                           float(premised[written].mean()) if written.any() else 0.0,
                                       "written_fraction": float(written.mean())},
                 # proof status (audit F03): proved / holds only under an unproven premise / not established
                 "proof_status": {"unproven_premises": premises,
                                  "proved_by_certificate": sorted({str(r).split("@")[0][:200] for p in rows
                                                                   for r in (p["reasons"] or {})
                                                                   if str(r).startswith("proved:")}),
                                  "meaning": "complete_fraction counts proved references only; elements that hold "
                                             "only under an unproven premise are counted apart and are not used by "
                                             "the statistics"},
                 "special_values": special.get(name),
                 "not_established_reasons_seed0": rows[0]["reasons"],
                 "aborted_programs_seed0": rows[0]["aborted"]}
        if admitted_contract:
            entry["comparison_kind"] = ("exact_boolean" if contract_output_kinds.get(name) == "boolean"
                                        else "numeric")
            if contract_output_kinds.get(name) == "boolean":
                entry["boolean_task_comparison"] = _boolean_task_comparison(task_totals[name])
        # rc3 02 5.3 / 9: a set target (L_E, e.g. a contended atomic's returned value) encloses every allowed
        # reference choice; its width does not shrink with precision (DSL v2 increment 4)
        sets = sorted({str(r).split("@")[0] for p in rows for r in (p["reasons"] or {}) if str(r).startswith("set:")})
        entry["guarantee"] = {"fragment": "L_E in part (set targets)" if sets else "point (L_Q / L_C)",
                              "set_targets": sets}
        execution = _execution_status(rows, launch_info, r_exec)
        entry["execution"] = execution
        task_decomposition_admitted = admitted_contract and not entry["depends_on_non_triton_intermediates"]
        ob = (observe or {}).get(name)
        if ob and ob.get("layer") == "increment":
            from .reference_eval.observation import summarize_increment
            units = [p["increment"] for p in rows]
            entry["observation"] = {"layer": "increment", "old": ob["old"], "floor": ob.get("floor"),
                                    "directions": "R2 / R3 follow the reference increment (positive R3: the written "
                                                  "increment is scaled down against it)",
                                    "summary": summarize_increment(units), "per_unit": units}
            if mode == "B":
                if admitted_contract:
                    total_meaning = "total = K against the admitted task specification"
                    semantic_meaning = ("legacy field semantic = K_R against that task specification"
                                        if task_decomposition_admitted else
                                        "legacy field semantic is only a kernel-level reference difference because "
                                        "the decomposition contains non-Triton intermediates")
                else:
                    total_meaning = "total = K against the supplied reference"
                    semantic_meaning = ("legacy field semantic = K_R against that reference; this is not a semantic "
                                        "verdict")
                entry["observation"]["decomposition"] = {
                    "meaning": f"increment frame: numerical = K against K_R (summary above), {total_meaning}, "
                               f"{semantic_meaning}",
                    "total": summarize_increment([p["increment_total"] for p in
                                                  (task_totals.get(name, rows) if admitted_contract else rows)]),
                    "semantic": summarize_increment([p["increment_semantic"] for p in rows]),
                    "per_unit_total": [p["increment_total"] for p in
                                       (task_totals.get(name, rows) if admitted_contract else rows)],
                    "per_unit_semantic": [p["increment_semantic"] for p in rows]}
            kr = kr - np.stack([p["old"] for p in rows])     # the decision layer's reference frame: the increment
        comparisons = [("n", "e_num = K - K_R")]
        if mode == "B":
            comparisons.extend((("s", "G_pi - admitted task specification" if task_decomposition_admitted else
                                        "reference_difference = K_R - supplied reference"),
                                ("t", "K - admitted task specification" if admitted_contract else
                                        "total_difference = K - supplied reference")))
        for key, label in comparisons:
            comparison_rows = task_totals.get(name, rows) if key == "t" and admitted_contract else rows
            lo = np.stack([p[key][0] for p in comparison_rows])
            hi = np.stack([p[key][1] for p in comparison_rows])
            valid = ok_total if key == "t" else ok
            if key == "t" and admitted_contract:
                valid = np.stack([p["ok_total"] for p in comparison_rows])
                alignment = np.stack([p["f"] for p in comparison_rows])
            else:
                alignment = f_mid if key == "t" else kr
            if key == "n" and execution["statistics"] == "within-input mean":
                lo, hi = _within_input_mean_residual(rows)
            elif key == "t" and execution["statistics"] == "within-input mean":
                lo, hi = _within_input_mean_total(comparison_rows)
            t_phase = time.time()
            task_direct = key == "t" and admitted_contract
            if execution["statistics"] == "withheld" and not task_direct:
                rec = {"comparison": f"{name}: {label}", "verdict": "NOT_ESTABLISHED", "reason": execution["status"]}
            else:
                rec, _ = assess_units(f"{name}: {label}", lo, hi, alignment, valid, n_dev, RULES, alpha=alpha,
                                      alignment_reference=alignment, unit_ids=list(dev) + list(conf),
                                      equivalence_rel=equivalence_rel,
                                      magnitude_bound=magnitude_bound if key == "n" else None)
                if equivalence_rel is None:
                    rec["equivalence_scale"] = {"delta_rel": None, "meaning": "no equivalence bound declared: no "
                                                "equivalence statement"}
                if task_direct:
                    rec["execution_basis"] = (
                        "one observed public call per independent input; repeated launches are averaged within the "
                        "input only for identified float-atomic order variation")
                    if execution["statistics"] == "withheld":
                        rec["decomposition_execution_note"] = execution["status"]
            timing["statistics"] += time.time() - t_phase
            mid = 0.5 * (lo + hi)
            if valid.any():
                widths = (np.stack([p["f_hi"] - p["f_lo"] for p in comparison_rows]) if key == "t" else
                          np.stack([p["width"] for p in rows]))
                rec["scale"] = {"mean_abs_residual": float(np.abs(mid[valid]).mean()),
                                "max_abs_residual": float(np.abs(mid[valid]).max()),
                                "rms_reference": float(np.sqrt((alignment[valid] ** 2).mean())),
                                "relative_rms": float(np.sqrt((mid[valid] ** 2).mean() /
                                                                max((alignment[valid] ** 2).mean(), 1e-300))),
                                "max_reference_width": float(widths[valid].max())}
                if key != "t":
                    rec["scale"]["max_K_R_width"] = rec["scale"]["max_reference_width"]
            destination = "numerical" if key == "n" else "semantic" if key == "s" else "total"
            entry[destination] = rec
            if key == "s":
                rec["verdict_scope"] = ("G_pi_minus_admitted_task_specification"
                                        if task_decomposition_admitted else
                                        "reference_difference; not a semantic verdict")
            elif key == "t":
                rec["verdict_scope"] = ("candidate_minus_admitted_task_specification"
                                        if admitted_contract else
                                        "candidate versus supplied reference; not a semantic verdict")
                if rec.get("scale"):
                    rec["relative_rms"] = rec["scale"]["relative_rms"]
                    rec["max_abs"] = rec["scale"]["max_abs_residual"]
                if admitted_contract:
                    entry["task_comparison_elements_per_seed"] = int(valid.shape[1])
                    entry["task_comparison_covers_complete_public_output"] = True
                    entry["task_special_values"] = _contract_special_values(comparison_rows)
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
                "relative_rms": float(np.sqrt((mid_all ** 2).mean() / max((kr[ok] ** 2).mean(), 1e-300))),
                "verdict_scope": ("G_pi_minus_admitted_task_specification"
                                  if task_decomposition_admitted else
                                  "reference_difference; not a semantic verdict")}
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
        entry["task_semantics"] = ("not checked (no specification)" if mode == "A" else
                                   "candidate checked against an admitted task specification and reviewed call binding; "
                                   "G_pi decomposition also has call-level inputs"
                                   if task_decomposition_admitted else
                                   "candidate checked against an admitted task specification and reviewed call binding; "
                                   "G_pi decomposition includes non-Triton intermediates"
                                   if admitted_contract else
                                   "not established (supplied reference is not an admitted contract binding)")
        report["outputs"][name] = entry
    # A reviewed public-call binding can cover an output even when its value is not written by a recorded Triton
    # launch.  Preserve the direct K-f result, while stating plainly that no K-G_pi decomposition was established.
    if admitted_contract:
        for name, rows in task_totals.items():
            if name in report["outputs"]:
                continue
            lo = np.stack([p["t"][0] for p in rows])
            hi = np.stack([p["t"][1] for p in rows])
            valid = np.stack([p["ok_total"] for p in rows])
            alignment = np.stack([p["f"] for p in rows])
            t_phase = time.time()
            rec, _ = assess_units(f"{name}: K - admitted task specification", lo, hi, alignment, valid, n_dev,
                                  RULES, alpha=alpha, alignment_reference=alignment,
                                  unit_ids=list(dev) + list(conf), equivalence_rel=equivalence_rel)
            if equivalence_rel is None:
                rec["equivalence_scale"] = {"delta_rel": None, "meaning": "no equivalence bound declared: no "
                                            "equivalence statement"}
            rec["execution_basis"] = "one observed public call per independent input"
            timing["statistics"] += time.time() - t_phase
            mid = 0.5 * (lo + hi)
            if valid.any():
                widths = np.stack([p["f_hi"] - p["f_lo"] for p in rows])
                rec["scale"] = {"mean_abs_residual": float(np.abs(mid[valid]).mean()),
                                "max_abs_residual": float(np.abs(mid[valid]).max()),
                                "rms_reference": float(np.sqrt((alignment[valid] ** 2).mean())),
                                "relative_rms": float(np.sqrt((mid[valid] ** 2).mean() /
                                                                max((alignment[valid] ** 2).mean(), 1e-300))),
                                "max_reference_width": float(widths[valid].max())}
                rec["relative_rms"] = rec["scale"]["relative_rms"]
                rec["max_abs"] = rec["scale"]["max_abs_residual"]
            rec["verdict_scope"] = "candidate_minus_admitted_task_specification"
            report["outputs"][name] = {
                "elements_per_seed": 0, "shape": list(rows[0]["shape"]),
                "task_comparison_elements_per_seed": int(valid.shape[1]),
                "task_comparison_covers_complete_public_output": True,
                "task_special_values": _contract_special_values(rows),
                "numerical": {"verdict": "NOT_ESTABLISHED",
                              "reason": "no recorded Triton write was bound to this public output"},
                "total": rec, "task_semantics": "candidate checked against an admitted task specification and "
                                                   "reviewed call binding; K-G_pi not established"}
            if contract_output_kinds.get(name) == "boolean":
                report["outputs"][name]["comparison_kind"] = "exact_boolean"
                report["outputs"][name]["boolean_task_comparison"] = _boolean_task_comparison(rows)
    # cost by phase (evaluation plan, work item H): compile / warm-up, input generation, capture (the launches under
    # the recorder), reference (TTIR evaluation), specification, statistics; "seconds" keeps its old meaning
    report["timing_seconds"] = {k: round(v, 3) for k, v in timing.items()}
    return report


# Execution validity findings of the reference (audit F07), by producer (ttir_eval): a race found by the reference;
# validity not established (the reference could not decide whether the accesses are ordered).  Both stop every
# statistic that depends on the execution being valid.
RACE_MARKERS = ("execution race",                     # _run_programs, _same_program_write_read, _op_atomic_cas
                "cross-program write race",           # _op_store
                "cross-program race on load",         # _op_load
                "conflicting lanes in one store")     # _op_store: one store writes an address from several lanes
UNKNOWN_VALIDITY_MARKERS = ("execution validity",     # _same_program_write_read: thread mapping not available
                            "progress not proven")     # atomic_poll / spin loops: termination not shown


def execution_findings(reasons) -> dict:
    """{"race": [...], "unknown_validity": [...]}: the execution-validity findings among the reference's reasons."""
    keys = sorted({str(r).split("@")[0] for r in (reasons or {})})
    return {"race": [r for r in keys if any(m in r for m in RACE_MARKERS)],
            "unknown_validity": [r for r in keys if any(m in r for m in UNKNOWN_VALIDITY_MARKERS)]}


def _execution_status(rows, launch_info, r_exec) -> dict:
    """Execution validity of one output over the units (DSL v2 rc3 02 8.7): race and validity findings of the
    reference, and repeated launches.  Statistics run per launch only when no finding questions the execution and the
    repeats are bitwise identical; repeats that differ only on elements written by float atomics are averaged within
    the input (order of the updates); anything else withholds them (audit F07)."""
    race, unknown = set(), set()
    for p in rows:
        f = execution_findings(p["reasons"])
        race.update(f["race"])
        unknown.update(f["unknown_validity"])
    race, unknown = sorted(race), sorted(unknown)
    differ, reps_done, inputs_differ, outside_atomics = [], 0, False, False
    for p in rows:
        w = p["written"]
        reps_done = max(reps_done, len(p["k_reps"]))
        inputs_differ |= p["repeat_inputs_differ"]
        diff = np.zeros(np.shape(p["k"]), dtype=bool)
        for k in p["k_reps"]:
            if k.shape == p["k"].shape:
                diff |= w & ~((k == p["k"]) | (np.isnan(k) & np.isnan(p["k"])))
        if diff.any():
            differ.append(p["seed"])
            atomic = p.get("atomic_written")   # element-level evidence: written by float atomics
            if atomic is None or (diff & ~np.asarray(atomic, dtype=bool)).any():
                outside_atomics = True
    atomics = any(li.get("float_atomics") for li in (launch_info or []))
    out = {"launches_per_input": 1 + reps_done, "units_with_different_repeats": len(differ),
           "units": len(rows), "repeat_inputs_not_reproducible": inputs_differ, "race_findings": race,
           "unknown_validity_findings": unknown, "float_atomics": atomics,
           "differences_outside_float_atomic_elements": outside_atomics,
           "validity": "race" if race else ("not established" if unknown else "no finding")}
    if race:
        out.update(status="execution race found by the reference: statistics withheld", statistics="withheld")
    elif unknown:
        out.update(status="execution validity not established by the reference: statistics withheld",
                   statistics="withheld")
    elif differ and atomics and not inputs_differ and not outside_atomics:
        out.update(status=f"atomic execution randomness: residuals averaged within the input over {1 + reps_done} "
                          "launches", statistics="within-input mean")
    elif differ:
        why = ("; the regenerated inputs differ" if inputs_differ else
               "; they differ on elements not written by float atomics" if atomics and outside_atomics else "")
        out.update(status="repeated launches differ without an identified cause: execution validity not established "
                          "(diagnosis needed)" + why, statistics="withheld")
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


def _within_input_mean_total(rows):
    """Per unit outward enclosure of the repeated-launch mean of K - f."""
    lo_u, hi_u = [], []
    for p in rows:
        ks = [p["k"]] + [k for k in p["k_reps"] if k.shape == p["k"].shape]
        los, his = zip(*(residual_interval(k, p["f_lo"], p["f_hi"]) for k in ks))
        okm = p["ok_total"]
        L = np.where(okm, np.stack(los), 0.0)
        H = np.where(okm, np.stack(his), 0.0)
        s_lo, s_hi = iv.fsum_bounds(L, H, axis=0)
        n = float(len(ks))
        lo_u.append(iv.div_bounds(s_lo, n)[0])
        hi_u.append(iv.div_bounds(s_hi, n)[1])
    return np.stack(lo_u), np.stack(hi_u)


def run_black_box(case, dev=DEV, conf=CONF, equivalence_rel=None, observe=None, alpha=0.05, magnitude_bound=None):
    """Black-box mode (evaluation plan, work item C) for implementations without TTIR (NumPy, CUDA, cuBLAS): the
    specification is the only reference, so only the total K - f is measured, through the same decision layer.
    No decomposition into e_num and e_sem; reported separately from the decomposed results.  ``observe``: as in
    ``run`` (the increment layer takes the old value from the inputs before the call).  The precision switches the
    candidate runs under are recorded, never set."""
    timing = {"setup_compile_warmup": 0.0, "inputs": 0.0, "call": 0.0, "specification": 0.0, "statistics": 0.0}
    t_phase = time.time()
    case.setup()
    timing["setup_compile_warmup"] = time.time() - t_phase
    rows, special, incs, contract_bindings = {}, {}, {}, []
    precision_env = precision_environment()
    t0 = time.time()
    for seed in list(dev) + list(conf):
        t_phase = time.time()
        inp = case.inputs(seed)
        timing["inputs"] += time.time() - t_phase
        snap = InputSnapshot(inp)
        t_phase = time.time()
        outs = case.launch(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing["call"] += time.time() - t_phase
        t_phase = time.time()
        if getattr(case, "contract_reference", None) is not None:
            specs, binding_summary = case.bound_spec(inp, outs, [], snap)
            contract_bindings.append({"seed": seed, **binding_summary})
        else:
            with snap.pre_call_state():
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
            ob = (observe or {}).get(name)
            ref_mid = np.where(f_fin, 0.5 * (f_lo + f_hi), 0.0)
            if ob and ob.get("layer") == "increment":
                from .reference_eval.observation import floor_of, increment_unit
                old = snap.pre_call_values(ob["old"])
                incs.setdefault(name, []).append({"seed": seed, **increment_unit(
                    k, np.where(f_fin, f_lo, 0.0), np.where(f_fin, f_hi, 0.0), old, ok, floor_of(ob.get("floor"), old))})
                ref_mid = ref_mid - old                  # directions follow the reference increment
            rows.setdefault(name, []).append((np.where(ok, lo, 0.0), np.where(ok, hi, 0.0),
                                              ref_mid, ok, tuple(np.shape(out))))
    n_dev = len(list(dev))
    admitted_contract = len(contract_bindings) == len(list(dev) + list(conf)) and \
        all(row.get("reference_usable_for_bias_measurement") is True for row in contract_bindings)
    report = {"case": case.name, "mode": "black-box", "implementation": case.implementation,
              "specification": case.specification, "spec_bound": case.spec_bound,
              "reference_kind": ("admitted_task_specification" if admitted_contract else
                                 "implementation_reference"), "semantic_verdict": "not_established",
              "semantic_reason": ("admitted task specification and reviewed call binding; bias decisions are in the "
                                  "per-output statistical queries" if admitted_contract else
                                  "supplied reference has no admitted contract and complete kernel binding"),
              "contract_bindings": contract_bindings,
              "comparison": ("K - admitted task specification only (no TTIR: no K_R decomposition)"
                             if admitted_contract else
                             "K - supplied reference only (no TTIR: no K_R, no decomposition)"),
              "seeds": {"development": [list(dev)[0], list(dev)[-1]], "confirmation": [list(conf)[0], list(conf)[-1]]},
              "seconds": round(time.time() - t0, 1), "outputs": {}}
    for name, r in rows.items():
        lo, hi, fm, ok = (np.stack([x[i] for x in r]) for i in range(4))
        t_phase = time.time()
        rec, _ = assess_units(f"{name}: K - f (black box)", lo, hi, fm, ok, n_dev, RULES, alignment_reference=fm,
                              unit_ids=list(dev) + list(conf), equivalence_rel=equivalence_rel, alpha=alpha,
                              magnitude_bound=magnitude_bound)
        timing["statistics"] += time.time() - t_phase
        mid = 0.5 * (lo + hi)
        if ok.any():
            rec["scale"] = {"mean_abs_residual": float(np.abs(mid[ok]).mean()),
                            "max_abs_residual": float(np.abs(mid[ok]).max()),
                            "rms_reference": float(np.sqrt((fm[ok] ** 2).mean())),
                            "relative_rms": float(np.sqrt((mid[ok] ** 2).mean() / max((fm[ok] ** 2).mean(), 1e-300)))}
        rec["verdict_scope"] = ("candidate_minus_admitted_task_specification"
                                if admitted_contract else "reference_difference; not a semantic verdict")
        report["outputs"][name] = {"elements_per_seed": int(ok.shape[1]), "shape": list(r[0][4]),
                                   "finite_fraction": float(ok.mean()), "total_black_box": rec,
                                   "special_values": special[name],
                                   "task_semantics": ("checked against an admitted task specification and reviewed "
                                                      "call binding" if admitted_contract else
                                                      "not established (supplied reference is not an admitted "
                                                      "contract binding)")}
        if name in incs:
            from .reference_eval.observation import summarize_increment
            ob = observe[name]
            report["outputs"][name]["observation"] = {
                "layer": "increment", "old": ob["old"], "floor": ob.get("floor"),
                "directions": "R2 / R3 follow the reference increment f - old (positive R3: the written increment is "
                              "scaled down against it)",
                "summary": summarize_increment(incs[name]), "per_unit": incs[name]}
    report["precision_environment"] = precision_env
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
