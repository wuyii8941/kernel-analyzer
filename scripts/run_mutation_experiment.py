#!/usr/bin/env python3
"""Real-kernel mutation experiment (round-2 item 2).

For each supported kernel and each declared single-site mutation, the
original and the mutant run on identical inputs and the tool reports:

* instruction-processing coverage of the mutant (unsupported mutants counted, not dropped);
* whether the declared semantics is preserved (reference intervals of the
  original and the mutant intersect on every element);
* elements whose device value changed, and the residual of the mutant against
  the original's reference (the specification f = the original's declared
  computation) and against its own reference;
* the statement fixed in advance (local only), checked: directed rounding at
  the final operation fixes the residual sign; an omitted term is detected
  with the specification and is invisible without it; negative controls
  (rerun, bitwise-equivalent rewrite) give identical values and references;
* localization without being told the site: (a) the TTIR operations that
  differ between the two captured programs, (b) from numerical evidence only,
  the changed output elements and the size of their dependency cone in the
  original TTIR.  Region sizes are reported.

Whether the average effect is zero is not part of any expectation.

    python scripts/run_mutation_experiment.py --out results/reference_eval/mutation_experiment.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, load_launch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, decode_storage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

# kind -> statement fixed in advance
KNOWN = {
    "rounding_down_final": "the mutated operation is the last before the store and its inputs are exact "
                           "(captured operands): K - K_R(own) <= 0 on every element",
    "rounding_up_final": "the mutated operation is the last before the store and its inputs are exact "
                         "(captured operands): K - K_R(own) >= 0 on every element",
    "rounding_down_local": "local residual of the mutated operation <= 0 on its actual inputs; its inputs carry "
                           "upstream rounding, so the output residual sign is not determined (not checked)",
    "precision": "local residual of the cast = RN_bf16(t) - t on the given input; declared semantics unchanged",
    "order": "declared semantics unchanged",
    "approximation": "declared semantics unchanged",
    "approximation_exact_divisor": "divisor is a power of two (n_cols = 512): approximate and correctly rounded "
                                   "division agree exactly; expected no numeric difference, IR difference only",
    "reformulation": "same real value, different expression: reference intervals intersect",
    "omission": "declared semantics changes: detected with the specification, invisible without it",
    "rerun": "negative control: identical values and identical references",
    "equivalent": "negative control: identical values and identical references",
    "constant_change": "declared expression changes (rounded constant): shows in K_R - f",
}


def own_kernel_specs():
    import torch

    from scripts import mutation_kernels as mk

    g = lambda seed: torch.Generator(device="cuda").manual_seed(seed)  # noqa: E731
    n = 4096

    def scale(mut):
        x = torch.randn(n, device="cuda", generator=g(1))
        return lambda: mk.m_scale[(n // 256,)](x, torch.empty_like(x), 0.1, n, BLOCK=256, MUT=mut)

    def sum4(mut):
        a, b, c, d = (torch.randn(n, device="cuda", generator=g(s)) * 10.0 ** (s % 3) for s in (2, 3, 4, 5))
        return lambda: mk.m_sum4[(n // 256,)](a, b, c, d, torch.empty_like(a), n, BLOCK=256, MUT=mut)

    def row_sum(mut):
        x = torch.randn(64, 300, device="cuda", generator=g(6))
        return lambda: mk.m_row_sum[(64,)](x, torch.empty(64, device="cuda"), 300, BLOCK=512, MUT=mut)

    def softmax(mut):
        x = torch.randn(32, 200, device="cuda", generator=g(7)) * 3.0
        return lambda: mk.m_softmax[(32,)](x, torch.empty_like(x), 200, 200, BLOCK=256, MUT=mut)

    def layernorm(mut):
        x = torch.randn(32, 200, device="cuda", generator=g(8)) + 0.5
        w = torch.randn(200, device="cuda", generator=g(9))
        b = torch.randn(200, device="cuda", generator=g(10))
        return lambda: mk.m_layernorm[(32,)](x, w, b, torch.empty_like(x), 200, 1e-5, BLOCK=256, MUT=mut)

    def matmul(mut):
        A = torch.randn(64, 64, device="cuda", generator=g(11))
        B = torch.randn(64, 32, device="cuda", generator=g(12))
        return lambda: mk.m_matmul[(2, 2)](A, B, torch.empty(64, 32, device="cuda"), 64, 32, 64,
                                          BM=32, BN=16, BK=16, MUT=mut)

    def accumulate(mut):
        acc = torch.randn(n, device="cuda", generator=g(13)) * 100.0
        c = torch.randn(n, device="cuda", generator=g(14))
        return lambda: mk.m_accumulate[(n // 256,)](acc, c, n, BLOCK=256, MUT=mut)

    return {
        "scale": (scale, "Y", {1: "rounding_down_final", 2: "rounding_up_final", 3: "precision", 4: "equivalent"}),
        "sum4": (sum4, "Y", {1: "order", 2: "rounding_down_local", 3: "precision", 4: "omission", 5: "equivalent"}),
        "row_sum": (row_sum, "Y", {1: "order", 2: "precision", 3: "omission", 4: "equivalent"}),
        "softmax": (softmax, "Y", {1: "approximation", 2: "precision", 3: "reformulation"}),
        "layernorm": (layernorm, "Y", {1: "approximation", 2: "reformulation", 3: "precision"}),
        "matmul": (matmul, "C", {1: "approximation", 2: "precision", 3: "order", 4: "omission"}),
        "accumulate": (accumulate, "ACC", {1: "rounding_down_final", 2: "rounding_up_final", 3: "equivalent"}),
    }


def capture(fn):
    import torch

    recorder = TritonLaunchRecorder()
    with recorder:
        fn()
        torch.cuda.synchronize()
    return recorder.launches[-1]


def evaluate(launch):
    module = parse_ttir(launch.asm["ttir"])
    return module, KernelReferenceEvaluator(module).evaluate(launch)


def buffer(result, name):
    return next(b for b in result.buffers.values() if b.name == name)


def op_signature(op):
    attrs = {k: v for k, v in op.attrs.items() if k not in ("iv_type",)}
    return op.name + "|" + json.dumps(attrs, sort_keys=True, default=str)


COMMUTATIVE = {"arith.addf", "arith.mulf", "arith.addi", "arith.muli", "arith.maxnumf", "arith.minnumf",
               "arith.maximumf", "arith.minimumf", "arith.andi", "arith.ori", "arith.xori", "arith.maxsi",
               "arith.minsi", "arith.maxui", "arith.minui"}


def structural_hashes(module):
    """Hash of every op from its name, attributes and the hashes of its operands' producers."""

    import hashlib

    func = module.entry()
    params = {p[0]: f"param{i}" for i, p in enumerate(func.params)}
    value_hash = dict(params)
    out = []

    def visit(region, depth):
        for block in region.blocks:
            for i, (name, _) in enumerate(block.args):
                value_hash[name] = f"arg{depth}.{i}"
            for op in block.ops:
                operands = [value_hash.get(v, "?") for v in op.operands]
                if op.name in COMMUTATIVE:  # IEEE add/mul/min/max are bitwise commutative
                    operands = sorted(operands)
                key = op_signature(op) + "(" + ",".join(operands) + ")"
                h = hashlib.sha1(key.encode()).hexdigest()[:16]
                for j, r in enumerate(op.results):
                    value_hash[r] = f"{h}.{j}"
                out.append((h, op.name))
                for sub in op.regions:
                    visit(sub, depth + 1)

    visit(func.body, 0)
    return out


def ir_diff(m0, m1):
    """(a) instruction-mix difference; (b) structural difference (changed op and everything it feeds)."""

    c0 = collections.Counter(op_signature(o) for f in m0.funcs.values() for o in f.walk())
    c1 = collections.Counter(op_signature(o) for f in m1.funcs.values() for o in f.walk())
    removed, added = c0 - c1, c1 - c0
    h0 = collections.Counter(h for h, _ in structural_hashes(m0))
    h1 = collections.Counter(h for h, _ in structural_hashes(m1))
    names0 = dict(structural_hashes(m0))
    names1 = dict(structural_hashes(m1))
    s_removed, s_added = h0 - h1, h1 - h0
    return {"mix_removed": sum(removed.values()), "mix_added": sum(added.values()),
            "ops": sorted({s.split("|")[0] for s in list(removed) + list(added)}),
            "structural_changed_ops_original": sum(s_removed.values()),
            "structural_changed_ops_mutant": sum(s_added.values()),
            "structural_ops": sorted({names0[h] for h in s_removed} | {names1[h] for h in s_added}),
            "total_ops_original": sum(c0.values())}


def store_cones(module):
    """For every stored buffer parameter: the number of TTIR ops in its backward slice."""

    func = module.entry()
    defs, nested = {}, {}

    def index(region, parent):
        for block in region.blocks:
            for op in block.ops:
                for r in op.results:
                    defs[r] = op
                nested[id(op)] = parent
                for sub in op.regions:
                    index(sub, op)

    index(func.body, None)
    params = {p[0] for p in func.params}
    all_ops = list(func.walk())

    def slice_of(values):
        seen, stack = set(), list(values)
        while stack:
            v = stack.pop()
            op = defs.get(v)
            if op is None or id(op) in seen:
                continue
            seen.add(id(op))
            stack.extend(op.operands)
            for sub in op.regions:  # region ops bring all their nested ops
                for inner in _walk(sub):
                    if id(inner) not in seen:
                        seen.add(id(inner))
                        stack.extend(inner.operands)
            parent = nested.get(id(op))
            while parent is not None:  # control: enclosing region ops
                if id(parent) not in seen:
                    seen.add(id(parent))
                    stack.extend(parent.operands)
                parent = nested.get(id(parent))
        return seen

    def root_param(v):
        while v not in params:
            op = defs.get(v)
            if op is None or not op.operands:
                return None
            v = op.operands[0]
        return v

    cones = collections.defaultdict(set)
    for op in all_ops:
        if op.name in ("tt.store", "tt.atomic_rmw"):
            target = root_param(op.operands[0])
            cones[target] |= slice_of(op.operands) | {id(op)}
    return {k: len(v) for k, v in cones.items()}, len(all_ops)


def _walk(region):
    for block in region.blocks:
        for op in block.ops:
            yield op
            for sub in op.regions:
                yield from _walk(sub)


def sign_counts(actual, lo, hi):
    r_lo = iv.add_bounds(actual, -hi)[0]
    r_hi = iv.add_bounds(actual, -lo)[1]
    pos, neg = int((r_lo > 0).sum()), int((r_hi < 0).sum())
    return {"positive": pos, "negative": neg, "contains_zero": int(actual.size - pos - neg),
            "max_abs": float(np.max(np.maximum(np.abs(r_lo), np.abs(r_hi)))) if actual.size else 0.0,
            "mean_bounds": [float(np.mean(r_lo)), float(np.mean(r_hi))] if actual.size else None}


def analyse_pair(name, kind, launch0, launch1, out_name):
    m0, r0 = evaluate(launch0)
    m1, r1 = evaluate(launch1)
    cov = kernel_coverage(m1)
    row = {"kernel": name, "mutation_kind": kind, "known_in_advance": KNOWN[kind],
           "coverage_complete": cov["complete"], "aborted_programs": len(r1.aborted)}
    if not cov["complete"] or r1.aborted:
        row["unsupported"] = [r["rejected"] for r in cov["rejected"]][:3] or list(r1.aborted.values())[:3]
        return row
    b0, b1 = buffer(r0, out_name), buffer(r1, out_name)
    m = b0.written & b1.written
    k0, k1 = b0.actual_after[m], b1.actual_after[m]
    lo0, hi0, lo1, hi1 = b0.lo[m], b0.hi[m], b1.lo[m], b1.hi[m]
    ok = (b0.st[m] == 0) & (b1.st[m] == 0)
    k0, k1, lo0, hi0, lo1, hi1 = (a[ok] for a in (k0, k1, lo0, hi0, lo1, hi1))
    intersect = (lo1 <= hi0) & (lo0 <= hi1)
    identical_ref = bool(np.array_equal(lo0, lo1) and np.array_equal(hi0, hi1))
    changed = k1 != k0
    sem_lo = iv.add_bounds(lo1, -hi0)[0]
    sem_hi = iv.add_bounds(hi1, -lo0)[1]
    semantic_detected = int(((sem_lo > 0) | (sem_hi < 0)).sum())
    own = sign_counts(k1, lo1, hi1)
    spec = sign_counts(k1, lo0, hi0)
    orig = sign_counts(k0, lo0, hi0)
    row.update({
        "elements": int(k0.size), "elements_changed": int(changed.sum()),
        "declared_semantics_preserved": bool(intersect.all()), "references_identical": identical_ref,
        "semantic_term_detected_elements": semantic_detected,
        "residual_against_specification": spec, "residual_against_own_reference": own,
        "residual_of_original": orig,
    })
    checks = {}
    if kind == "rounding_down_final":
        checks["sign_fixed"] = bool((iv.add_bounds(k1, -lo1)[1] <= 0).all())
    elif kind == "rounding_up_final":
        checks["sign_fixed"] = bool((iv.add_bounds(k1, -hi1)[0] >= 0).all())
    elif kind == "omission":
        checks["detected_with_specification"] = semantic_detected > 0
        checks["invisible_without_specification"] = bool(own["max_abs"] <= max(orig["max_abs"], 1e-30) * 4)
    elif kind in ("rerun", "equivalent", "approximation_exact_divisor"):
        checks["values_identical"] = bool(not changed.any())
        checks["references_identical"] = identical_ref
    elif kind == "reformulation":
        checks["references_intersect"] = bool(intersect.all())
    row["checks"] = checks
    row["change_seen"] = bool(changed.any() or semantic_detected > 0)
    diff = ir_diff(m0, m1)
    cones, total = store_cones(m0)
    out_param = next((p for p in cones if p and p.lstrip("%") == out_name), None)
    row["localization"] = {
        "ir_diff": diff,
        "numeric": {"changed_output_elements": int(changed.sum()), "output_elements": int(k0.size),
                    "dependency_cone_ops": cones.get(out_param), "kernel_ops": total},
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ce", type=Path, default=ROOT / ".cache/exp_intervention")
    parser.add_argument("--rms", type=Path, default=ROOT / ".cache/rms_mutations")
    args = parser.parse_args()
    rows = []
    for name, (make, out_name, table) in own_kernel_specs().items():
        base = capture(make(0))
        rerun = capture(make(0))
        rows.append(analyse_pair(name, "rerun", base, rerun, out_name))
        for mut, kind in table.items():
            try:
                launch = capture(make(mut))
            except Exception as exc:  # compilation failures are reported, not dropped
                rows.append({"kernel": name, "mutation_kind": kind, "coverage_complete": False,
                             "unsupported": [f"compile: {type(exc).__name__}: {str(exc)[:120]}"]})
                continue
            rows.append(analyse_pair(name, kind, base, launch, out_name))
        print(name, "done", flush=True)
    if args.rms.exists():
        base = load_launch(args.rms / "original" / "launch000")
        for variant, kind in (("rerun", "rerun"), ("rsqrt_rn", "approximation"), ("div_rn", "approximation_exact_divisor"),
                              ("omit_eps", "omission")):
            row = analyse_pair("liger_rmsnorm_forward", kind, base, load_launch(args.rms / variant / "launch000"), "Y_ptr")
            row["variant"] = variant
            rows.append(row)
    if args.ce.exists():
        ce_variants = (("div_rn", "approximation"), ("div_d_rn", "approximation"), ("div_n_rn", "approximation"),
                       ("libdevice", "approximation"), ("exp2_up", "constant_change"))
        for variant, kind in ce_variants:
            agg = None
            for j in range(64):
                row = analyse_pair("liger_cross_entropy", kind, load_launch(args.ce / "original" / f"launch{j:03d}"),
                                   load_launch(args.ce / variant / f"launch{j:03d}"), "X_ptr")
                if agg is None:
                    agg = row
                    agg["launches"] = 1
                else:
                    agg["launches"] += 1
                    for key in ("elements", "elements_changed", "semantic_term_detected_elements"):
                        agg[key] += row.get(key, 0)
                    for key in ("residual_against_specification", "residual_against_own_reference", "residual_of_original"):
                        for c in ("positive", "negative", "contains_zero"):
                            agg[key][c] += row[key][c]
                    agg["declared_semantics_preserved"] &= row["declared_semantics_preserved"]
                    agg["references_identical"] &= row["references_identical"]
            agg["variant"] = variant
            rows.append(agg)
    payload = {"schema": "kernel-analyzer-mutation-experiment-v1", "reference_mode": "numerical_difference",
               "scope": "kernels whose original evaluation has complete references", "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    for r in rows:
        print(f"{r['kernel']:22s} {r.get('variant', r['mutation_kind']):16s} cov={r.get('coverage_complete')} "
              f"seen={r.get('change_seen')} sem_kept={r.get('declared_semantics_preserved')} "
              f"changed={r.get('elements_changed')}/{r.get('elements')} checks={r.get('checks')} "
              f"ir_mix={r.get('localization', {}).get('ir_diff', {}).get('ops')} "
              f"struct={r.get('localization', {}).get('ir_diff', {}).get('structural_changed_ops_original')}")


if __name__ == "__main__":
    main()
