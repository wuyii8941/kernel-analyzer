"""Plants (protocol.md section 6): one-line edits of real kernels, replayed on the captured inputs.

semantic plant: f = the original kernel's exact value G.  C reports G' - f (element-wise rule of the audit's reference
comparison) and K' - G'; B-f reports K' - f; B-G reports K' - G'.
numeric plant: a bf16 round trip inserted on a known line; C's attribution (true premises) and A's attribution
(textbook premises) each name a top site, compared with the planted line.

    python plants.py plants_spec.json plants.jsonl
spec: [{"id", "unit", "kind": "semantic"|"numeric"|"control", "line", "old", "new"}]
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _exact(L, progs):
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval import intervals as iv
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    m = parse_ttir(L.asm["ttir"])
    m.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(m, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    original = iv.elementary_bounds
    iv.elementary_bounds = BP._float64_elementary(original)
    try:
        return {b.name: b for b in ev.evaluate(L, programs=progs).buffers.values() if b.kind == "f" and b.written.any()}
    finally:
        iv.elementary_bounds = original


def _attribution_line(L, progs, output, textbook: bool):
    """Top site (largest |contribution| to the output's alignment) with its source line, under the true premises
    (C) or the textbook premises (A)."""
    import numpy as np

    from kernel_analyzer.reference_eval import emulate as E
    from kernel_analyzer.reference_eval.premises import PREMISES, textbook_emulate
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK
    from patch_replay import source_line

    run = (lambda **kw: textbook_emulate(L, premises=PREMISES, programs=progs, **kw)) if textbook else \
        (lambda **kw: E.emulate(L, programs=progs, **kw))
    base, emu = run()
    bb = next(x for x in base.buffers.values() if x.name == output)
    best, best_c = None, 0.0
    for node in E.rounding_nodes(emu):
        try:
            r, _ = run(exact_nodes=[node])
        except Exception:
            continue
        x = next(y for y in r.buffers.values() if y.name == output)
        m = bb.written & (bb.st == ST_OK) & (x.st == ST_OK)
        ref = x.lo[m]
        den = float((ref * ref).sum())
        c = float(((bb.lo[m] - x.lo[m]) * ref).sum() / den) if den > 0 else 0.0
        if abs(c) > abs(best_c):
            best, best_c = node, c
    if best is None:
        return None
    src = source_line(L, best)
    return {"node": best, "contribution": best_c, "line": src[1] if src else None, "code": src[2].strip() if src else None}


def one(spec: dict) -> dict:
    import numpy as np

    from analyze_units import _per_program_elements, select_programs
    from kernel_analyzer.reference_eval.audit import _alignment, _se
    from kernel_analyzer.reference_eval import intervals as iv
    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK
    from patch_replay import run_patched

    L = load_launch(Path(spec["unit"]))
    progs = select_programs(L.grid, _per_program_elements(L))
    out = HERE / "plants" / spec["id"]
    run_patched(spec["unit"], [[spec["line"], spec["old"], spec["new"]]], out)
    L2 = load_launch(out)
    G, G2 = _exact(L, progs), _exact(L2, progs)
    row = {"id": spec["id"], "kind": spec["kind"], "unit": spec["unit"], "edit": [spec["line"], spec["old"], spec["new"]],
           "outputs": {}}
    for name, b in G.items():
        b2 = G2.get(name)
        if b2 is None:
            continue
        ok = (b.written & b2.written & (b.st == ST_OK) & (b2.st == ST_OK) & (b.actual_after_st == 0)
              & (b2.actual_after_st == 0) & np.isfinite(b.actual_after) & np.isfinite(b2.actual_after))
        if not ok.any():
            continue
        f = 0.5 * (b.lo[ok] + b.hi[ok])                   # task reference: the original program's exact value
        g2 = 0.5 * (b2.lo[ok] + b2.hi[ok])
        h2 = 0.5 * (b2.hi[ok] - b2.lo[ok])
        k, k2 = b.actual_after[ok], b2.actual_after[ok]
        theta = max(1e-6, 2.0 ** -iv.FLOAT_FORMATS[b.elem][0])
        rms = float(np.sqrt(np.mean(f * f)))
        exceed = int(((np.abs(g2 - f) - h2) > theta * np.maximum(np.abs(f), rms)).sum())
        row["outputs"][name] = {
            "elem": b.elem, "n": int(ok.sum()),
            "C_semantic": {"exceeding": exceed, "alignment": _alignment(g2 - f, f), "verdict": "不一致" if exceed else "一致"},
            "C_numeric_after": [_alignment(k2 - g2, g2), _se(k2 - g2, g2)],
            "C_numeric_before": [_alignment(k - f, f), _se(k - f, f)],
            "B_f": [_alignment(k2 - f, f), _se(k2 - f, f)],
            "B_G": [_alignment(k2 - g2, g2), _se(k2 - g2, g2)]}
        if spec["kind"] == "numeric":
            row["outputs"][name]["C_top_site"] = _attribution_line(L2, progs, name, textbook=False)
            row["outputs"][name]["A_top_site"] = _attribution_line(L2, progs, name, textbook=True)
    return row


def main():
    spec_path, out_path = sys.argv[1], sys.argv[2]
    specs = json.loads(Path(spec_path).read_text())
    done = set()
    if Path(out_path).exists():
        done = {json.loads(l)["id"] for l in open(out_path) if l.strip()}
    with open(out_path, "a") as f:
        for s in specs:
            if s["id"] in done:
                continue
            try:
                row = one(s)
            except Exception as e:
                row = {"id": s["id"], "error": f"{type(e).__name__}: {str(e)[:200]}", "trace": traceback.format_exc()[-800:]}
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            print(json.dumps({k: row.get(k) for k in ("id", "kind", "error")}), flush=True)


if __name__ == "__main__":
    main()
