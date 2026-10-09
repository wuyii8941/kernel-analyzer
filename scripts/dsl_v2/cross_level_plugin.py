"""pytest plugin (DSL v2 increment 11): the same captured launch evaluated at two IR levels.

For each launch of an official test (official main build): the reference from its TTIR, and the references from the
TTGIR the official compiler produces for sm_90 and sm_100 out of that TTIR (compiled only; nothing runs on those
targets).  The real semantics of the two levels is the same, so where both references are established their
enclosures must intersect.  One JSON line per launch and target:

    {"test", "kernel", "target", "compile", "ttir_status", "ttgir_status", "both_ok", "disjoint", "ttir_only_ok",
     "ttgir_only_ok", "ttgir_reasons", "ttgir_aborted"}

Load with ``-p cross_level_plugin`` (scripts/dsl_v2 and src on PYTHONPATH) and KA_CROSS_LEVEL_OUT=<jsonl>.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pytest

MAX_BYTES = int(os.environ.get("KA_MAIN_CAPTURE_MAX_BYTES", str(16 << 20)))
TARGETS = (90, 100)
_COMPILED: dict = {}


def _ttgir_for(ttir: str, arch: int):
    key = (hashlib.sha256(ttir.encode()).hexdigest(), arch)
    if key in _COMPILED:
        return _COMPILED[key]
    import triton
    from triton.backends.compiler import GPUTarget
    d = Path(os.environ.get("TMPDIR", "/tmp")) / "cross_level"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key[0][:16]}.ttir"
    path.write_text(ttir)
    try:
        ck = triton.compile(str(path), target=GPUTarget("cuda", arch, 32))
        out = ("ok", ck.asm["ttgir"])
    except Exception as exc:  # noqa: BLE001 -- recorded
        out = (f"{type(exc).__name__}: {exc}"[:200], None)
    _COMPILED[key] = out
    return out


def _status(ref):
    from kernel_analyzer.reference_eval.ttir_eval import ST_NINF
    written = complete = 0
    for buf in ref.buffers.values():
        w = np.asarray(buf.written)
        written += int(w.sum())
        complete += int((w & (np.asarray(buf.st) <= ST_NINF)).sum())
    return "aborted" if ref.aborted else ("complete" if written and complete == written else
                                          ("partial" if written else "nothing written"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    TritonLaunchRecorder.install_hook()
    rec = TritonLaunchRecorder(max_launches=2, keep_storages=True)
    with rec:
        yield
    out = os.environ.get("KA_CROSS_LEVEL_OUT")
    if not out or not rec.launches:
        return
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK, evaluate_sequence
    lines = []
    for launch in rec.launches:
        if "ttir" not in launch.asm:
            continue
        size = sum((a.storage_nbytes or 0) for a in launch.args if a.kind == "tensor")
        if size > MAX_BYTES:
            continue
        try:
            base = evaluate_sequence([launch]).launches[0]
        except Exception as exc:  # noqa: BLE001
            lines.append({"test": item.nodeid, "kernel": launch.kernel_name, "target": "ttir",
                          "compile": "n/a", "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        for arch in TARGETS:
            row = {"test": item.nodeid, "kernel": launch.kernel_name, "target": f"sm_{arch}",
                   "ttir_status": _status(base)}
            status, ttgir = _ttgir_for(launch.asm["ttir"], arch)
            row["compile"] = status
            if ttgir is not None:
                import re
                row["target_ops"] = sorted(set(re.findall(r"\b(ttng\.[a-z_0-9]+|ttg\.(?:memdesc_trans|fp4_to_fp|local_[a-z_]+))", ttgir)))
            if ttgir is None:
                lines.append(row)
                continue
            other = dataclasses.replace(launch, asm={"ttgir": ttgir})
            try:
                ref = evaluate_sequence([other]).launches[0]
            except Exception as exc:  # noqa: BLE001 -- an evaluator crash is a defect
                row.update(ttgir_status="error", error=f"{type(exc).__name__}: {exc}"[:200],
                           traceback=traceback.format_exc()[-1200:])
                lines.append(row)
                continue
            both = disjoint = only_a = only_b = 0
            for ident, b1 in base.buffers.items():
                b2 = ref.buffers.get(ident)
                if b2 is None:
                    continue
                w = np.asarray(b1.written) | np.asarray(b2.written)
                ok1 = w & (np.asarray(b1.st) == ST_OK)
                ok2 = w & (np.asarray(b2.st) == ST_OK)
                both_ok = ok1 & ok2
                both += int(both_ok.sum())
                only_a += int((ok1 & ~ok2).sum())
                only_b += int((ok2 & ~ok1).sum())
                if both_ok.any():
                    lo1, lo2 = np.asarray(b1.lo)[both_ok], np.asarray(b2.lo)[both_ok]
                    hi1 = np.asarray(b1.hi)[both_ok] if b1.hi is not None else lo1
                    hi2 = np.asarray(b2.hi)[both_ok] if b2.hi is not None else lo2
                    disjoint += int(((hi1 < lo2) | (hi2 < lo1)).sum())
            row.update(ttgir_status=_status(ref), both_ok=both, disjoint=disjoint, ttir_only_ok=only_a,
                       ttgir_only_ok=only_b,
                       ttgir_reasons=sorted(k for k in ref.reasons if k.startswith("not_established"))[:4],
                       ttgir_aborted=sorted(set(ref.aborted.values()))[:2])
            lines.append(row)
    with open(out, "a") as handle:
        for row in lines:
            handle.write(json.dumps(row, default=str) + "\n")
