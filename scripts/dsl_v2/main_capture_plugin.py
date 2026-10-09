"""pytest plugin (DSL v2 increment 3, W3): capture the launches of official Triton tests run under the official main
build and evaluate each with the tool-4.0 reference evaluator.  One JSON line per captured launch:

    {"test", "kernel", "grid", "written", "complete", "status": complete|partial|aborted|error, "reasons", "aborted",
     "set_reasons" (complete lanes that are set targets, L_E: DSL v2 increment 4)}

Load with ``-p main_capture_plugin`` (scripts/dsl_v2 and src on PYTHONPATH) and KA_MAIN_CAPTURE_OUT=<jsonl>.
Measurement only: the official tests' own assertions are unaffected (the recorder copies operands around launches).
"""
from __future__ import annotations

import json
import os
import traceback

import numpy as np
import pytest

MAX_LAUNCHES = int(os.environ.get("KA_MAIN_CAPTURE_MAX", "4"))
MAX_BYTES = int(os.environ.get("KA_MAIN_CAPTURE_MAX_BYTES", str(64 << 20)))


def _small(name, index):
    return True


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    TritonLaunchRecorder.install_hook()
    rec = TritonLaunchRecorder(select=_small, max_launches=MAX_LAUNCHES, keep_storages=True)
    with rec:
        outcome = yield
    out = os.environ.get("KA_MAIN_CAPTURE_OUT")
    if not out or not rec.launches:
        return
    from kernel_analyzer.reference_eval.ttir_eval import ST_NINF, evaluate_sequence
    lines = []
    for launch in rec.launches:
        size = sum((a.storage_nbytes or 0) for a in launch.args if a.kind == "tensor")
        row = {"test": item.nodeid, "kernel": launch.kernel_name, "grid": list(launch.grid),
               "test_outcome": "failed" if outcome.excinfo else "passed", "bytes": size}
        if size > MAX_BYTES:
            row.update(status="skipped", reason="operands above the capture size bound")
            lines.append(row)
            continue
        try:
            ref = evaluate_sequence([launch]).launches[0]
            written = complete = 0
            for buf in ref.buffers.values():
                w = np.asarray(buf.written)
                written += int(w.sum())
                complete += int((w & (np.asarray(buf.st) <= ST_NINF)).sum())  # special values are established
            row.update(written=written, complete=complete,
                       status="aborted" if ref.aborted else ("complete" if written and complete == written
                                                             else ("partial" if written else "nothing written")),
                       aborted=sorted(set(ref.aborted.values()))[:3],
                       reasons=sorted(k for k in ref.reasons if k.startswith("not_established"))[:5],
                       set_reasons=sorted(k for k in ref.reasons if k.startswith("set:"))[:5],
                       notes=sorted(k for k in ref.reasons if k.startswith(("assumed:", "dot_input_precision:")))[:5],
                       execution={k: v for k, v in ref.rules.items()
                                  if k.startswith(("execution.", "premise.", "atomic.", "dot.integer"))})
        except Exception as exc:  # noqa: BLE001 -- recorded: an evaluator crash is a defect, never hidden
            row.update(status="error", reason=f"{type(exc).__name__}: {exc}"[:300],
                       traceback=traceback.format_exc()[-1500:])
        lines.append(row)
    with open(out, "a") as handle:
        for row in lines:
            handle.write(json.dumps(row, default=str) + "\n")
