#!/usr/bin/env python3
"""Capture the first launches of each kernel of official Triton tutorials under the official main build and evaluate
them with the tool-4.0 reference evaluator (DSL v2 increment 3, A3 expectation on tutorials).  One JSON line per
captured launch, in the format of main_capture_plugin.py.

    python scripts/dsl_v2/tutorial_capture.py --out OUT.jsonl TUTORIAL.py [...]

Run in the official env (PYTHONPATH: triton_main site-packages, src, scripts/dsl_v2, ka_main site-packages).  The
tutorials run unchanged (their own correctness checks and benchmarks); only the first ``--per-kernel`` launches of each
kernel name are copied.  Measurement only.
"""
from __future__ import annotations

import argparse
import collections
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-kernel", type=int, default=2)
    ap.add_argument("--max-bytes", type=int, default=64 << 20)
    ap.add_argument("tutorials", nargs="+", type=Path)
    a = ap.parse_args()
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence
    TritonLaunchRecorder.install_hook()
    for tut in a.tutorials:
        seen = collections.Counter()

        def select(name, index):
            seen[name] += 1
            return seen[name] <= a.per_kernel

        rec = TritonLaunchRecorder(select=select, max_launches=64, keep_storages=True)
        t0 = time.time()
        run_error = None
        argv = sys.argv
        try:
            sys.argv = [str(tut)]
            with rec:
                runpy.run_path(str(tut), run_name="__main__")
        except BaseException as exc:  # noqa: BLE001 -- recorded; a tutorial failure is data, not a crash of the run
            run_error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            sys.argv = argv
        rows = []
        for launch in rec.launches:
            size = sum((x.storage_nbytes or 0) for x in launch.args if x.kind == "tensor")
            row = {"tutorial": tut.name, "kernel": launch.kernel_name, "grid": list(launch.grid), "bytes": size,
                   "tutorial_error": run_error}
            if size > a.max_bytes:
                row.update(status="skipped", reason="operands above the capture size bound")
                rows.append(row)
                continue
            t1 = time.time()
            try:
                ref = evaluate_sequence([launch]).launches[0]
                written = complete = premised = 0
                for ident, buf in ref.buffers.items():
                    w = np.asarray(buf.written)
                    written += int(w.sum())
                    complete += int(ref.complete(ident).sum())  # special values are established
                    premised += int(ref.under_premise(ident).sum())  # apart: unproven premise (audit F03)
                row.update(written=written, complete=complete, complete_under_premise=premised,
                           status="aborted" if ref.aborted else ("complete" if written and complete == written
                                                                 else ("complete under premise" if written and
                                                                       complete + premised == written
                                                                       else ("partial" if written
                                                                             else "nothing written"))),
                           aborted=sorted(set(ref.aborted.values()))[:3],
                           reasons=sorted(k for k in ref.reasons if k.startswith("not_established"))[:5],
                           set_reasons=sorted(k for k in ref.reasons if k.startswith("set:"))[:5],
                           notes=sorted(k for k in ref.reasons if k.startswith(("assumed:", "dot_input_precision:")))[:5])
            except Exception as exc:  # noqa: BLE001 -- an evaluator crash is a defect, recorded
                row.update(status="error", reason=f"{type(exc).__name__}: {exc}"[:300],
                           traceback=traceback.format_exc()[-1500:])
            row["eval_seconds"] = round(time.time() - t1, 2)
            rows.append(row)
        with open(a.out, "a") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
        print(tut.name, "launches", len(rec.launches), "error", run_error, "seconds", round(time.time() - t0, 1),
              flush=True)


if __name__ == "__main__":
    main()
