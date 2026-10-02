#!/usr/bin/env python3
"""Layered coverage report for the automatic reference (static / evaluation / rules).

1. Static instruction-processing coverage of every TTIR file in the corpus.
2. Evaluation coverage per kernel: output elements in the three classes.
3. Rule triggers per kernel: how often the reference-memory, path,
   observability and atomic rules (plus races and undefined values) fired.

Kernels are captured live (exercise and Inductor workloads) or loaded from
capture packages (Liger, torchao).  The Liger dW accumulation classes come
from the 96 per-unit summaries; its rule counts from one recaptured unit.
Runs in ka_main.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

RULE_GROUPS = {
    "reference_memory": ("memory.load_reads_reference_value", "memory.external_reentry"),
    "path": ("path.branch_decided_by_reference", "path.branch_union", "path.select_union_lanes",
             "path.program_aborted"),
    "observability": ("observability.pinned_load_lanes",),
    "atomic": ("atomic.return_value_not_established_lanes", "atomic.folded_order_free_lanes"),
    "other": ("memory.cross_program_race_lanes", "memory.masked_load_undefined_lanes"),
}


def static_layer():
    from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    rows = []
    dialect_ops = collections.Counter()
    dialect_names = collections.defaultdict(set)
    for path in sorted(glob.glob(str(ROOT / "results/reference_eval/ttir_corpus/*/*.ttir"))):
        module = parse_ttir(Path(path).read_text())
        rep = kernel_coverage(module)
        for op_row in rep["rows"]:
            dialect = op_row["op"].split(".")[0]
            dialect_ops[dialect] += 1
            dialect_names[dialect].add(op_row["op"])
        env, name = Path(path).parent.name, Path(path).name
        rows.append({"environment": env, "file": name, "group": name.split("__")[0], "kernel": rep["kernel"],
                     "operations": rep["operations"], "categories": rep["categories"],
                     "complete": rep["complete"], "rejected": len(rep["rejected"])})
    return {"files": len(rows), "distinct_kernel_names": len({r["kernel"] for r in rows}),
            "complete": sum(r["complete"] for r in rows),
            "scope": "parse + instruction-processing coverage only; corpus capture stored no operands",
            "by_dialect": {d: {"operation_instances": dialect_ops[d], "distinct_operations": len(dialect_names[d])}
                           for d in sorted(dialect_ops)},
            "rows": rows}


def classes_of(result):
    out = collections.Counter()
    for buf in result.buffers.values():
        m = buf.written
        out["complete_composed"] += int(((buf.st == 0) & ~buf.cond & m).sum())
        out["conditional_local"] += int(((buf.st == 0) & buf.cond & m).sum())
        out["special_value_complete"] += int(((buf.st >= 1) & (buf.st <= 3) & ~buf.cond & m).sum())
        out["not_established"] += int(((buf.st >= 4) & m).sum())
    return dict(out)


def evaluate(launch, **kwargs):
    import time

    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir

    t0 = time.time()
    result = KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch, **kwargs)
    result.seconds = time.time() - t0
    return result


def row(source, launch, result, note=""):
    return {"source": source, "kernel": launch.kernel_name, "grid": list(launch.grid),
            "classes": classes_of(result), "rules": result.rules, "aborted_programs": len(result.aborted),
            "seconds": round(result.seconds, 3), "note": note}


def live_layers():
    import torch

    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from scripts import validate_inductor_reference, validate_ttir_reference
    from scripts import reference_eval_kernels as k

    rows = []
    for name, fn in validate_ttir_reference.workloads():
        recorder = TritonLaunchRecorder()
        with recorder:
            fn()
            torch.cuda.synchronize()
        launch = recorder.launches[-1]
        rows.append(row("exercise", launch, evaluate(launch), name))
        if name == "store_load_chain":
            rows.append(row("exercise", launch, evaluate(launch, pin_loads=("%u",)), "store_load_chain, load pinned"))
    x = torch.full((16,), 2.0 ** 24, device="cuda")
    recorder = TritonLaunchRecorder()
    with recorder:
        k.branch_after_rounding[(1,)](x, torch.empty_like(x), 2.0 ** 24, BLOCK=16)
        k.masked_copy[(1,)](torch.randn(128, device="cuda"), torch.empty(128, device="cuda"),
                            torch.empty(128, device="cuda"), 100, BLOCK=128)
        torch.cuda.synchronize()
    for launch, note in zip(recorder.launches, ("2**24 + 1 then branch", "masked load without other")):
        rows.append(row("exercise", launch, evaluate(launch), note))
    for name, make in validate_inductor_reference.workloads().items():
        torch._dynamo.reset()
        recorder = TritonLaunchRecorder()
        with recorder:
            make()()
            torch.cuda.synchronize()
        seen = set()
        for launch in recorder.launches:
            key = (launch.kernel_name, launch.asm["ttir"])
            if key not in seen:
                seen.add(key)
                rows.append(row("inductor", launch, evaluate(launch), name))
    return rows


def package_layers(liger_kernels: Path, torchao: Path):
    from kernel_analyzer.reference_eval.capture import load_launch

    rows = []
    if liger_kernels.exists():
        manifest = json.loads((liger_kernels / "manifest.json").read_text())
        ce = None
        for r in manifest["launches"]:
            launch = load_launch(liger_kernels / f"launch{r['launch']:03d}")
            result = evaluate(launch)
            entry = row("liger", launch, result)
            if launch.kernel_name == "liger_cross_entropy_kernel":
                if ce is None:
                    ce = dict(entry, launches=0, seconds=0.0, aborted_programs=0,
                              classes=collections.Counter(), rules=collections.Counter())
                ce["launches"] += 1
                ce["seconds"] = round(ce["seconds"] + entry["seconds"], 3)
                ce["aborted_programs"] += entry["aborted_programs"]
                ce["classes"].update(entry["classes"])
                ce["rules"].update(entry["rules"])
            else:
                rows.append(entry)
        if ce is not None:
            ce["classes"], ce["rules"] = dict(ce["classes"]), dict(ce["rules"])
            ce["note"] = f"{ce['launches']} launches summed"
            rows.append(ce)
    if torchao.exists():
        manifest = json.loads((torchao / "manifest.json").read_text())
        for r in manifest["launches"]:
            launch = load_launch(torchao / f"launch{r['launch']:03d}")
            rows.append(row("torchao", launch, evaluate(launch), f"step {r['step']}"))
    return rows


def liger_order_layer(units_dir: Path, capture: Path):
    from kernel_analyzer.reference_eval.capture import load_launch
    from kernel_analyzer.reference_eval.ttir_eval import evaluate_sequence

    totals = collections.Counter()
    for path in sorted(units_dir.glob("unit*.json")):
        unit = json.loads(path.read_text())
        for variant in ("original", "reverse"):
            totals.update(unit[variant]["classes"])
    seconds = [json.loads(p.read_text())[v]["evaluation_seconds"] for p in sorted(units_dir.glob("unit*.json"))
               for v in ("original", "reverse")]
    entry = {"source": "liger_order", "kernel": "accumulate (Liger FP32 dW chunk accumulation)",
             "classes": dict(totals), "aborted_programs": 0,
             "seconds": round(sum(seconds), 1), "seconds_per_sequence": round(sum(seconds) / len(seconds), 1),
             "note": "96 draws (32-state bank) x 2 orders x 64 launches on 524,288 coordinates"}
    if capture.exists():
        design = json.loads((capture / "design.json").read_text())
        variant_dir = next(p for p in sorted(capture.glob("unit*/*")) if p.is_dir())
        launches = [load_launch(p) for p in sorted(variant_dir.glob("launch*"))]
        rows_pids = [(int(r), 0, 0) for r in design["rows"]]
        sequence = evaluate_sequence(launches, programs_for=lambda launch: rows_pids)
        rules = collections.Counter()
        for res in sequence.launches:
            rules.update(res.rules)
        rules["memory.external_reentry"] = len(sequence.external_writes)
        entry["rules"] = dict(rules)
        entry["rules_from"] = f"one recaptured unit ({variant_dir.parent.name}/{variant_dir.name})"
    return entry


def rule_table(rows):
    table = []
    for r in rows:
        rules = r.get("rules") or {}
        table.append({"source": r["source"], "kernel": r["kernel"], "note": r.get("note", ""),
                      **{group: {k: rules.get(k, 0) for k in keys if rules.get(k, 0)}
                         for group, keys in RULE_GROUPS.items()}})
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/reference_eval/layered_report.json")
    parser.add_argument("--liger-kernels", type=Path, default=ROOT / ".cache/liger_kernels")
    parser.add_argument("--torchao", type=Path, default=ROOT / ".cache/torchao_capture")
    parser.add_argument("--liger-order-capture", type=Path, default=ROOT / ".cache/liger_order_rules")
    args = parser.parse_args()
    static = static_layer()
    rows = live_layers() + package_layers(args.liger_kernels, args.torchao)
    rows.append(liger_order_layer(ROOT / "results/reference_eval/liger_order_units", args.liger_order_capture))
    payload = {"schema": "kernel-analyzer-layered-coverage-v1", "reference_mode": "numerical_difference",
               "static_instruction_coverage": static, "evaluation_coverage": rows,
               "rule_triggers": rule_table(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({"static": {k: static[k] for k in ("files", "distinct_kernel_names", "complete")},
                      "evaluated_rows": len(rows)}))
    for r in rows:
        print(f"{r['source']:12s} {r['kernel'][:44]:44s} {r.get('note', '')[:26]:26s} "
              f"abort={r.get('aborted_programs')} t={r.get('seconds')} {r['classes']}")


if __name__ == "__main__":
    main()
