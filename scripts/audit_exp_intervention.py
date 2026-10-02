#!/usr/bin/env python3
"""Audit of the exp / division intervention: what changed in the executed code, and the references.

For every variant relative to the original kernel:

* TTIR operation counts that differ (declared program);
* PTX instruction-mnemonic counts that differ (what actually executed);
* whether the variant's own K_R equals the original K_R on every launch.

Declared-semantics-preserving replacements (libdevice exp, correctly rounded
division) must leave K_R unchanged; the exp2 rewrites change the declared
expression and therefore K_R.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.capture import load_launch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402


def ttir_counts(text):
    module = parse_ttir(text)
    return collections.Counter(op.name for fn in module.funcs.values() for op in fn.walk())


def ptx_counts(text):
    counts = collections.Counter()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("//", ".", "{", "}", "$", "@!")) and not line.startswith("@"):
            continue
        m = re.match(r"(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)", line)
        if m and not m.group(1).endswith(":"):
            counts[m.group(1)] += 1
    return counts


def diff(a, b):
    keys = sorted(set(a) | set(b))
    return {k: [a.get(k, 0), b.get(k, 0)] for k in keys if a.get(k, 0) != b.get(k, 0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    variants = list(manifest["variants"])
    n = manifest["variants"]["original"]["launches"]
    base0 = load_launch(args.capture / "original" / "launch000")
    base_ttir, base_ptx = ttir_counts(base0.asm["ttir"]), ptx_counts(base0.asm["ptx"])
    modules = {}

    def reference(launch):
        module = modules.setdefault(launch.asm["ttir"], parse_ttir(launch.asm["ttir"]))
        result = KernelReferenceEvaluator(module).evaluate(launch)
        buf = next(b for b in result.buffers.values() if b.name == "X_ptr")
        return buf.lo[buf.written], buf.hi[buf.written]

    base_refs = [reference(load_launch(args.capture / "original" / f"launch{j:03d}")) for j in range(n)]
    report = {"schema": "kernel-analyzer-exp-intervention-audit-v1", "variants": {}}
    for v in variants:
        first = load_launch(args.capture / v / "launch000")
        same_ref = True
        for j in range(n):
            lo, hi = reference(load_launch(args.capture / v / f"launch{j:03d}")) if v != "original" else base_refs[j]
            same_ref &= bool(np.array_equal(lo, base_refs[j][0]) and np.array_equal(hi, base_refs[j][1]))
        report["variants"][v] = {
            "ttir_operation_count_changes": diff(base_ttir, ttir_counts(first.asm["ttir"])),
            "ptx_instruction_count_changes": diff(base_ptx, ptx_counts(first.asm["ptx"])),
            "own_reference_equals_original_reference": same_ref,
        }
        print(v, json.dumps(report["variants"][v])[:400])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
