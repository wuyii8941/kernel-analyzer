#!/usr/bin/env python3
"""Static binding of the three_way Inductor kernels to their ATen graph fragments (PLAN_v3 V3, static part).

For every unit of 1_experiments/three_way/results/units.jsonl:
  1. the kernel file named in the captured TTIR (#loc) is matched to a kernel block of Inductor's output code under
     .cache/inductor -- exactly, or, for template kernels launched while autotuning, after removing the autotuning
     configuration lines (block sizes, warps, stages), which leaves the matrix sizes, strides and epilogue;
  2. the graph fragment written above that block is parsed (kernel_spec/fx_text.py);
  3. every node's operation and overload is checked against the operations covered by the base contracts (GRAPH).

    python 5_specification/corpus/bind_inductor.py      -> corpus/bindings/three_way.json.gz and a summary on stdout
"""
from __future__ import annotations

import collections
import glob
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent
REPO = SPEC.parent
sys.path.insert(0, str(SPEC))

from kernel_spec import fx_text  # noqa: E402
from kernel_spec.graph import covered_ops  # noqa: E402

CONFIG = re.compile(r"(num_stages|num_warps|BLOCK_[MNK]|GROUP_M|EVEN_K|ACC_TYPE|USE_FAST_ACCUM|matrix_instr|waves_per_eu|"
                    r"kpack|'configs'|autotune_hints)")
KC = re.compile(r"^(\w+) = async_compile\.triton\('(\w+)', '''", re.M)


def norm_template(body):
    return "\n".join(l.rstrip() for l in body.splitlines() if l.strip() and not CONFIG.search(l))


def index_output_code():
    exact, templ = {}, {}
    for f in sorted(glob.glob(str(REPO / ".cache/inductor/*/*.py"))):
        t = open(f, errors="ignore").read()
        if "Graph fragment" not in t:
            continue
        for m in KC.finditer(t):
            head = t.rfind("# Topologically Sorted Source Nodes:", 0, m.start())
            end = t.find("''', device_str=", m.end())
            if head < 0 or end < 0:
                continue
            comment = t[head:m.start()]
            frag = comment.split("# Graph fragment:", 1)[1].splitlines() if "# Graph fragment:" in comment else []
            orig = re.search(r"Original ATen: \[(.*?)\]", comment)
            rec = {"kernel": m.group(2), "file": str(Path(f).relative_to(REPO)), "original_aten": orig.group(1) if orig else "",
                   "fragment": [l for l in frag if l.startswith("#   %")]}
            body = t[m.end():end]
            exact.setdefault(hashlib.sha256(body.strip().encode()).hexdigest(), rec)
            if "_tem_" in m.group(2):
                templ.setdefault((m.group(2), hashlib.sha256(norm_template(body).encode()).hexdigest()), rec)
    return exact, templ


def main():
    exact, templ = index_output_code()
    ops = covered_ops()
    units = [json.loads(l) for l in open(REPO / "1_experiments/three_way/results/units.jsonl")]
    out, uncovered, status = [], collections.Counter(), collections.Counter()
    for u in units:
        unit = u["unit"].rstrip("/")
        model, config, name = unit.split("/")[-3:]
        rec, method, problems = None, None, []
        cap = REPO / ".cache/three_way" / "/".join(unit.split("/")[-4:])
        srcs = list((cap / "compiled").glob("*.source")) if cap.is_dir() else []
        loc = re.search(r'#loc = loc\("([^"]+\.py)"', srcs[0].read_text(errors="ignore")) if srcs else None
        kfile = Path(loc.group(1)) if loc else None
        if kfile and kfile.is_file():
            body = kfile.read_text()
            rec = exact.get(hashlib.sha256(body.strip().encode()).hexdigest())
            method = "exact kernel source" if rec else None
            if rec is None and "_tem_" in u.get("kernel", ""):
                rec = templ.get((u["kernel"], hashlib.sha256(norm_template(body).encode()).hexdigest()))
                method = "template source without autotuning configuration" if rec else None
        elif kfile:
            problems.append(f"kernel file missing: {kfile}")
        nodes_ops, bad = [], []
        if rec:
            try:
                nodes = fx_text.parse_fragment(rec["fragment"])
                g = fx_text.to_graph(nodes)
                for n in g["nodes"]:
                    nodes_ops.append(n["op"])
                    if n["op"] not in ops:
                        bad.append(n["op"])
            except fx_text.ParseError as e:
                problems.append(f"parse: {e}")
        st = ("bound" if rec and not bad and not problems else "bound, uncovered operations" if rec and bad
              else "bound, fragment not parsed" if rec else "not bound")
        status[st] += 1
        uncovered.update(bad)
        out.append({"unit": f"{model}/{config}/{name}", "kernel": u.get("kernel"), "verify": u.get("verify"), "status": st,
                    "method": method, "output_code": rec and rec["file"], "original_aten": rec and rec["original_aten"],
                    "graph_ops": sorted(set(nodes_ops)), "nodes": len(nodes_ops), "uncovered": sorted(set(bad)),
                    "problems": problems})
    (HERE / "bindings").mkdir(exist_ok=True)
    with gzip.GzipFile(HERE / "bindings/three_way.json.gz", "wb", mtime=0) as fh:
        fh.write(json.dumps(out, indent=0, ensure_ascii=False).encode())
    print("units", len(out), dict(status))
    print("uncovered operations", dict(uncovered))
    print("binding methods", dict(collections.Counter(r["method"] for r in out)))
    for r in out:
        if r["problems"]:
            print("problem", r["unit"], r["problems"][:2])
    return 0 if status["bound"] == len(out) else 1


if __name__ == "__main__":
    sys.exit(main())
