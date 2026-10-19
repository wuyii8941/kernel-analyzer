#!/usr/bin/env python3
"""Static matching of the three_way Inductor kernels to their ATen graph fragments (PLAN_v3 V3, static part).

For every unit of 1_experiments/three_way/results/units.jsonl:
  1. the kernel file named in the captured TTIR (#loc) is matched to a kernel block of Inductor's output code under
     .cache/inductor -- exactly, or, for template kernels launched while autotuning, after removing the autotuning
     configuration lines (block sizes, warps, stages), which leaves the matrix sizes, strides and epilogue;
  2. the graph fragment written above that block is parsed (kernel_spec/fx_text.py);
  3. every node's operation and overload is checked against the operations covered by the base contracts (GRAPH).

    python 5_specification/corpus/bind_inductor.py --out NEW.json.gz

The output must not already exist. Static matches are candidate evidence only, not complete task binding.
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import gzip
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


def index_output_code(code_dir=None):
    exact, templ = {}, {}
    code_dir = Path(code_dir).resolve() if code_dir is not None else REPO / ".cache/inductor"
    for f in sorted(glob.glob(str(code_dir / "*/*.py"))):
        t = Path(f).read_text(errors="ignore")
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
            rec = {"kernel": m.group(2), "launcher": m.group(1), "file": str(Path(f).relative_to(REPO)), "original_aten": orig.group(1) if orig else "",
                   "fragment": [l for l in frag if l.startswith("#   %")]}
            body = t[m.end():end]
            matches = exact.setdefault(body.strip(), [])
            if rec not in matches:
                matches.append(rec)
            if "_tem_" in m.group(2):
                matches = templ.setdefault((m.group(2), norm_template(body)), [])
                if rec not in matches:
                    matches.append(rec)
    return exact, templ


def candidates_at_call_site(candidates, frames):
    """Candidates whose launcher call contains an actually recorded Python location.

    Source matching has already happened. This only selects a caller, not a
    mathematical task or complete parameter mapping. Ambiguity is retained.
    """
    if not isinstance(frames, list):
        return []
    selected = []
    for candidate in candidates:
        path = (REPO / candidate["file"]).resolve()
        locations = [frame["line"] for frame in frames if isinstance(frame, dict)
                     and isinstance(frame.get("file"), str) and type(frame.get("line")) is int
                     and Path(frame["file"]).resolve() == path]
        if not locations or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError):
            continue
        launcher = candidate.get("launcher", candidate["kernel"])
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "run"
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == launcher]
        if any(node.lineno <= line <= node.end_lineno for node in calls for line in locations):
            selected.append(candidate)
    return selected


def write_results(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
            fh.write(json.dumps(rows, indent=0, ensure_ascii=False).encode())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new gzip JSON result file; existing files are refused")
    parser.add_argument("--units", type=Path, help="explicit inventory for a new capture run")
    parser.add_argument("--capture-root", type=Path, help="capture directory containing model/config/unit")
    parser.add_argument("--code-dir", type=Path, help="Inductor code directory for this capture run")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a new file; existing records are never overwritten")
    with (args.units or REPO / "1_experiments/three_way/results/units.jsonl").open() as source:
        units = [json.loads(line) for line in source]
    if not units:
        parser.error("the unit inventory is empty")
    exact, templ = index_output_code(args.code_dir)
    ops = covered_ops()
    out, uncovered, status = [], collections.Counter(), collections.Counter()
    for u in units:
        unit = u["unit"].rstrip("/")
        model, config, name = unit.split("/")[-3:]
        rec, method, problems, candidates = None, None, [], []
        source_candidates, candidate_selection = [], None
        cap = REPO / ".cache/three_way" / "/".join(unit.split("/")[-4:])
        if args.capture_root is not None:
            cap = args.capture_root / model / config / name
        srcs = list((cap / "compiled").glob("*.source")) if cap.is_dir() else []
        loc = re.search(r'#loc = loc\("([^"]+\.py)"', srcs[0].read_text(errors="ignore")) if srcs else None
        kfile = Path(loc.group(1)) if loc else None
        if kfile and kfile.is_file():
            body = kfile.read_text()
            candidates = exact.get(body.strip(), [])
            method = "exact kernel source" if candidates else None
            if not candidates and "_tem_" in u.get("kernel", ""):
                candidates = templ.get((u["kernel"], norm_template(body)), [])
                method = "template source without autotuning configuration" if candidates else None
            source_candidates = candidates
            manifest = cap / "launch.json"
            if len(candidates) > 1 and manifest.is_file():
                frames = json.loads(manifest.read_text()).get("call_stack")
                selected = candidates_at_call_site(candidates, frames)
                if len(selected) == 1:
                    candidates = selected
                    candidate_selection = "recorded Python launcher call"
            if len(candidates) == 1:
                rec = candidates[0]
            elif len(candidates) > 1:
                problems.append("multiple graph origins match this source; no task was selected")
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
        # what this checks: the kernel source matches an output-code block, its graph fragment parses, and every
        # operation is covered by a base contract. It does NOT bind inputs, outputs, state or profile (see
        # dynamic_three_way.py for that); hence the status names below.
        st = ("multiple matching graph fragments" if len(candidates) > 1
              else "fragment matched, operations covered" if rec and not bad and not problems
              else "fragment matched, uncovered operations" if rec and bad
              else "fragment matched, not parsed" if rec else "no fragment")
        status[st] += 1
        uncovered.update(bad)
        out.append({"unit": f"{model}/{config}/{name}", "kernel": u.get("kernel"), "verify": u.get("verify"), "status": st,
                    "method": method, "output_code": rec and rec["file"], "original_aten": rec and rec["original_aten"],
                    "graph_ops": sorted(set(nodes_ops)), "nodes": len(nodes_ops), "uncovered": sorted(set(bad)),
                    "problems": problems, "candidates": candidates,
                    "source_candidates": source_candidates, "candidate_selection": candidate_selection,
                    "candidate_status": "ambiguous" if len(candidates) > 1 else "matched" if rec else "not_matched",
                    "binding_status": "not_established", "reference_status": "not_evaluated",
                    "semantic_verdict": "not_established"})
    write_results(args.out, out)
    print("units", len(out), dict(status))
    print("uncovered operations", dict(uncovered))
    print("binding methods", dict(collections.Counter(r["method"] for r in out)))
    for r in out:
        if r["problems"]:
            print("problem", r["unit"], r["problems"][:2])
    return 0 if status["fragment matched, operations covered"] == len(out) else 1


if __name__ == "__main__":
    sys.exit(main())
