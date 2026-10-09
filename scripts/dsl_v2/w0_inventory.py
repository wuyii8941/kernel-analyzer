#!/usr/bin/env python3
"""W0 (reference DSL v2 rc3 04): official inventories of a locked Triton source tree, written for
tools/reconcile_official_inventory.py of the rc3 package.  Nothing of the official tree is executed: Python modules are
read with ``ast``, TableGen files are expanded by the ``mlir-tblgen`` of the LLVM build the tree pins.

    python scripts/dsl_v2/w0_inventory.py api --src .cache/upstream/triton --out results/dsl_v2/w0/api_<commit>.json
    python scripts/dsl_v2/w0_inventory.py ops --src .cache/upstream/triton --llvm <llvm prefix> \
        --out results/dsl_v2/w0/source_ops_<commit>.json

``api``: the public names of every frontend module (``__all__`` where present, otherwise the public top-level names),
plus the public methods of the tensor classes; a signature digest per name.  ``ops``: every operation definition the
op TableGen files of the tree instantiate (dialect.mnemonic), with a digest of its normalized contract (arguments with
their constraints, results, regions, successors, traits).  Records follow the reconcile schema; ``complete_for_profile``
is attested only for what the producer enumerates (all op .td files of the tree), and the receipt lists the commands
and the hashes of every input file.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FRONTEND_ROOTS = ["python/triton/language", "python/triton/experimental/gluon"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_head(src: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()


def _all_names(tree: ast.Module):
    names = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
                try:
                    val = ast.literal_eval(node.value)
                except ValueError:
                    return None  # computed __all__: fall back to top-level names
                names = (names or []) + list(val)
    return names


def _public_toplevel(tree: ast.Module):
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)
        elif isinstance(node, ast.Assign):
            out += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.ImportFrom):
            out += [a.asname or a.name for a in node.names if a.name != "*"]
        elif isinstance(node, ast.Import):
            out += [(a.asname or a.name).split(".")[0] for a in node.names]
    return [n for n in out if not n.startswith("_")]


def _defs(tree: ast.Module) -> dict:
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node.args) if not isinstance(node, ast.ClassDef) else \
                "class:" + ",".join(ast.unparse(b) for b in node.bases)
    return out


def api(src: Path) -> dict:
    records, files = [], {}
    for root in FRONTEND_ROOTS:
        for init in sorted((src / root).rglob("__init__.py")):
            rel = init.relative_to(src)
            text = init.read_bytes()
            files[str(rel)] = sha(text)
            tree = ast.parse(text)
            mod = ".".join(rel.parts[1:-1])
            names = _all_names(tree)
            how = "__all__"
            if names is None:
                names, how = _public_toplevel(tree), "public top-level names"
            defs = _defs(tree)
            for n in sorted(set(names)):
                records.append({"id": f"{mod}::{n}", "how": how,
                                "contract_hash": sha((defs.get(n) or "re-export").encode())})
    # public methods of the tensor classes
    for path, cls in (("python/triton/language/core.py", "tensor"),
                      ("python/triton/experimental/gluon/language/_core.py", "tensor")):
        p = src / path
        if not p.exists():
            continue
        files[path] = sha(p.read_bytes())
        tree = ast.parse(p.read_bytes())
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and not m.name.startswith("_"):
                        records.append({"id": f"{path}::{cls}.{m.name}", "how": "method",
                                        "contract_hash": sha(ast.dump(m.args).encode())})
    return {"kind": "api", "profile_id": f"official-frontends-{git_head(src)[:12]}", "source_commit": git_head(src),
            "producer": "scripts/dsl_v2/w0_inventory.py api (ast, no execution)",
            "receipt": {"files_sha256": files}, "complete_for_profile": False,
            "completeness_note": "public names of the frontend modules and tensor methods only; dynamic re-exports, "
                                 "decorator-generated builtins and runtime attributes are not enumerated",
            "records": records}


import re

_REC = re.compile(r"^def (\S+) \{\t?// ?(.*)$")
_FIELD = re.compile(r"^  (\S.*?) (\w+) = (.*);$")


def _records(text: str) -> dict:
    """mlir-tblgen --print-records -> {name: {"classes": [...], field: value}} (single-line fields only)."""
    recs, cur = {}, None
    for line in text.splitlines():
        m = _REC.match(line)
        if m:
            cur = recs[m.group(1)] = {"classes": m.group(2).split()}
            continue
        if line == "}":
            cur = None
            continue
        if cur is not None:
            f = _FIELD.match(line)
            if f:
                cur[f.group(2)] = f.group(3)
    return recs


def _resolve(value: str, recs: dict) -> str:
    """Replace anonymous constraint records (numbering depends on the file) by their summary."""
    def sub(m):
        r = recs.get(m.group(0), {})
        return "anon<" + r.get("summary", "?").strip('"') + ">"
    return re.sub(r"anonymous_\d+", sub, value or "")


def ops(src: Path, llvm: Path) -> dict:
    tblgen = llvm / "bin" / "mlir-tblgen"
    incs = [llvm / "include", src / "include", src / "third_party", src / "third_party/nvidia/include",
            src / "third_party/amd/include", src / "third_party/proton/Dialect/include", src / "lib"]
    files, commands, failures, uniq = {}, [], [], {}
    for td in sorted(list((src / "include").rglob("*Ops.td")) + list((src / "third_party").rglob("*Ops.td"))):
        rel = str(td.relative_to(src))
        files[rel] = sha(td.read_bytes())
        cmd = [str(tblgen), "--print-records", str(td)] + [f"-I{p}" for p in incs] + [f"-I{td.parent}"]
        commands.append(" ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failures.append({"file": rel, "stderr": r.stderr[-2000:]})
            continue
        recs = _records(r.stdout)
        for name, rec in recs.items():
            if "Op" not in rec["classes"] or "opName" not in rec or name.startswith("anonymous_"):
                continue
            dialect = recs.get(rec.get("opDialect", ""), {}).get("name", "?").strip('"')
            ident = f"{dialect}.{rec['opName'].strip(chr(34))}"
            norm = {k: _resolve(rec.get(k), recs) for k in ("arguments", "results", "regions", "successors", "traits",
                                                            "assemblyFormat", "hasCustomAssemblyFormat")}
            if ident not in uniq:
                uniq[ident] = {"id": ident, "defined_by": rel, "record": name, "contract": norm,
                               "contract_hash": sha(json.dumps(norm, sort_keys=True).encode())}
    return {"kind": "source", "profile_id": f"official-ir-{git_head(src)[:12]}", "source_commit": git_head(src),
            "producer": "scripts/dsl_v2/w0_inventory.py ops (mlir-tblgen --print-records of every *Ops.td of the tree)",
            "receipt": {"files_sha256": files, "mlir_tblgen_sha256": sha(tblgen.read_bytes()), "commands": commands,
                        "failures": failures},
            "complete_for_profile": not failures,
            "completeness_note": "operations instantiated by the *Ops.td files of the tree (all dialects of the tree, "
                                 "NVIDIA / AMD / Proton included), with the upstream MLIR dialects those files include; "
                                 "op definitions in other .td files and dialects only the pipelines load are not listed",
            "records": sorted(uniq.values(), key=lambda r: r["id"])}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("api")
    a.add_argument("--src", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    o = sub.add_parser("ops")
    o.add_argument("--src", type=Path, required=True)
    o.add_argument("--llvm", type=Path, required=True)
    o.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"refusing to overwrite {args.out}")
    out = api(args.src) if args.cmd == "api" else ops(args.src, args.llvm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(args.cmd, len(out["records"]), "records", "complete" if out["complete_for_profile"] else "not attested complete")


if __name__ == "__main__":
    main()
