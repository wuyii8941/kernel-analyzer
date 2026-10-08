"""Enumeration test of the rule registry (general-capability round, protocol section 2)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from kernel_analyzer.reference_eval import ttir_mapping as M
from kernel_analyzer.reference_eval.rule_registry import build

ROOT = Path(__file__).resolve().parents[1]
REG = build()


def test_every_operation_of_the_locked_build_has_an_entry():
    ops = {name for names in json.loads(Path(M.REGISTRY_PATH).read_text())["operations"].values() for name in names}
    covered = {e["key"]["op"] for e in REG["entries"]}
    assert ops <= covered, sorted(ops - covered)[:10]


def test_no_entry_is_silent():
    for e in REG["entries"]:
        assert e["status"] in ("SUPPORTED", "DECLARED_PREMISE", "NOT_ESTABLISHED", "REJECTED"), e
        if e["status"] in ("SUPPORTED", "DECLARED_PREMISE"):
            assert e["containment_argument"] and e["counterexample_test"] and e["negative_control"], e["key"]
        else:
            assert e["reason"] and e["counterexample_test"], e["key"]


def _defined(ref):
    path, fn = ref.split("::")
    tree = ast.parse((ROOT / path).read_text())
    return any(isinstance(n, ast.FunctionDef) and n.name == fn for n in ast.walk(tree))


def test_every_referenced_test_exists():
    refs = {r for e in REG["entries"] for r in (e["counterexample_test"], e["negative_control"]) if r}
    missing = [r for r in sorted(refs) if not _defined(r)]
    assert not missing, missing


def test_registered_combiners_and_tables_are_all_entries():
    subs = {e["key"]["subregion"] for e in REG["entries"] if e["key"]["op"] == "tt.reduce"}
    for comb in set(M._SIMPLE_COMBINERS.values()) | {"max_select", "min_select", "argmax", "argmin"}:
        assert comb in subs
    assert any(s.startswith("welford") for s in subs) and "other" in subs
    attrs = {e["key"]["attrs"] for e in REG["entries"] if e["key"]["op"] == "tt.extern_elementwise"}
    assert all(f"symbol={s}" in attrs for s in M.LIBDEVICE)
    asm = {e["key"]["attrs"] for e in REG["entries"] if e["key"]["op"] == "tt.elementwise_inline_asm"}
    assert all(f"asm~{p}" in asm for p in M.INLINE_ASM)


def test_committed_registry_is_current():
    committed = json.loads((ROOT / "results/general/rule_registry.json").read_text())
    assert committed == json.loads(json.dumps(REG, ensure_ascii=False))


GENERIC = ["src/kernel_analyzer/check.py", "src/kernel_analyzer/measure.py"] + \
    [str(p.relative_to(ROOT)) for p in (ROOT / "src/kernel_analyzer/reference_eval").glob("*.py")]


def test_no_semantic_branch_on_kernel_names_in_the_generic_path():
    """kernel names may label reports; they must not decide semantics (no if / comparison / match on them)."""
    pat = re.compile(r"(kernel_name|func\.name|kernel\.name)")
    bad = []
    for f in GENERIC:
        tree = ast.parse((ROOT / f).read_text())
        for node in ast.walk(tree):
            tests = []
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                tests.append(node.test)
            if isinstance(node, ast.Compare):
                tests.append(node)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                    node.func.attr in ("startswith", "endswith", "match", "search", "fullmatch"):
                tests.append(node)
            for t in tests:
                if pat.search(ast.unparse(t)):
                    bad.append(f"{f}:{node.lineno}: {ast.unparse(t)[:100]}")
    assert not bad, bad
