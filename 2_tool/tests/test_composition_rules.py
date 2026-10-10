"""The composition-rule registry (``kernel_analyzer.composition_rules``): every implementation entry resolves, every
referenced test exists, every report field is written by the unified entry, and the expanded declaration names the
rule set."""
from __future__ import annotations

import re
from pathlib import Path

from kernel_analyzer import composition_rules as C

TOOL = Path(__file__).resolve().parents[1]


def test_registry_entries_and_tests_resolve():
    assert C.check_registry(TOOL) == []


def test_every_rule_states_the_whole_chain():
    for r in C.RULES:
        assert r["obligation"] and r["entry"] and r["independent_answer"] and r["tests"], r["id"]
        assert r["id"][0] in "MTPQK", r["id"]


def test_report_fields_are_written_by_the_unified_entry():
    src = "".join((TOOL / "src" / "kernel_analyzer" / f).read_text() for f in ("measure.py", "check.py"))
    for r in C.RULES:
        for field in r["report"]:
            key = [k for k in re.split(r"[.\[\]*]+", field) if k][-1]
            assert f'"{key}"' in src, (r["id"], field)


def test_expanded_declaration_names_the_rule_set():
    from kernel_analyzer import measure
    e = measure.expand({"call": "c.py:f", "inputs": {"x": {"sampler": {"normal": [0, 1]}, "shape": [4],
                                                            "dtype": "float32"}},
                        "compare": {"mode": "A", "measure": ["y"]},
                        "budget": {"cpu_seconds": 1, "gpu_seconds": 1, "case_timeout": 1, "max_units": 18}})
    assert e["composition_rules"]["registry_sha256"] == C.registry_digest()
    assert set(e["composition_rules"]["rules"]) == {r["id"] for r in C.RULES}
