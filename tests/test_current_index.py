"""The single current index (CURRENT.json) and the entry documents: every linked path exists, the versions agree, and
the release fields match the git tag."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUR = json.loads((ROOT / "CURRENT.json").read_text())


def _paths(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _paths(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _paths(v)
    elif isinstance(obj, str):
        for m in re.finditer(r"(?:docs|results|specs|scripts|examples|src|tests|archive|bugs)/[\w./-]+", obj):
            yield m.group(0).rstrip(".")


def test_every_path_in_the_current_index_exists():
    missing = sorted({p for p in _paths(CUR) if not (ROOT / p).exists()})
    assert not missing, missing


def test_entry_documents_link_only_existing_files():
    for doc in ("README.md", "docs/README.md", "AGENTS.md", "CLAUDE.md"):
        text = (ROOT / doc).read_text()
        base = (ROOT / doc).parent
        for target in re.findall(r"\]\(([^)#]+)\)", text):
            if target.startswith("http"):
                continue
            assert (base / target).exists(), f"{doc} links to missing {target}"


def test_versions_agree():
    tool = re.search(r'TOOL_VERSION = "([0-9.]+)"', (ROOT / "src/kernel_analyzer/check.py").read_text()).group(1)
    pkg = re.search(r'^version = "([0-9.a-z]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert CUR["versions"]["tool"]["value"] == tool
    assert CUR["release"]["package_version"] == pkg
    assert pkg.startswith(tool + ".")


def test_release_matches_the_tag():
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{CUR['release']['tag']}^{{commit}}"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:                                # not a git checkout with tags (e.g. an sdist)
        return
    assert r.stdout.strip() == CUR["release"]["commit"]
    tree = subprocess.run(["git", "rev-parse", f"{CUR['release']['tag']}:src"], cwd=ROOT, capture_output=True, text=True)
    assert tree.stdout.strip() == CUR["release"]["src_tree"]
