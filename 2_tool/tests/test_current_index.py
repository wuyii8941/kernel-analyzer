"""The single current index (CURRENT.json) and the documents of the four parts: every path the index names exists, every
relative markdown link in the repository's own documents resolves, the versions agree, and the release fields match the
git tag."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CUR = json.loads((REPO / "CURRENT.json").read_text())
PARTS = ("1_experiments", "2_tool", "3_audits", "4_bugs_cases")
# received external originals keep their own internal links
EXTERNAL = ("3_audits/received/", "1_experiments/dsl_v2/design_rc3/", "1_experiments/specs/", "/blind_records/")


def _strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def test_every_path_in_the_current_index_exists():
    pat = re.compile(r"(?:%s)/[\w./-]*[\w/]" % "|".join(PARTS))
    missing = sorted({m.group(0) for s in _strings(CUR) for m in pat.finditer(s) if not (REPO / m.group(0)).exists()})
    assert not missing, missing


def test_the_four_parts_and_root_files_exist():
    for part in PARTS:
        assert (REPO / part / "README.md").is_file(), part
    for name in ("README.md", "CURRENT.json", "CLAUDE.md", "AGENTS.md", "pyproject.toml"):
        assert (REPO / name).is_file(), name
    stale = [d for d in ("docs", "results", "scripts", "src", "tests", "specs", "bugs", "examples", "archive")
             if subprocess.run(["git", "ls-files", "--", d], cwd=REPO, capture_output=True, text=True).stdout.strip()]
    assert not stale, f"tracked files outside the four parts: {stale}"


def test_relative_links_resolve():
    docs = [REPO / n for n in ("README.md", "CLAUDE.md", "AGENTS.md")]
    docs += [p for part in PARTS for p in (REPO / part).rglob("*.md")
             if not any(e in p.relative_to(REPO).as_posix() for e in EXTERNAL)]
    broken = []
    for doc in docs:
        text = re.sub(r"```.*?```", "", doc.read_text(), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http", "#", "mailto:")):
                continue
            path = target.split("#")[0]
            if path and not (doc.parent / path).exists():
                broken.append(f"{doc.relative_to(REPO)} -> {target}")
    assert not broken, broken[:20]


def test_versions_agree():
    tool = re.search(r'TOOL_VERSION = "([0-9.]+)"',
                     (REPO / "2_tool/src/kernel_analyzer/check.py").read_text()).group(1)
    pkg = re.search(r'^version = "([0-9.a-z]+)"', (REPO / "pyproject.toml").read_text(), re.M).group(1)
    assert CUR["versions"]["tool"]["value"] == tool == CUR["development"]["tool_version"]
    assert CUR["development"]["package_version"] == pkg
    assert pkg.startswith(tool + ".")


def test_release_and_history_tags():
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{CUR['release']['tag']}^{{commit}}"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode != 0:                                # not a git checkout with tags (e.g. an sdist)
        return
    assert r.stdout.strip() == CUR["release"]["commit"]
    tree = subprocess.run(["git", "rev-parse", f"{CUR['release']['tag']}:src"], cwd=REPO, capture_output=True,
                          text=True)
    assert tree.stdout.strip() == CUR["release"]["src_tree"]
    h = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{CUR['history_tag']}^{{commit}}"], cwd=REPO,
                       capture_output=True, text=True)
    assert h.returncode == 0, "the history tag of the reorganization is missing"
