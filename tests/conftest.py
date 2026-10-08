"""Session guard: the test suite must not modify any tracked file (frozen results, protocols, answers, checksums).

At session start the tracked files already modified in the working tree are recorded with their content hashes; at
session end any tracked file that became modified, or whose content changed, fails the session.  Nothing is restored
automatically: a test that writes into the repository has to be fixed to write under tmp_path.  Skipped when the
tests do not run inside a git checkout."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _modified():
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    files = {}
    for line in out.stdout.splitlines():
        path = line[3:].split(" -> ")[-1]
        p = ROOT / path
        files[path] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return files


PROTECTED = ["results", "docs", "specs", "bugs", "tests/data"]


def _mtimes():
    """modification times of the tracked files under the protected directories: a rewrite with identical bytes is
    still a rewrite of a frozen record."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--"] + PROTECTED, cwd=ROOT, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    res = {}
    for path in out.stdout.decode().split("\0"):
        if path:
            try:
                res[path] = (ROOT / path).stat().st_mtime_ns
            except OSError:
                res[path] = None
    return res


def pytest_sessionstart(session):
    session.config._tracked_before = _modified()
    session.config._mtimes_before = _mtimes()


def pytest_sessionfinish(session, exitstatus):
    before = getattr(session.config, "_tracked_before", None)
    after = _modified()
    if before is None or after is None:
        return
    changed = sorted(p for p, h in after.items() if p not in before or before[p] != h)
    m0, m1 = getattr(session.config, "_mtimes_before", None), _mtimes()
    if m0 and m1:
        changed += sorted(p for p, t in m1.items() if p in m0 and m0[p] != t and p not in changed)
    if changed:
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        msg = "tests modified or rewrote tracked files (write under tmp_path instead): " + ", ".join(changed[:20])
        if tr is not None:
            tr.write_line(msg, red=True)
        session.exitstatus = 1
