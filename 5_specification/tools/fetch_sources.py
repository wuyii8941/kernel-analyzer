#!/usr/bin/env python3
"""Archive source documents for the contracts and register them in SOURCES.json.

    python tools/fetch_sources.py LIST.json [--proxy http://127.0.0.1:27890]

LIST.json: [{"sid": "T_ADD", "url": "...", "title": "...", "kind": "versioned_document", "version": "2.10",
             "items": {"DEF": "what the clause relies on", ...}}, ...]
or, for files of the installed torch: {"sid": ..., "file": "torch/_prims/__init__.py", "kind": "torch_source", ...},
or, for a file of this repository (e.g. an issuer's release document): {"sid": ..., "repo_file": "1_experiments/...", ...}.

For a web page the main text is extracted (scripts, navigation and styles dropped) and stored as
sources/<SID>.txt.gz; the SHA-256 of the raw HTML and of the stored text are both recorded. A torch source file is
stored as sources/<SID>.py.gz. Existing entries are replaced only with --replace.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import gzip
import hashlib
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "SOURCES.json"
STORE = ROOT / "sources"


class _Text(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr", "pre", "dt", "dd", "table", "section"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def extract(html: str) -> str:
    p = _Text()
    p.feed(html)
    text = re.sub(r"[ \t]+\n", "\n", "".join(p.out))
    text = re.sub(r"\n\s*\n+", "\n", text)
    m = re.search(r"\n\s*([\w .()-]+)#\n", text)          # the article starts at its first heading anchor ("torch.div#")
    if m:
        text = text[m.start() + 1:]
    for marker in ("\nPrevious\n", "\nNext\n", "\nTo analyze traffic"):
        end = text.find(marker)
        if end > 0:
            text = text[:end]
    return text.strip() + "\n"


def is_code(url: str, raw: bytes) -> bool:
    """Raw source files are archived byte for byte; only HTML pages go through text extraction."""
    head = raw[:512].lstrip().lower()
    return ("raw.githubusercontent.com" in url or url.rsplit(".", 1)[-1] in ("py", "txt", "json", "yaml", "md", "c", "cpp", "h",
                                                                              "cu")) and not head.startswith((b"<!doctype", b"<html"))


def snapshot_url(sid, url, proxy):
    raw = fetch(url, proxy)
    if is_code(url, raw):
        ext = "." + url.rsplit(".", 1)[-1] if "." in url.rsplit("/", 1)[-1] else ".txt"
        (STORE / f"{sid}{ext}.gz").write_bytes(gz_bytes(raw))
        return {"snapshot": f"sources/{sid}{ext}.gz", "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                "snapshot_note": "raw file, byte for byte"}
    text = extract(raw.decode("utf-8", "replace")).encode()
    (STORE / f"{sid}.txt.gz").write_bytes(gz_bytes(text))
    return {"snapshot": f"sources/{sid}.txt.gz", "snapshot_sha256": hashlib.sha256(text).hexdigest(),
            "raw_html_sha256": hashlib.sha256(raw).hexdigest(), "snapshot_note": "main text extracted from the HTML page"}


def gz_bytes(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def fetch(url, proxy):
    r = subprocess.run(["curl", "-sS", "-L", "--fail", "--max-time", "90", "-x", proxy, url], capture_output=True)
    if r.returncode:
        raise RuntimeError(f"{url}: {r.stderr.decode()[:200]}")
    return r.stdout


@contextlib.contextmanager
def locked_sources():
    """Read-modify-write of SOURCES.json under an exclusive lock (several authors may fetch at the same time)."""
    with open(ROOT / ".sources.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        S = json.loads(SOURCES.read_text())
        yield S
        SOURCES.write_text(json.dumps(S, indent=2, ensure_ascii=False) + "\n")
        fcntl.flock(lk, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list", type=Path)
    ap.add_argument("--proxy", default="http://127.0.0.1:27890")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--items-only", action="store_true", help="update title / items / kind of existing entries, no fetch")
    ap.add_argument("--resnapshot", action="store_true",
                    help="LIST is a JSON list of SIDs: re-archive their URL, keeping every other field of the entry")
    a = ap.parse_args()
    todo = json.loads(a.list.read_text())
    if a.resnapshot:
        S0 = json.loads(SOURCES.read_text())
        new = {}
        for sid in todo:
            old = S0[sid]
            for k in ("raw_html_sha256", "snapshot_note"):
                old.pop(k, None)
            stale = ROOT / old["snapshot"]
            new[sid] = snapshot_url(sid, old["url"], a.proxy)
            if stale.is_file() and str(stale) != str(ROOT / new[sid]["snapshot"]):
                stale.unlink()
            print(f"{sid}: {new[sid]['snapshot']} {new[sid]['snapshot_sha256'][:12]}")
        with locked_sources() as S:
            for sid, upd in new.items():
                for k in ("raw_html_sha256", "snapshot_note"):
                    S[sid].pop(k, None)
                S[sid].update(upd)
                S[sid]["resnapshot"] = datetime.date.today().isoformat() + ": raw file archived (was HTML text extraction)"
        return 0
    if a.items_only:
        with locked_sources() as S:
            for e in todo:
                if e["sid"] in S:
                    for k in ("items", "title", "kind", "version", "identity", "note"):
                        if k in e:
                            S[e["sid"]][k] = e[k]
                    print(f"{e['sid']}: items updated")
                else:
                    print(f"{e['sid']}: not registered (fetch it first)")
        return 0
    S = json.loads(SOURCES.read_text())
    STORE.mkdir(exist_ok=True)
    today = datetime.date.today().isoformat()
    failed = []
    for e in todo:
        sid = e["sid"]
        if sid in S and not a.replace:
            continue
        entry = {k: v for k, v in e.items() if k not in ("sid", "file", "repo_file")}
        try:
            if "repo_file" in e:
                path = ROOT.parent / e["repo_file"]
                raw = path.read_bytes()
                (STORE / f"{sid}{path.suffix}.gz").write_bytes(gz_bytes(raw))
                entry.update(snapshot=f"sources/{sid}{path.suffix}.gz", snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                             path=e["repo_file"])
            elif "file" in e:
                import torch
                path = Path(torch.__file__).parent.parent / e["file"]
                raw = path.read_bytes()
                (STORE / f"{sid}.py.gz").write_bytes(gz_bytes(raw))
                entry.update(snapshot=f"sources/{sid}.py.gz", snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                             path=e["file"], torch_version=torch.__version__)
            else:
                entry.update(snapshot_url(sid, e["url"], a.proxy))
        except Exception as err:  # noqa: BLE001
            failed.append(f"{sid}: {err}")
            continue
        entry.setdefault("retrieved", today)
        with locked_sources() as S2:
            if sid in S2 and not a.replace:
                continue
            S2[sid] = entry
        print(f"{sid}: {entry['snapshot']} {entry['snapshot_sha256'][:12]}")
    for f in failed:
        print("FAILED", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
