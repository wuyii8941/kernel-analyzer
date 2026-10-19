#!/usr/bin/env python3
"""Archive source documents for the contracts and register them in SOURCES.json.

    python tools/fetch_sources.py LIST.json [--proxy http://127.0.0.1:27891]

LIST.json: [{"sid": "T_ADD", "url": "...", "title": "...", "kind": "versioned_document", "version": "2.10",
             "items": {"DEF": "what the clause relies on", ...}}, ...]
or, for files of the installed torch: {"sid": ..., "file": "torch/_prims/__init__.py", "kind": "torch_source", ...},
or, for a file of this repository (e.g. an issuer's release document): {"sid": ..., "repo_file": "1_experiments/...", ...}.

For a web page the main text is extracted (scripts, navigation and styles dropped) and stored as
sources/<SID>.txt.gz. Raw code is kept byte for byte. No hashes are generated.
Existing snapshots and source identities are never replaced: use a new SID for
a new version. Archiving a file does not verify its claimed upstream version.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import fcntl
import gzip
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

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
        ext = Path(urlsplit(url).path).suffix or '.txt'
        return {"snapshot": save_snapshot(sid, ext, raw), "snapshot_note": "raw file, byte for byte",
                "retrieval_method": "requested_url"}
    text = extract(raw.decode("utf-8", "replace")).encode()
    return {"snapshot": save_snapshot(sid, '.txt', text),
            "snapshot_note": "main text extracted from the HTML page", "retrieval_method": "requested_url"}


def save_snapshot(sid, ext, raw):
    if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9_]+', sid):
        raise ValueError('source ID must contain only letters, digits and underscores')
    if not re.fullmatch(r'\.[A-Za-z0-9]+', ext):
        raise ValueError('unsupported source file extension')
    path = STORE / f'{sid}{ext}.gz'
    if any(STORE.glob(f'{sid}.*.gz')):
        raise FileExistsError(f'{sid}: a retained snapshot already exists; use a new SID')
    with path.open('xb') as output:
        output.write(gz_bytes(raw))
    return f'sources/{path.name}'


def gz_bytes(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def fetch(url, proxy):
    if not isinstance(proxy, str) or not proxy.strip():
        raise ValueError('an explicit proxy is required; no direct-network fallback')
    r = subprocess.run(["curl", "-sS", "-L", "--fail", "--max-time", "90", "--noproxy", "", "-x", proxy, url], capture_output=True)
    if r.returncode:
        raise RuntimeError(f"{url}: {r.stderr.decode()[:200]}")
    return r.stdout


@contextlib.contextmanager
def locked_sources():
    """Read-modify-write of SOURCES.json under an exclusive lock (several authors may fetch at the same time)."""
    with open(ROOT / ".sources.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        S = json.loads(SOURCES.read_text())
        before = copy.deepcopy(S)
        yield S
        if S != before:
            SOURCES.write_text(json.dumps(S, indent=2, ensure_ascii=False) + "\n")
        fcntl.flock(lk, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list", type=Path)
    ap.add_argument("--proxy", default="http://127.0.0.1:27891")
    ap.add_argument("--replace", action="store_true", help="retired: use a new source ID")
    ap.add_argument("--items-only", action="store_true", help="update title / items / kind of existing entries, no fetch")
    ap.add_argument("--resnapshot", action="store_true",
                    help="retired: use a new source ID")
    a = ap.parse_args()
    if a.replace or a.resnapshot:
        print('FAILED: existing sources are immutable; archive the new version under a new SID')
        return 1
    todo = json.loads(a.list.read_text())
    if (not isinstance(todo, list) or not todo or any(not isinstance(e, dict)
            or not isinstance(e.get('sid'), str) or not re.fullmatch(r'[A-Za-z0-9_]+', e['sid']) for e in todo)
            or len({e['sid'] for e in todo}) != len(todo)):
        print('FAILED: require a nonempty list with distinct valid source IDs')
        return 1
    if a.items_only:
        with locked_sources() as S:
            for e in todo:
                if e['sid'] not in S:
                    print(f"FAILED: {e['sid']} is not registered")
                    return 1
                for key in set(e) - {'sid', 'title', 'kind', 'note', 'items'}:
                    if e[key] != S[e['sid']].get(key):
                        print(f"FAILED: {e['sid']} cannot change source {key}; use a new SID")
                        return 1
            for e in todo:
                if e["sid"] in S:
                    for k in ("title", "kind", "note"):
                        if k in e:
                            S[e["sid"]][k] = e[k]
                    if "items" in e:                      # merge, so that authors updating one source do not drop items
                        S[e["sid"]].setdefault("items", {}).update(e["items"])
                    print(f"{e['sid']}: items updated")
                else:
                    print(f"{e['sid']}: not registered (fetch it first)")
        return 0
    STORE.mkdir(exist_ok=True)
    today = datetime.date.today().isoformat()
    failed = []
    for e in todo:
        sid = e["sid"]
        entry = {k: v for k, v in e.items() if k not in ("sid", "file", "repo_file", "snapshot_sha256", "raw_html_sha256", "retrieved")}
        try:
            # One lock covers the identity decision, immutable snapshot creation
            # and registration, so concurrent authors cannot reuse one source ID.
            with locked_sources() as S:
                if sid in S:
                    for key in ('url', 'version', 'edition', 'identity'):
                        if key in e and e[key] != S[sid].get(key):
                            raise ValueError(f'existing source {key} differs; use a new SID')
                    for key in ('file', 'repo_file'):
                        if key in e and e[key] != S[sid].get('path'):
                            raise ValueError('existing source path differs; use a new SID')
                    print(f'{sid}: already retained')
                    continue
                if "repo_file" in e:
                    path = (ROOT.parent / e["repo_file"]).resolve()
                    if ROOT.parent.resolve() not in path.parents:
                        raise ValueError('repo_file must stay within the repository')
                    raw = path.read_bytes()
                    entry.update(snapshot=save_snapshot(sid, path.suffix or '.txt', raw),
                                 path=e["repo_file"], retrieval_method='repository_file')
                elif "file" in e:
                    import torch
                    torch_root = Path(torch.__file__).parent.resolve()
                    path = (torch_root.parent / e["file"]).resolve()
                    if torch_root not in path.parents:
                        raise ValueError('file must refer to a file in the installed torch package')
                    raw = path.read_bytes()
                    entry.update(snapshot=save_snapshot(sid, '.py', raw), path=e["file"],
                                 torch_version=torch.__version__, retrieval_method='installed_torch_file')
                else:
                    entry.update(snapshot_url(sid, e["url"], a.proxy))
                entry['retrieved'] = today
                S[sid] = entry
        except Exception as err:  # noqa: BLE001
            failed.append(f"{sid}: {err}")
            continue
        print(f"{sid}: {entry['snapshot']} (retained; upstream version review is separate)")
    for f in failed:
        print("FAILED", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
