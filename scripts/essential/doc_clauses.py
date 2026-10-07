"""Protocol section 1.4: the text of every documentation page the phase-1 specs cite, for the locked 2.10 docs and main,
extracted from the article body and compared sentence by sentence.

    python -I scripts/essential/doc_clauses.py .cache/essential/docs results/essential/phase1/doc_clauses.json
"""
import difflib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class _Article(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth, self.parts, self.skip = 0, [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "article" or (self.depth == 0 and a.get("role") == "main"):
            self.depth = 1
        elif self.depth:
            self.depth += 1
            if tag in ("script", "style") or "headerlink" in (a.get("class") or ""):
                self.skip += 1

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1
            if self.skip and tag in ("script", "style", "a"):
                self.skip -= 1

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.parts.append(data)


def text(path):
    p = _Article()
    p.feed(Path(path).read_text(encoding="utf-8", errors="replace"))
    t = re.sub(r"\s+", " ", " ".join(p.parts)).strip()
    t = re.sub(r"\[source\]|#\s", " ", t)
    return [s.strip() for s in re.split(r"(?<=[.:;])\s+", t) if s.strip()]


def main(root, out):
    root = Path(root)
    res = {}
    for f in sorted((root / "2.10").glob("*.html")):
        a, b = text(f), text(root / "main" / f.name)
        diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
        res[f.stem] = {"sentences_2.10": len(a), "sentences_main": len(b), "differences": diff}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for k, v in res.items():
        print(k, v["sentences_2.10"], v["sentences_main"], len(v["differences"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
