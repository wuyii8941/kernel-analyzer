"""Summary of the composition cells (RQ2): status per form, per primitive family, the cells that are not bit identical
grouped by cause, and the not-established cells grouped by the emulator's reason."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def reason_of(r) -> str:
    """One line for a not-established / mismatch cell."""
    parts = set()
    for x in r.get("launches", []) or []:
        if x.get("not_emulable"):
            parts.add("not emulable: " + re.sub(r"^\S+ \(|\)$", "", x["not_emulable"][0]))
        elif x.get("aborted_programs"):
            reason = (x.get("abort_reasons") or ["?"])[0]
            reason = re.sub(r"tt\.\w+@\d+: ", "", reason)
            reason = re.sub(r"__nv_\w+", "__nv_<fn>", reason)
            parts.add("aborted: " + reason[:90])
        elif x.get("status") == "emulator error":
            parts.add("emulator error: " + (x.get("error") or "")[:90])
        elif x.get("status") == "not_established":
            parts.add("no emulated element")
    return "; ".join(sorted(parts)) or r.get("status", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, help="jsonl from cells.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.cells).read_text().splitlines() if l.strip()]
    by_form = defaultdict(Counter)
    by_family = defaultdict(Counter)
    mism = defaultdict(list)
    ne = defaultdict(list)
    elements = Counter()
    for r in rows:
        fam = r["pid"].split("|")[0] + ":" + (r["pid"].split("|")[1] if r["pid"].startswith("F1") else "libdevice")
        by_form[r["form"]][r["status"]] += 1
        by_family[r["pid"].split("|")[1] if r["pid"].startswith("F1") else "libdevice"][r["status"]] += 1
        for x in r.get("launches", []) or []:
            for k, v in (x.get("elements") or {}).items():
                elements[k] += v
        if r["status"] in ("mismatch", "emulator error", "harness error", "launch error", "build error",
                           "child crashed (compiler abort)", "not compiled"):
            mism[(r["status"], r["pid"].split("|")[1])].append(f'{r["form"]} {r["param"]}')
        elif r["status"] == "not established":
            ne[reason_of(r)].append(r["pid"])
    out = {"cells": len(rows), "status": dict(Counter(r["status"] for r in rows)), "elements": dict(elements),
           "by_form": {k: dict(v) for k, v in sorted(by_form.items(), key=lambda kv: int(kv[0][1:]))},
           "not_bit_identical": {f"{k[0]} | {k[1]}": {"cells": len(v), "forms": sorted(set(v))[:30]}
                                 for k, v in sorted(mism.items())},
           "not_established_reasons": {k: {"cells": len(v), "primitives": sorted(set(v))[:12],
                                           "distinct_primitives": len(set(v))}
                                       for k, v in sorted(ne.items(), key=lambda kv: -len(kv[1]))}}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("cells", "status", "elements")}, ensure_ascii=False))
    print("forms:", {k: v.get("bit identical", 0) for k, v in out["by_form"].items()})
    print(len(out["not_bit_identical"]), "groups not bit identical;", len(out["not_established_reasons"]),
          "not-established reasons")


if __name__ == "__main__":
    main()
