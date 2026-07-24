#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace one Phase 1.5 level with a rerun measurement.")
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--replacement", required=True)
    parser.add_argument("--level", required=True)
    args = parser.parse_args()

    path = Path(args.measurements)
    rows = read_rows(path)
    replacements = [row for row in read_rows(Path(args.replacement)) if str(row.get("level")) == args.level]
    if len(replacements) != 1:
        raise SystemExit(f"expected exactly one replacement for {args.level}, found {len(replacements)}")
    replaced = 0
    output = []
    for row in rows:
        if str(row.get("level")) == args.level:
            output.append(replacements[0])
            replaced += 1
        else:
            output.append(row)
    if replaced != 1:
        raise SystemExit(f"expected exactly one existing {args.level} row, found {replaced}")
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"path": str(path), "level": args.level, "rows": len(output)}, indent=2))


if __name__ == "__main__":
    main()
