#!/usr/bin/env python
"""Freeze the preregistered SST-2 impact state bank and identity file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--identity-jsonl", required=True)
    parser.add_argument("--start", type=int, default=256)
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    identity_path = Path(args.identity_jsonl)
    if out_dir.exists() or identity_path.exists():
        raise FileExistsError("impact bank or identity file already exists")
    dataset = load_dataset(
        "stanfordnlp/sst2", split="validation", cache_dir=args.cache_dir
    )
    stop = args.start + args.count
    if stop > len(dataset):
        raise ValueError("requested range exceeds SST-2 validation split")
    selected = dataset.select(range(args.start, stop))
    selected.save_to_disk(out_dir)
    rows = []
    for offset, row in enumerate(selected):
        rows.append(
            {
                "source_index": args.start + offset,
                "idx": int(row["idx"]),
                "sentence": row["sentence"],
                "label": int(row["label"]),
            }
        )
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    with identity_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    summary = {
        "source": "stanfordnlp/sst2 validation",
        "source_range": [args.start, stop],
        "count": len(rows),
        "identity_sha256": hashlib.sha256(payload).hexdigest(),
        "label_counts": {
            str(label): sum(row["label"] == label for row in rows)
            for label in sorted({row["label"] for row in rows})
        },
        "saved_to": str(out_dir),
        "identity_file": str(identity_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
