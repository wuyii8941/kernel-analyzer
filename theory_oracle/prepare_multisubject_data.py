#!/usr/bin/env python
"""Download and freeze non-discrepancy-selected state banks for real subjects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def digest_rows(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/external_datasets")
    parser.add_argument("--count", type=int, default=128)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "hf_cache"

    specifications = [
        {
            "name": "sst2",
            "repo": "stanfordnlp/sst2",
            "split": "validation",
            "columns": ["idx", "sentence", "label"],
        },
        {
            "name": "cifar10",
            "repo": "uoft-cs/cifar10",
            "split": "test",
            "columns": ["img", "label"],
        },
    ]
    manifest = {"schema_version": "forkcert.multisubject-state-bank.v1", "count_per_partition": args.count, "datasets": []}
    for spec in specifications:
        dataset = load_dataset(spec["repo"], split=spec["split"], cache_dir=str(cache))
        if len(dataset) < 2 * args.count:
            raise ValueError(f"{spec['repo']} has only {len(dataset)} rows")
        dataset_entry = {key: spec[key] for key in ["name", "repo", "split"]}
        dataset_entry["source_rows"] = len(dataset)
        dataset_entry["partitions"] = []
        for partition, start in [("discovery", 0), ("confirmation", args.count)]:
            stop = start + args.count
            selected = dataset.select(range(start, stop))
            destination = root / f"{spec['name']}_{partition}_{args.count}"
            if destination.exists():
                raise FileExistsError(destination)
            selected.save_to_disk(destination)
            identity_rows = []
            for offset, row in enumerate(selected):
                item = {"source_index": start + offset, "label": int(row["label"])}
                if spec["name"] == "sst2":
                    item["idx"] = int(row["idx"])
                    item["sentence"] = row["sentence"]
                else:
                    image = row["img"]
                    item["image_mode"] = image.mode
                    item["image_size"] = list(image.size)
                    item["image_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
                identity_rows.append(item)
            identity_path = root / f"{spec['name']}_{partition}_{args.count}.jsonl"
            with identity_path.open("w", encoding="utf-8") as handle:
                for row in identity_rows:
                    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            dataset_entry["partitions"].append(
                {
                    "name": partition,
                    "source_range": [start, stop],
                    "saved_to": str(destination),
                    "identity_file": str(identity_path),
                    "identity_sha256": digest_rows(identity_rows),
                    "label_counts": {
                        str(label): sum(int(row["label"]) == label for row in identity_rows)
                        for label in sorted({int(row["label"]) for row in identity_rows})
                    },
                }
            )
        manifest["datasets"].append(dataset_entry)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
