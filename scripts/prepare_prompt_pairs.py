#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE_PAIRS = [
    ("Question: What is 2 + 2?\nAnswer:", " 4"),
    ("Question: If a car travels 60 miles in 2 hours, its speed is\nAnswer:", " 30 miles per hour."),
    ("Complete the sentence: numerical stability matters because", " small errors can change decisions."),
    ("Write a Python expression for the square of x:", " x ** 2"),
    ("Question: What is the capital of France?\nAnswer:", " Paris"),
    ("Translate to French: hello", " bonjour"),
    ("Classify sentiment: I liked the clear explanation.\nLabel:", " positive"),
    ("Question: List the first three prime numbers.\nAnswer:", " 2, 3, 5"),
    ("Complete: In PPO, clipping limits", " the policy ratio update."),
    ("Question: What is 10 minus 7?\nAnswer:", " 3"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fixed prompt-response pairs for ForkCert smoke experiments.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", default="data/prompt_pairs.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            prompt, response = BASE_PAIRS[i % len(BASE_PAIRS)]
            row = {
                "case_id": f"sample_{i:06d}",
                "prompt": prompt,
                "response": response,
                "metadata": {"source": "forkcert_builtin_smoke", "template_index": i % len(BASE_PAIRS)},
            }
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {args.count} prompt-response pairs to {out}")


if __name__ == "__main__":
    main()

