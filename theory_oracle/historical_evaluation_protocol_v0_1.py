#!/usr/bin/env python
"""Seal a generic locator certificate or score it after an external reveal.

Usage:
  historical_evaluation_protocol_v0_1.py seal --certificate locator.json --out sealed.json
  historical_evaluation_protocol_v0_1.py score --certificate sealed.json --truth evaluator_truth.json --out score.json

The truth file is evaluator-owned and must not be present during ``seal``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forkcert.historical_evaluation import seal_pre_reveal_certificate, score_post_reveal


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal")
    seal.add_argument("--certificate", required=True, type=Path)
    seal.add_argument("--out", required=True, type=Path)
    score = commands.add_parser("score")
    score.add_argument("--certificate", required=True, type=Path)
    score.add_argument("--truth", required=True, type=Path)
    score.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "seal":
        result = seal_pre_reveal_certificate(read(args.certificate))
        write(args.out, result)
        print(json.dumps({"out": str(args.out), "certificate_sha256": result["pre_reveal"]["certificate_sha256"]}, sort_keys=True))
    else:
        result = score_post_reveal(read(args.certificate), read(args.truth))
        write(args.out, result)
        print(json.dumps({"out": str(args.out), "valid": result["valid"], "allowed_claim": result["allowed_claim"]}, sort_keys=True))


if __name__ == "__main__":
    main()
