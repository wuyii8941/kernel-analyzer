#!/usr/bin/env python3
"""CLI of the captured-package analysis (``kernel_analyzer.reference_eval.analysis``; installed as
``kernel-analyzer-analyze``).

    # composed reference for some units (can run in parallel on disjoint unit sets)
    python scripts/run_reference_analysis.py --declaration D.json --stage reference --out DIR --units unit000,unit001
    # statistics over all units of DIR
    python scripts/run_reference_analysis.py --declaration D.json --stage statistics --out DIR --report R.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.cli import analysis_main  # noqa: E402


def main():
    return analysis_main()


if __name__ == "__main__":
    sys.exit(main())
