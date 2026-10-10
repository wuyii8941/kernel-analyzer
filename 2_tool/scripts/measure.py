#!/usr/bin/env python3
"""Unified entry CLI: python scripts/measure.py --declaration D.json --out R.json
(the installed package provides the same command as ``kernel-analyzer-measure``)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel_analyzer.cli import measure_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(measure_main())
