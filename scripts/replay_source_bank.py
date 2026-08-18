#!/usr/bin/env python3
"""Run a candidate-blind FP32/TF32 source-mapping replay over the natural bank.

This runner deliberately writes to a separate output directory.  Corrected
raw-storage dispatch results must never overwrite an older replay produced by
an invalid observer mapping.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


STEPS = (0, 1, 2, 4, 8, 16, 32, 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, choices=(64, 128, 256), required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="3")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    mapping = json.loads(args.mapping.read_text())
    if mapping.get("candidate_values_used_to_select_or_classify") is not False:
        raise SystemExit("mapping is not candidate-blind")
    denominator = mapping.get("denominator", {})
    if not denominator.get("mapped_invocations", 0):
        raise SystemExit("mapping has no mapped invocations")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("evolving_triton_observation.py")
    for step in STEPS:
        tag = "tf32" if args.tf32 else "fp32"
        output = args.output_dir / f"{tag}_seq{args.seq_len}_step{step}.json"
        if output.exists() and not args.overwrite:
            print(json.dumps({"step": step, "status": "EXISTS", "output": str(output)}))
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        env["TORCHINDUCTOR_CACHE_DIR"] = str(args.cache_dir)
        command = [
            args.python,
            str(script),
            "--seq-len", str(args.seq_len),
            "--step", str(step),
            "--dtype", "fp32",
            "--dtype-mapping", str(args.mapping),
            "--output", str(output),
        ]
        if args.tf32:
            command.append("--tf32")
        print(json.dumps({"step": step, "status": "RUNNING", "output": str(output)}), flush=True)
        subprocess.run(command, env=env, check=True)
        result = json.loads(output.read_text())
        if result.get("checkpoint_step") != step or result.get("seq_len") != args.seq_len:
            raise SystemExit(f"replay metadata mismatch at step {step}")
        if result.get("warmed_symbol_count") != result.get("expected_campaign_symbol_count"):
            raise SystemExit(f"warmed symbol gate failed at step {step}")
        if result.get("changed_region_ids_expected") != int(denominator["mapped_invocations"]):
            raise SystemExit(
                f"mapped denominator mismatch at step {step}: "
                f"worker={result.get('changed_region_ids_expected')} "
                f"mapping={denominator['mapped_invocations']}"
            )
        gates = result.get("gates", {})
        if not gates.get("all_expected_ordinary_regions_observed_twice"):
            raise SystemExit(f"ordinary region coverage gate failed at step {step}")
        if not gates.get("all_changed_region_ids_retained_twice"):
            raise SystemExit(f"mapped invocation coverage gate failed at step {step}")
        if not gates.get("all_observation_repeats_stable"):
            raise SystemExit(f"repeat stability gate failed at step {step}")
        if gates.get("candidate_values_used_to_select_regions"):
            raise SystemExit(f"candidate leakage gate failed at step {step}")
        print(json.dumps({"step": step, "status": "OK", "regions": result.get("expected_ordinary_triton_regions")}), flush=True)


if __name__ == "__main__":
    main()
