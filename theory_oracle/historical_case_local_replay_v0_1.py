#!/usr/bin/env python
"""Minimal blind local-replay slice for the adaptive-pool reduction case.

This script deliberately stops before repair or root-cause claims.  It tests
whether a same-input boundary replay can distinguish a local producer from a
full-program observation and records the provenance emitted by Inductor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable


def tensor_hash(value: Any) -> dict[str, Any]:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "device": str(value.device),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(left: Any, right: Any) -> bool:
    return bool((left.detach() == right.detach()).all().item())


def delta(left: Any, right: Any) -> dict[str, Any]:
    value = (left.detach().float() - right.detach().float()).abs()
    return {
        "max_abs": float(value.max().item()),
        "mean_abs": float(value.mean().item()),
        "nonzero": int((value != 0).sum().item()),
    }


def program(torch: Any, value: Any) -> Any:
    pooled = torch.nn.functional.adaptive_avg_pool2d(value, 7)
    return pooled.flatten(1).sum(dim=-1)


def pool_only(torch: Any, value: Any) -> Any:
    return torch.nn.functional.adaptive_avg_pool2d(value, 7)


def reduce_only(value: Any) -> Any:
    return value.flatten(1).sum(dim=-1)


def noop_program(torch: Any, value: Any) -> Any:
    pooled = torch.nn.functional.adaptive_avg_pool2d(value, 7)
    return (pooled + 0.0).flatten(1).sum(dim=-1)


def compile_once(torch: Any, fn: Callable[..., Any], cache_dir: Path, value: Any) -> Any:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor")
    result = compiled(value)
    torch.cuda.synchronize()
    return result, compiled


def provenance(cache_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(cache_dir.glob("**/*.debug/output_code.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        kernels = re.findall(r"# kernel path: (.+)", text)
        sources = re.findall(r"# Topologically Sorted Source Nodes: (.+)", text)
        atens = re.findall(r"Original ATen: (.+)", text)
        rows.append(
            {
                "output_code": str(path),
                "output_code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kernel_paths": kernels,
                "source_nodes": sources,
                "original_aten": atens,
            }
        )
    return {"artifact_count": len(rows), "artifacts": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--input-artifact", required=True)
    parser.add_argument("--reference-artifact", required=True)
    args = parser.parse_args()

    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(Path(args.case_manifest).read_text(encoding="utf-8"))
    input_path = Path(args.input_artifact).resolve()
    reference_path = Path(args.reference_artifact).resolve()
    if manifest["input"]["sha256"] != file_sha256(input_path):
        raise RuntimeError("input artifact hash does not match case manifest")
    if manifest["reference"]["sha256"] != file_sha256(reference_path):
        raise RuntimeError("reference artifact hash does not match case manifest")
    x = torch.load(input_path, map_location="cuda").to(device="cuda")
    expected_reference = torch.load(reference_path, map_location="cuda").to(device="cuda")
    root = Path(args.cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    eager_full = program(torch, x.clone())
    eager_pool = pool_only(torch, x.clone())
    eager_suffix = reduce_only(eager_pool.clone())

    full_compiled, _ = compile_once(torch, lambda value: program(torch, value), root / "full", x.clone())
    pool_compiled, _ = compile_once(torch, lambda value: pool_only(torch, value), root / "pool", x.clone())
    suffix_compiled, suffix_fn = compile_once(torch, reduce_only, root / "suffix", eager_pool.clone())
    noop_eager = noop_program(torch, x.clone())
    noop_compiled, _ = compile_once(torch, lambda value: noop_program(torch, value), root / "noop", x.clone())

    # Repeat the complete compiled call without changing inputs or compilation.
    full_repeat, _compiled_again = compile_once(torch, lambda value: program(torch, value), root / "full_repeat", x.clone())

    # Same-input local replay: both suffixes receive the identical eager pool tensor.
    local_compiled_on_eager_boundary = suffix_compiled
    # Fixed compiled suffix on two boundary states: only the boundary value changes.
    suffix_on_compiled_boundary = suffix_fn(pool_compiled.clone())
    torch.cuda.synchronize()

    input_ref = tensor_hash(eager_pool)
    input_local = tensor_hash(eager_pool.clone())
    local_production = not exact(eager_suffix, local_compiled_on_eager_boundary)
    boundary_mediation = not exact(local_compiled_on_eager_boundary, suffix_on_compiled_boundary)
    artifact_provenance = {
        name: provenance(root / name)
        for name in ("full", "pool", "suffix", "noop")
    }
    provenance_complete = all(item["artifact_count"] > 0 for item in artifact_provenance.values())
    report = {
        "schema_version": "forkcert.historical_case_local_replay.v0.1",
        "case_id": manifest["case_id"],
        "case_manifest": {
            "path": str(Path(args.case_manifest).resolve()),
            "sha256": file_sha256(Path(args.case_manifest).resolve()),
            "input_artifact_sha256": file_sha256(input_path),
            "reference_artifact_sha256": file_sha256(reference_path),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "input": tensor_hash(x),
        "complete_witness": {
            "eager": tensor_hash(eager_full),
            "reference_artifact": tensor_hash(expected_reference),
            "eager_matches_reference_artifact": exact(eager_full, expected_reference),
            "compiled": tensor_hash(full_compiled),
            "compiled_repeat": tensor_hash(full_repeat),
            "eager_vs_compiled": delta(eager_full, full_compiled),
            "compiled_repeat_exact": exact(full_compiled, full_repeat),
        },
        "local_replay": {
            "boundary_reference": input_ref,
            "boundary_replay_input": input_local,
            "boundary_inputs_exact": input_ref == input_local,
            "eager_suffix": tensor_hash(eager_suffix),
            "compiled_suffix_on_same_input": tensor_hash(local_compiled_on_eager_boundary),
            "production": delta(eager_suffix, local_compiled_on_eager_boundary),
            "production_observed": local_production,
            "mediation": delta(local_compiled_on_eager_boundary, suffix_on_compiled_boundary),
            "mediation_observed": boundary_mediation,
            "compiled_pool_vs_eager_pool": delta(eager_pool, pool_compiled),
            "compiled_pool_exact": exact(eager_pool, pool_compiled),
        },
        "controls": {
            "noop_eager": tensor_hash(noop_eager),
            "noop_compiled": tensor_hash(noop_compiled),
            "noop_delta": delta(noop_eager, noop_compiled),
        },
        "provenance": artifact_provenance,
        "gates": {
            "complete_witness": bool(not exact(eager_full, full_compiled)),
            "same_input_local_replay": input_ref == input_local,
            "local_discrepancy_reproducible": local_production,
            "provenance_complete": provenance_complete,
            "candidate_realization_preserved": False,
            "intervention_executed": False,
            "oracle_recomputed": False,
            "non_target_context_invariant": False,
            "lower_level_replay": False,
            "first_bad_stage_isolated": False,
            "null_controls_valid": bool(exact(full_compiled, full_repeat)),
        },
        "allowed_claim_level": "LOCAL_INJECTION" if local_production else "OBSERVATION",
        "limitations": [
            "no repair or injection was executed",
            "same-input local replay uses an isolated compiled suffix, not a claim about a unique compiler root cause",
            "generated-kernel provenance is recorded but does not prove the first bad stage",
            "the artifact scan does not prove autotuning/runtime identity beyond this process",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("case_id", "complete_witness", "local_replay", "gates", "allowed_claim_level")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
