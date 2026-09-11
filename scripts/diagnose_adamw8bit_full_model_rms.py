#!/usr/bin/env python3
"""Result-aware per-parameter audit for the large full-model RMS ratio."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.run_adamw8bit_full_model_block_probe import (
    MODEL, load, make_parameter_copies, token_ids, verify_protocol,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("choose a new output under /data1/tzh")

    import torch
    from transformers import AutoModelForCausalLM
    protocol = verify_protocol(args.root)
    unit_index = 0
    indices = protocol["unit_population_indices"][unit_index]
    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True,
    ).to(args.device).eval()
    model.config.use_cache = False
    names, base, copies, optimizers = make_parameter_copies(model, args.device)
    final_gradients = None
    for population_index_value in indices:
        model.zero_grad(set_to_none=True)
        tokens = torch.tensor([token_ids(protocol, population_index_value)],
                              dtype=torch.long, device=args.device)
        model(input_ids=tokens, labels=tokens).loss.backward()
        gradients = [parameter.grad.detach() for _, parameter in model.named_parameters()]
        for key in copies:
            with torch.no_grad():
                for target, source in zip(copies[key], base):
                    target.copy_(source)
            for target, gradient in zip(copies[key], gradients):
                target.grad = gradient.clone()
            optimizers[key].step()
        final_gradients = gradients

    rows = []
    for index, (name, source) in enumerate(zip(names, base)):
        reference_write = copies["reference"][index].detach() - source
        row = {
            "parameter": name,
            "coordinate_count": source.numel(),
            "reference_write_energy": float(torch.sum(reference_write.double().square()).item()),
            "reference_nonzero_writes": int(torch.count_nonzero(reference_write).item()),
        }
        for key in ("block64", "block256"):
            write = copies[key][index].detach() - source
            effect = write - reference_write
            row[key + "_effect_energy"] = float(torch.sum(effect.double().square()).item())
            row[key + "_effect_nonzero"] = int(torch.count_nonzero(effect).item())
            row[key + "_effect_gradient_inner"] = float(torch.sum(
                effect.double() * final_gradients[index].double()
            ).item())
        rows.append(row)

    total_repair = math.fsum(row["reference_write_energy"] for row in rows)
    totals = {key: math.fsum(row[key + "_effect_energy"] for row in rows)
              for key in ("block64", "block256")}
    saved = load(args.root / "units" / f"unit-{unit_index:03d}.json")
    recomputed_rms = {key: math.sqrt(totals[key] / total_repair)
                      for key in ("block64", "block256")}
    rows.sort(key=lambda row: row["block256_effect_energy"], reverse=True)
    result = {
        "schema": "adamw8bit-full-model-rms-diagnostic-v1",
        "status": "COMPLETE",
        "data_use": "RESULT_AWARE_NUMERICAL_AUDIT_NOT_CONFIRMATION",
        "unit_index": unit_index,
        "recomputed_full_model_rms": recomputed_rms,
        "saved_full_model_rms": saved["full_model_write_rms"],
        "aggregate_matches_saved": all(math.isclose(
            recomputed_rms[key], saved["full_model_write_rms"][key], rel_tol=1e-12
        ) for key in recomputed_rms),
        "total_reference_write_energy": total_repair,
        "parameters_with_zero_reference_write": sum(
            row["reference_write_energy"] == 0 for row in rows
        ),
        "top_parameters_by_block256_effect_energy": rows[:20],
        "all_parameter_statistics": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "aggregate_matches_saved", "recomputed_full_model_rms",
        "parameters_with_zero_reference_write", "total_reference_write_energy"
    )}, indent=2))


if __name__ == "__main__":
    main()
