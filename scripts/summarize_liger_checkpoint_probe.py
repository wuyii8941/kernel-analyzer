#!/usr/bin/env python3
"""Join same-protocol checkpoint diagnostics to the already reported loss gap."""
import json
import math
from scripts.probe_liger_language_checkpoints import OUT, TRAINING
from scripts.run_liger_single_boundary_collapse import file_sha256
from scripts.run_training_numerical_v2 import save_new


def main():
    plan = json.loads((OUT / "plan.json").read_text())
    loss_path = TRAINING / "summary.json"
    if file_sha256(loss_path) != plan["training_summary_sha256"]:
        raise RuntimeError("Linked training result changed")
    loss = {r["pair"]: r for r in json.loads(loss_path.read_text())["rows"]}
    rows = []
    for pair in range(plan["pairs"]):
        conditions = {}
        offsets = []
        for condition in ("candidate", "reference"):
            folder = OUT / f"pair{pair}" / condition
            path = folder / "summary.json"
            if not path.exists():
                raise SystemExit(f"Incomplete: pair {pair}, {condition}; no partial-set conclusion")
            result = json.loads(path.read_text())
            if (result["status"] != "COMPLETE_FIXED_CHECKPOINT_PROBE"
                    or result["plan_sha256"] != file_sha256(OUT / "plan.json")
                    or result["checkpoint_sha256"] != plan["checkpoint_sha256"][f"pair{pair}/{condition}"]):
                raise RuntimeError("Probe provenance mismatch")
            states = [json.loads(line) for line in (folder / "states.jsonl").read_text().splitlines()]
            if [row["probe_step_index"] for row in states] != plan["training_step_indices"]:
                raise RuntimeError("Incomplete state set")
            offsets.append([row["offsets"] for row in states])
            for stage in ("gradient", "update"):
                for field in ("effect_energy_sum", "repair_energy_sum"):
                    key = "effect_energy" if field == "effect_energy_sum" else "repair_energy"
                    if not math.isclose(math.fsum(r[stage][key] for r in states), result[stage][field], rel_tol=1e-12):
                        raise RuntimeError("State energies do not reconstruct summary")
            conditions[condition] = {"gradient": result["gradient"], "update": result["update"],
                                     "source": str(path), "sha256": file_sha256(path)}
        if offsets[0] != offsets[1]:
            raise RuntimeError("Probe inputs differ across trajectory states")
        rows.append({"pair": pair, "candidate_minus_reference_final_loss": loss[pair]["candidate_minus_reference_loss"],
                     "checkpoint_diagnostics": conditions})
    values = [r["checkpoint_diagnostics"][c]["update"] for r in rows for c in ("candidate", "reference")]
    summary = {"status": "COMPLETE_SAME_TRAINING_CHECKPOINT_MECHANISM_FOLLOWUP",
        "data_use": plan["data_use"], "rows": rows, "checkpoint_count": len(values),
        "checkpoints_with_nonzero_direct_update_energy": sum(v["effect_energy_sum"] > 0 for v in values),
        "checkpoints_with_positive_split_half_mean_inner_product": sum(v["split_half_mean_inner_product"] > 0 for v in values),
        "update_total_relative_rms_range": [min(v["total_relative_rms"] for v in values), max(v["total_relative_rms"] for v in values)],
        "update_aligned_ratio_of_sums_range": [min(v["aligned_ratio_of_sums"] for v in values), max(v["aligned_ratio_of_sums"] for v in values)],
        "plan_sha256": file_sha256(OUT / "plan.json"), "training_summary_sha256": file_sha256(loss_path),
        "claim_boundary": "Direct update is measured at the real final states of the same training runs. Fixed-state batch direction is not persistence across training time. Association with a loss gap does not prove the measured component caused that gap; no new population test or threshold is introduced."}
    save_new(OUT / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
