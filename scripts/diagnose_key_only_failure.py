"""Observe the frozen KEY_ONLY stream without changing its update arithmetic."""
import argparse
import json
from pathlib import Path

from scripts import run_structured_residual_training as frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(frozen.ROOT) or output.exists():
        parser.error("choose a new output directory inside the repository")
    import torch
    protocol = frozen.checked(frozen.ROOT / "results/property/result_analysis_v3/structured_training")
    output.mkdir(parents=True)
    frozen.save_new(output / "protocol.json", {
        "data_use": "POST_OUTCOME_FAILURE_DIAGNOSIS_NOT_NEW_INDEPENDENT_RUN",
        "stream": 7, "condition": "KEY_ONLY", "observe_from_step": 900,
        "runner": str(Path(__file__).resolve()),
        "frozen_runner": str(Path(frozen.__file__).resolve()),
        "purpose": "Locate first nonfinite parameter, gradient or negative effective second moment",
        "arithmetic_changed": False,
    })
    original_factory = frozen.make_optimizer

    def stats(tensor):
        value = tensor.detach().float()
        finite = torch.isfinite(value)
        return {"nonfinite": int((~finite).sum()),
                "negative": int((value < 0).sum()),
                "min": float(value.min()) if bool(finite.all()) else None,
                "max_abs": float(value.abs().max()) if bool(finite.all()) else None}

    def factory(condition, model, settings):
        optimizer = original_factory(condition, model, settings)
        names = {id(p): n for n, p in model.named_parameters()}
        original_step = optimizer.step
        count = 0

        def step():
            nonlocal count
            count += 1
            rows = []
            if count >= 900:
                for item in optimizer.optimizers:
                    for group in item.param_groups:
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            state = item.state[p]
                            second = item._read(state["exp_avg_sq"], state["exp_avg_sq_compensation"])
                            next_second = second.lerp(p.grad.float().square(), 1.0 - group["betas"][1])
                            rows.append({"name": names[id(p)], "compensated": item.compensation_enabled,
                                         "gradient": stats(p.grad), "parameter_before": stats(p),
                                         "effective_second": stats(second), "next_second": stats(next_second)})
                if count == 930:
                    torch.save({"model": model.state_dict(),
                                "optimizers": [o.state_dict() for o in optimizer.optimizers],
                                "gradients": {n: p.grad for n, p in model.named_parameters()},
                                "before_step": count}, output / "before_step_930.pt")
            original_step()
            if rows:
                params = dict(model.named_parameters())
                for row in rows:
                    row["parameter_after"] = stats(params[row["name"]])
                frozen.save_new(output / f"step_{count:04d}.json", {"step": count, "parameters": rows})
                print(json.dumps({"step": count, "nonfinite_parameters": [r["name"] for r in rows if r["parameter_after"]["nonfinite"]]}), flush=True)
        optimizer.step = step
        return optimizer

    frozen.make_optimizer = factory
    try:
        result = frozen.run_condition(protocol, 7, "KEY_ONLY", args.device)
        frozen.save_new(output / "run.json", result)
    except Exception as error:
        frozen.save_new(output / "terminal.json", {"error_type": type(error).__name__, "error": str(error),
                                                   "status": "NUMERICAL_FAILURE" if "nonfinite loss" in str(error) else "EXECUTION_FAILURE"})
        print(str(error), flush=True)
        if "nonfinite loss" not in str(error):
            raise


if __name__ == "__main__":
    main()
