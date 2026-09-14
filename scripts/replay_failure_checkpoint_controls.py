"""Compare residual readback controls from one saved pre-failure training state."""
import argparse
import copy
import json
from pathlib import Path

from scripts import run_structured_residual_training as frozen


def restore_optimizer(item, saved):
    """Preserve BF16 residual storage that standard optimizer loading casts to FP32."""
    item.load_state_dict(copy.deepcopy(saved))
    for group, saved_group in zip(item.param_groups, saved["param_groups"]):
        for parameter, identifier in zip(group["params"], saved_group["params"]):
            original_state = saved["state"][identifier]
            for key in ("exp_avg_compensation", "exp_avg_sq_compensation"):
                value = original_state[key]
                item.state[parameter][key] = None if value is None else value.to(parameter.device).clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(frozen.ROOT):
        parser.error("choose new repository output")
    import torch
    from transformers import AutoModelForCausalLM
    protocol = frozen.checked(frozen.ROOT / "results/property/result_analysis_v3/structured_training")
    # Keep CPU lr/step scalars on CPU, as in the frozen optimizer arithmetic.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    start = checkpoint["before_step"]
    frozen.save_new(output / "protocol.json", {
        "data_use": "POST_OUTCOME_COMMON_STATE_INTERVENTION",
        "checkpoint": str(args.checkpoint.resolve()),
        "script": str(Path(__file__).resolve()),
        "start": start, "end": 940, "stream": 7,
        "conditions": ["UNCHANGED_KEY_ONLY", "ENABLE_ALL_READBACK", "DISABLE_ALL_READBACK"],
        "scope": "Switch read multipliers from same historical state; does not erase earlier trajectory differences.",
        "prediction": "If immediate residual readback outside selected parameters is sufficient, ENABLE_ALL avoids the observed failure; otherwise no sufficiency claim.",
        "negative_result_allowed": True,
    })
    train = frozen.load(Path(protocol["train_banks"][7]))["states"]
    settings = dict(protocol["optimizer"])
    settings = {k: settings[k] for k in ("lr", "betas", "eps", "weight_decay")}
    settings["betas"] = tuple(settings["betas"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for condition in ("UNCHANGED_KEY_ONLY", "ENABLE_ALL_READBACK", "DISABLE_ALL_READBACK"):
        model = AutoModelForCausalLM.from_pretrained(frozen.original.MODEL, dtype=torch.float32, local_files_only=True).to(args.device)
        model.config.use_cache = False
        model.load_state_dict(checkpoint["model"])
        optimizer = frozen.make_optimizer("KEY_ONLY", model, settings)
        for item, saved in zip(optimizer.optimizers, checkpoint["optimizers"]):
            restore_optimizer(item, saved)
            for group in item.param_groups:
                if isinstance(group["lr"], torch.Tensor) and group["lr"].device.type != "cpu":
                    raise ValueError("restored learning-rate scalar moved off CPU")
                for parameter in group["params"]:
                    if item.state[parameter]["step"].device.type != "cpu":
                        raise ValueError("restored step scalar moved off CPU")
            if condition != "UNCHANGED_KEY_ONLY":
                item.compensation_enabled = condition == "ENABLE_ALL_READBACK"
        model.train()
        rows = []
        for step in range(start, 941):
            torch.manual_seed(7000000 + 193000 + step - 1)
            torch.cuda.manual_seed_all(7000000 + 193000 + step - 1)
            optimizer.zero_grad(set_to_none=True)
            if step == start:
                for name, p in model.named_parameters():
                    gradient = checkpoint["gradients"][name]
                    p.grad = None if gradient is None else gradient.to(args.device).clone()
                loss_value = None
            else:
                ids = torch.tensor([train[step-1]["token_ids"]],device=args.device)
                loss = model(input_ids=ids, labels=ids).loss
                if not bool(torch.isfinite(loss)):
                    rows.append({"step": step, "status": "NONFINITE_LOSS"})
                    break
                loss_value = float(loss.detach())
                loss.backward()
            optimizer.step()
            bad = [name for name,p in model.named_parameters() if not bool(torch.isfinite(p).all())]
            rows.append({"step":step,"loss":loss_value,"nonfinite_parameters":bad})
            print(json.dumps({"condition":condition,"step":step,"nonfinite_parameter_count":len(bad)}),flush=True)
        frozen.save_new(output / (condition + ".json"), {"condition":condition,"rows":rows})
        del optimizer, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
