"""Check storage-only predictions on a declared existing real-gradient history."""
import argparse
from pathlib import Path

from scripts import run_selective_parameter_compensation_probe as source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(source.ROOT):
        parser.error("choose a new repository output directory")
    import torch
    from transformers import AutoModelForCausalLM
    from kernel_analyzer.compensation_control import TensorScalarCompensationControl
    from kernel_analyzer.moment_reconstruction import reconstruction_terms, squared_norm
    original = source.verify(source.ROOT / "results/property/result_analysis_v2/selective_parameter_compensation")
    source.save_new(output / "protocol.json", {
        "data_use": "HISTORICAL_UNIT_ZERO_NEW_MECHANISM_MEASUREMENTS",
        "indices": original["unit_population_indices"][0],
        "parameters": list(source.KEY_PARAMETERS),
        "source": str(Path(__file__).resolve()),
        "hypothesis": "Storage residual recurrence predicts moment error without fitting a remainder.",
        "boundary": "One existing fixed-checkpoint history, two preselected parameters; no population or loss claim.",
    })
    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(source.MODEL, dtype=torch.float32, local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    parameters = {n: p for n, p in model.named_parameters() if n in source.KEY_PARAMETERS}
    settings = dict(original["optimizer"])
    settings["betas"] = tuple(settings["betas"])
    controls = {}
    references = {n: [torch.zeros_like(p), torch.zeros_like(p)] for n,p in parameters.items()}
    predictions = {}
    for mode in ("OFF", "ON"):
        copies = {n: torch.nn.Parameter(p.detach().clone()) for n,p in parameters.items()}
        opt = TensorScalarCompensationControl(copies.values(), compensation_enabled=mode == "ON", **settings)
        for n,p in copies.items():
            state = opt.state[p]
            state["step"] = torch.tensor(0.)
            for key,signed in (("exp_avg", True), ("exp_avg_sq", False)):
                state[key],state[key+"_compensation"] = opt._new_state(p,signed=signed)
                predictions[mode,n,key] = torch.zeros_like(p)
        controls[mode] = copies,opt
    rows=[]
    for step,index in enumerate(original["unit_population_indices"][0],1):
        model.zero_grad(set_to_none=True)
        tokens = torch.tensor([source.tokens(original,index)],device=args.device)
        model(input_ids=tokens,labels=tokens).loss.backward()
        previous_refs = {n:[v.clone() for v in values] for n,values in references.items()}
        for n,p in parameters.items():
            for i,beta in enumerate(settings["betas"]):
                signal = p.grad if i == 0 else p.grad.square()
                references[n][i] = previous_refs[n][i].lerp(signal,1-beta)
        for mode,(copies,opt) in controls.items():
            before={}
            unstored={}
            for n,p in copies.items():
                with torch.no_grad(): p.copy_(parameters[n])
                p.grad = parameters[n].grad.clone()
                for i,key in enumerate(("exp_avg","exp_avg_sq")):
                    state=opt.state[p]
                    before[n,key]=opt._read(state[key],state[key+"_compensation"]).clone()
                    unstored[n,key]=before[n,key].lerp(p.grad if i==0 else p.grad.square(),1-settings["betas"][i])
            opt.step()
            for n,p in copies.items():
                for i,key in enumerate(("exp_avg","exp_avg_sq")):
                    state=opt.state[p]
                    current=opt._read(state[key],state[key+"_compensation"])
                    beta=settings["betas"][i]
                    prediction=predictions[mode,n,key]*beta+(current-unstored[n,key])
                    predictions[mode,n,key]=prediction
                    error=current-references[n][i]
                    row={"step":step,"mode":mode,"parameter":n,"moment":key,
                         "storage_only_prediction_error_energy":squared_norm(prediction-error),
                         "actual_error_energy":squared_norm(error),
                         "unchanged_effective_state_coordinates":int((current==before[n,key]).sum()),
                         "coordinate_count":p.numel(),
                         "reconstruction":reconstruction_terms(reference_previous=previous_refs[n][i],
                             reference_current=references[n][i],candidate_previous_read=before[n,key],
                             candidate_current_unstored=unstored[n,key],candidate_current_read=current,beta=beta)}
                    rows.append(row)
        source.save_new(output / f"step_{step:02d}.json",{"rows":rows[-8:]})
        print(f"real history step {step} complete",flush=True)
    source.save_new(output / "result.json",{"status":"COMPLETE","rows":rows})


if __name__ == "__main__":
    main()
