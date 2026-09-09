#!/usr/bin/env python3
"""Reconstruct dW differences from actual chunk products at one saved state.

The telescoping identity proves the source of this observed residual. It does
not prove a distributional mean or persistence without additional assumptions.
"""
import importlib.util
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
import numpy as np
import torch
from scripts import run_liger_language_pilot as pilot
from scripts.run_liger_language_confirmation import OUT as TRAINING
from scripts.run_liger_single_boundary_collapse import file_sha256
from scripts.run_training_numerical_v2 import BASE, save_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=BASE / "language_accumulation_identity")
    out = parser.parse_args().output
    out.mkdir(exist_ok=False)
    source = TRAINING / "pair0"
    protocol = json.loads((source / "protocol.json").read_text())
    checkpoint = source / "candidate/final.pt"
    status = json.loads((source / "candidate/status.json").read_text())
    if file_sha256(checkpoint) != status["final_checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed")
    for name, digest in protocol["liger_sources"].items():
        if file_sha256(pilot.PACKAGE / name) != digest:
            raise RuntimeError("Liger source changed")
    save_new(out / "plan.json", {"data_use": "POST_CONFIRMATION_SOURCE_IDENTITY_CHECK",
        "selection": "pair0 candidate final state, first declared checkpoint probe batch (index 1024)",
        "checkpoint_sha256": status["final_checkpoint_sha256"], "source_sha256": file_sha256(Path(__file__)),
        "identity": "s_j=round_BF16(s_(j-1)+p_j), f_j=round_FP32(f_(j-1)+p_j); s_n-cast_BF16(f_n)=sum(e_BF16-e_FP32)-final_reference_cast_error",
        "acceptance": "Both reconstructed gradients equal the corresponding real Liger gradients; same products and input gradients; exact residual reconstruction in FP64"})
    spec = importlib.util.spec_from_file_location("liger_kernel", pilot.PACKAGE / "__init__.py", submodule_search_locations=[str(pilot.PACKAGE)])
    module = importlib.util.module_from_spec(spec); sys.modules["liger_kernel"] = module; spec.loader.exec_module(module)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    model, target = pilot.build_model(protocol, device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    pilot.materialize(model, {n:t.to(device) for n,t in state["master"].items()})
    tokens = np.load(source / "train.npy")
    if file_sha256(source / "train.npy") != protocol["data"]["train"]["encoded_sha256"]:
        raise RuntimeError("Token data changed")
    x,y,offsets = pilot.batch_from_stream(tokens,batch_size=2,sequence_length=128,stream=0,step=1024,seed=20260906,repeat_within_batch=False,device=device)
    with torch.no_grad():
        hidden = model.transformer(input_ids=x,use_cache=False).last_hidden_state.reshape(-1,256).detach()
    weight = model.lm_head.weight.detach().clone().requires_grad_(True)
    hidden.requires_grad_(True)
    modules = pilot.make_loss_modules(protocol,device)
    sums = [torch.zeros_like(weight), torch.zeros_like(weight,dtype=torch.float32)]
    error = torch.zeros_like(weight,dtype=torch.float64)
    products = []
    original_mm = torch.mm
    def observe(a,b,*args,**kwargs):
        product = original_mm(a,b,*args,**kwargs)
        if product.shape != weight.shape:
            raise RuntimeError("Unexpected torch.mm inside isolated Liger loss")
        p = product.float()
        with torch.no_grad():
            previous = [s.double() for s in sums]
            sums[0].add_(p); sums[1].add_(p)
            error.add_((sums[0].double()-previous[0]-p.double())-(sums[1].double()-previous[1]-p.double()))
        products.append(product.detach().cpu())
        return product
    try:
        torch.mm = observe
        modules["CANDIDATE"](weight,hidden,y.reshape(-1)).backward()
    finally:
        torch.mm = original_mm
    candidate_grad = weight.grad.detach().clone(); candidate_input = hidden.grad.detach().clone()
    weight.grad = hidden.grad = None
    product_index = 0
    def verify_products(a,b,*args,**kwargs):
        nonlocal product_index
        product = original_mm(a,b,*args,**kwargs)
        if product_index >= len(products) or not torch.equal(product.detach().cpu(),products[product_index]):
            raise RuntimeError("Reference chunk products differ")
        product_index += 1
        return product
    try:
        torch.mm = verify_products
        modules["REPAIR"](weight,hidden,y.reshape(-1)).backward()
    finally:
        torch.mm = original_mm
    reference_grad = weight.grad.detach()
    reference_cast = sums[1].to(weight.dtype)
    reconstructed = error-(reference_cast.double()-sums[1].double())
    actual = candidate_grad.double()-reference_grad.double()
    checks = {"candidate_gradient_exact": torch.equal(candidate_grad,sums[0]),
              "reference_gradient_exact": torch.equal(reference_grad,reference_cast),
              "same_input_gradient": torch.equal(candidate_input,hidden.grad),
              "same_chunk_products": product_index == len(products) and product_index > 0,
              "telescoping_residual_exact": torch.equal(reconstructed,actual)}
    save_new(out / "result.json", {"status": "IDENTITY_VERIFIED" if all(checks.values()) else "IDENTITY_NOT_VERIFIED",
        "checks": checks,"chunks": len(products),"offsets": offsets,
        "residual_energy": float(actual.square().sum()),
        "reconstruction_max_abs_error": float((reconstructed-actual).abs().max()),
        "scope": "One actual Liger dW source at the saved language-model state; isolated head gradient, not the total tied-embedding gradient. A deterministic identity does not establish the sign of an expectation or temporal persistence."})
    if not all(checks.values()): raise SystemExit("Source identity failed")


if __name__ == "__main__": main()
