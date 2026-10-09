"""B019: torch.optim.swa_utils.AveragedModel EMA of a bfloat16 model stagnates; an fp32 averaged copy is rejected.

AveragedModel deep-copies the model, so for a bf16 model the EMA buffer is bf16; get_ema_multi_avg_fn updates it with
_foreach_lerp_(ema, current, 1 - decay).  With decay 0.999 the increment (1 - decay)(p - ema) is below half a bf16 ulp
of ema for almost every element and is rounded away.  An fp32 copy of the model cannot be used instead:
update_parameters groups tensors by dtype and raises for an fp32 averaged copy of a bf16 model (multi_avg_fn path).

    python bugs/repro/B019_repro_averagedmodel_bf16_ema.py      # CPU or CUDA, any recent torch
"""
import copy

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, dev)
model = torch.nn.Linear(256, 256).to(dev).to(torch.bfloat16)
target = copy.deepcopy(model).float()
with torch.no_grad():
    for p in target.parameters():
        p.add_(0.01 * torch.randn_like(p))  # the trained model drifts away from its initial weights

ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
ema.update_parameters(model)  # first call copies
ref = [p.detach().float().clone() for p in model.parameters()]
with torch.no_grad():
    for p, t in zip(model.parameters(), target.parameters()):
        p.copy_(t)
for _ in range(1000):
    ema.update_parameters(model)
    with torch.no_grad():
        for r, p in zip(ref, model.parameters()):
            r.lerp_(p.float(), 1 - 0.999)

w_ema = next(ema.parameters()).float()
w_ref = ref[0]
dist_ema = (w_ema - target.weight.float()).norm() / target.weight.float().norm()
dist_ref = (w_ref - target.weight.float()).norm() / target.weight.float().norm()
print(f"EMA dtype: {next(ema.parameters()).dtype}")
print(f"after 1000 updates toward the new weights (expected fraction moved 1 - 0.999^1000 = {1 - 0.999 ** 1000:.3f}):")
print(f"  relative distance to the current weights: bf16 AveragedModel {dist_ema:.3f}, fp32 EMA {dist_ref:.3f}")

fp32_copy = AveragedModel(copy.deepcopy(model).float(), multi_avg_fn=get_ema_multi_avg_fn(0.999))
try:
    fp32_copy.update_parameters(model)
    fp32_copy.update_parameters(model)
    print("fp32 averaged copy of a bf16 model: accepted")
except RuntimeError as exc:
    print("fp32 averaged copy of a bf16 model: rejected:", str(exc).splitlines()[0][:120])

ok = AveragedModel(copy.deepcopy(model).float(), avg_fn=lambda e, p, n: e.lerp(p.float(), 1 - 0.999))
ok.update_parameters(model)
ok.update_parameters(model)
print("workaround (fp32 copy + per-parameter avg_fn that lerps in fp32): accepted")
