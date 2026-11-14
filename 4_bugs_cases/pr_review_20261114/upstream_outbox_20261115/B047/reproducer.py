import torch
import attorch

for loss_fn in (attorch.L1Loss(reduction="mean"), torch.nn.L1Loss(reduction="mean")):
    bias = torch.zeros((), device="cuda", requires_grad=True)
    prediction = bias.expand(8).contiguous()
    loss = loss_fn(prediction, torch.zeros_like(prediction))
    loss.backward()
    print(type(loss_fn).__module__, "loss=", loss.item(), "bias.grad=", bias.grad.item())
