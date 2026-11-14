import torch
from liger_kernel.transformers.kl_div import LigerKLDIVLoss

student = torch.zeros((32, 32768), device="cuda", requires_grad=True)
teacher = torch.full(student.shape, 1 / 32768, device="cuda", dtype=torch.float16)
reference = student.detach().clone().requires_grad_()

LigerKLDIVLoss(reduction="mean")(student, teacher).backward()
torch.nn.KLDivLoss(reduction="mean")(reference, teacher).backward()

print("Liger gradient:", student.grad[0, 0].item())
print("PyTorch gradient:", reference.grad[0, 0].item())
print("Liger nonzero gradients:", student.grad.count_nonzero().item())
