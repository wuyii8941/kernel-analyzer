
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.2.0+cu121
# torch cuda version: 12.1
# torch git version: 8ac9b20d4b090c213799e81acf48a55ea8d437d6


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Tue_Aug_15_22:02:13_PDT_2023 
# Cuda compilation tools, release 12.2, V12.2.140 
# Build cuda_12.2.r12.2/compiler.33191640_0 

# GPU Hardware Info: 
# Tesla T4 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        mul = torch.ops.aten.mul.Tensor(arg0_1, arg1_1)
        mul_1 = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        sub = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        return (exp,)
        
def load_args(reader):
    buf0 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf0, (), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf1, (), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
