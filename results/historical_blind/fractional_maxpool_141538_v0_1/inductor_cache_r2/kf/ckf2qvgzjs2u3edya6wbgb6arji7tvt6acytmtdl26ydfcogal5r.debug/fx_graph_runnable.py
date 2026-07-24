
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
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.1+cu121
# torch cuda version: 12.1
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


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
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        fractional_max_pool2d = torch.ops.aten.fractional_max_pool2d.default(arg1_1, [1, 1], [5, 5], arg0_1);  arg1_1 = arg0_1 = None
        getitem = fractional_max_pool2d[0];  fractional_max_pool2d = None
        return (getitem,)
        
def load_args(reader):
    buf0 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 1, 2), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 400, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1, 1, 10, 10), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)