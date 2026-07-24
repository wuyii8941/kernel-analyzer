
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
# Tesla T4 : 14 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        cat = torch.ops.aten.cat.default([arg3_1, arg2_1, arg1_1, arg0_1], 2);  arg3_1 = arg2_1 = arg1_1 = arg0_1 = None
        maximum = torch.ops.aten.maximum.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        mul = torch.ops.aten.mul.Tensor(cat, maximum);  cat = maximum = None
        tan = torch.ops.aten.tan.default(mul);  mul = None
        return (tan,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1190, dtype_hint=torch.float16)
    reader.tensor(buf0, (17, 5, 1, 7), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1190, dtype_hint=torch.float16)
    reader.tensor(buf1, (17, 5, 1, 7), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 13090, dtype_hint=torch.float16)
    reader.tensor(buf2, (17, 5, 11, 7), dtype=torch.float16, is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1190, dtype_hint=torch.float16)
    reader.tensor(buf3, (17, 5, 1, 7), dtype=torch.float16, is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 2, dtype_hint=torch.float16)
    reader.tensor(buf4, (), dtype=torch.float16, is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 2, dtype_hint=torch.float16)
    reader.tensor(buf5, (1,), dtype=torch.float16, is_leaf=True)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
