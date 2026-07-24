
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/data1/tzh/forkcert/results/calibration/kernel_plumbing_v0_2/inductor_cache'
os.environ['TORCH_COMPILE_DEBUG'] = '1'

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

torch._inductor.config.deterministic = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.11.0+cu126
# torch cuda version: 12.6
# torch git version: 70d99e998b4955e0049d13a98d77ae1b14db1f45


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Tue_Aug_15_22:02:13_PDT_2023 
# Cuda compilation tools, release 12.2, V12.2.140 
# Build cuda_12.2.r12.2/compiler.33191640_0 

# GPU Hardware Info: 
# Tesla T4 : 1 

torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(arg0_1, [-1]);  arg0_1 = None
        return (sum_1,)

def load_args(reader):
    buf0 = reader.storage(None, 4112, device=device(type='cuda', index=0))
    reader.tensor(buf0, (4, 257), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)