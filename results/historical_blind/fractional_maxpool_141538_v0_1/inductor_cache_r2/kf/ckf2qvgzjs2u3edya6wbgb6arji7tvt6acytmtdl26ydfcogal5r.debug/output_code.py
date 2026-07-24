# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /data1/tzh/forkcert/results/historical_blind/fractional_maxpool_141538_v0_1/inductor_cache_r2/le/clekt6nnljtir5jvdpxdnofc4ax3jye5k5azumzysbl2q7stutes.py
# Topologically Sorted Source Nodes: [fractional_max_pool2d], Original ATen: [aten.fractional_max_pool2d]
# Source node to ATen node mapping:
#   fractional_max_pool2d => getitem
# Graph fragment:
#   %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%fractional_max_pool2d, 0), kwargs = {})
triton_poi_fused_fractional_max_pool2d_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=75, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1024, multi_processor_count=40), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fractional_max_pool2d_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4421EC101B78291BCBFDB601FA26265B53D87A7FE51EA870127F2A92FE9F0748', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 5)
    x0 = xindex % 5
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp2 = x1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 + tmp1
    tmp5 = 2.25
    tmp6 = tmp4 * tmp5
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp1 * tmp5
    tmp9 = libdevice.floor(tmp8)
    tmp10 = tmp7 - tmp9
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp3 < tmp12
    tmp14 = tl.full([1], 9, tl.int64)
    tmp15 = tl.where(tmp13, tmp11, tmp14)
    tmp16 = tl.full([XBLOCK], 10, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 10)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 10")
    tmp23 = x0
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 + tmp22
    tmp26 = tmp25 * tmp5
    tmp27 = libdevice.floor(tmp26)
    tmp28 = tmp22 * tmp5
    tmp29 = libdevice.floor(tmp28)
    tmp30 = tmp27 - tmp29
    tmp31 = tmp30.to(tl.int64)
    tmp32 = tmp24 < tmp12
    tmp33 = tl.where(tmp32, tmp31, tmp14)
    tmp34 = tmp33 + tmp16
    tmp35 = tmp33 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp33)
    tl.device_assert(((0 <= tmp36) & (tmp36 < 10)) | ~(xmask), "index out of bounds: 0 <= tmp36 < 10")
    tmp38 = tl.load(in_ptr1 + (tmp36 + (10*tmp19)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp38, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 2), (2, 2, 1))
    assert_size_stride(arg1_1, (1, 1, 10, 10), (100, 100, 10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 1, 5, 5), (25, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [fractional_max_pool2d], Original ATen: [aten.fractional_max_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_fractional_max_pool2d_0.run(arg0_1, arg1_1, buf0, 25, grid=grid(25), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 2), (2, 2, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 10, 10), (100, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
