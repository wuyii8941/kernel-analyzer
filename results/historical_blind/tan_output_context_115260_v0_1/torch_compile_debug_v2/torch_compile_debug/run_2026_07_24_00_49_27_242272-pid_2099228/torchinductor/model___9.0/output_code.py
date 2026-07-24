
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused_cat_maximum_mul_tan_0 = async_compile.cpp('''
#include "/tmp/torchinductor_tzh/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(const half* in_ptr0,
                       const half* in_ptr1,
                       const half* in_ptr2,
                       const half* in_ptr3,
                       const half* in_ptr4,
                       const half* in_ptr5,
                       half* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(85L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
            {
                #pragma omp simd simdlen(8) 
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                {
                    auto tmp35 = static_cast<float>(in_ptr4[static_cast<long>(0L)]);
                    auto tmp37 = static_cast<float>(in_ptr5[static_cast<long>(0L)]);
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(1);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = static_cast<float>(in_ptr0[static_cast<long>(x2 + (7L*x0))]);
                        auto tmp7 = c10::convert<float>(tmp6);
                        return tmp7;
                    }
                    ;
                    auto tmp8 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp9 = tmp0 >= tmp3;
                    auto tmp10 = static_cast<long>(12);
                    auto tmp11 = tmp0 < tmp10;
                    auto tmp12 = tmp9 & tmp11;
                    auto tmp13 = [&]
                    {
                        auto tmp14 = static_cast<float>(in_ptr1[static_cast<long>((-7L) + x2 + (7L*x1) + (77L*x0))]);
                        auto tmp15 = c10::convert<float>(tmp14);
                        return tmp15;
                    }
                    ;
                    auto tmp16 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                    auto tmp17 = tmp0 >= tmp10;
                    auto tmp18 = static_cast<long>(13);
                    auto tmp19 = tmp0 < tmp18;
                    auto tmp20 = tmp17 & tmp19;
                    auto tmp21 = [&]
                    {
                        auto tmp22 = static_cast<float>(in_ptr2[static_cast<long>(x2 + (7L*x0))]);
                        auto tmp23 = c10::convert<float>(tmp22);
                        return tmp23;
                    }
                    ;
                    auto tmp24 = tmp20 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                    auto tmp25 = tmp0 >= tmp18;
                    auto tmp26 = static_cast<long>(14);
                    auto tmp27 = tmp0 < tmp26;
                    auto tmp28 = [&]
                    {
                        auto tmp29 = static_cast<float>(in_ptr3[static_cast<long>(x2 + (7L*x0))]);
                        auto tmp30 = c10::convert<float>(tmp29);
                        return tmp30;
                    }
                    ;
                    auto tmp31 = tmp25 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                    auto tmp32 = tmp20 ? tmp24 : tmp31;
                    auto tmp33 = tmp12 ? tmp16 : tmp32;
                    auto tmp34 = tmp4 ? tmp8 : tmp33;
                    auto tmp36 = c10::convert<float>(tmp35);
                    auto tmp38 = c10::convert<float>(tmp37);
                    auto tmp39 = max_propagate_nan(tmp36, tmp38);
                    auto tmp40 = decltype(tmp34)(tmp34 * tmp39);
                    auto tmp41 = std::tan(tmp40);
                    auto tmp42 = c10::convert<half>(tmp41);
                    out_ptr0[static_cast<long>(x2 + (7L*x1) + (98L*x0))] = tmp42;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride(arg0_1, (17, 5, 1, 7), (35, 7, 7, 1))
    assert_size_stride(arg1_1, (17, 5, 1, 7), (35, 7, 7, 1))
    assert_size_stride(arg2_1, (17, 5, 11, 7), (385, 77, 7, 1))
    assert_size_stride(arg3_1, (17, 5, 1, 7), (35, 7, 7, 1))
    assert_size_stride(arg4_1, (), ())
    assert_size_stride(arg5_1, (1, ), (1, ))
    buf0 = empty((17, 5, 14, 7), device='cpu', dtype=torch.float16)
    cpp_fused_cat_maximum_mul_tan_0(c_void_p(arg3_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    del arg1_1
    del arg2_1
    del arg3_1
    del arg4_1
    del arg5_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((17, 5, 1, 7), (35, 7, 7, 1), device='cpu', dtype=torch.float16)
    arg1_1 = rand_strided((17, 5, 1, 7), (35, 7, 7, 1), device='cpu', dtype=torch.float16)
    arg2_1 = rand_strided((17, 5, 11, 7), (385, 77, 7, 1), device='cpu', dtype=torch.float16)
    arg3_1 = rand_strided((17, 5, 1, 7), (35, 7, 7, 1), device='cpu', dtype=torch.float16)
    arg4_1 = rand_strided((), (), device='cpu', dtype=torch.float16)
    arg5_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
