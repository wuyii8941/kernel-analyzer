# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_tzh/6o/c6oulgitih7sq5mhkv5mrvnexew7xicwljoxu34w3jmyqlguufpw.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_6, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_1
#   hidden_states => convert_element_type_3
#   hidden_states_1 => mul_3
#   hidden_states_2 => mul_4
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_6 => convert_element_type_4
#   variance => mean
# Graph fragment:
#   %primals_1 : Tensor "i64[1, 128][128, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "bf16[128256, 3072][3072, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %embedding : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=embedding]
#   %buf5 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=buf5]
#   %primals_4 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_4]
#   %rsqrt : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %embedding : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_3, 2), kwargs = {})
#   %mean : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_3 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %rsqrt), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, %convert_element_type_4), kwargs = {})
#   return %embedding,%buf5,%rsqrt,%mul_4
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 2048, 'r0_': 3151872}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tl.device_assert(((0 <= tmp0) & (tmp0 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 128256")
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tl.store(out_ptr0 + (r0_1 + 3072*x0), tmp2, r0_mask & xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tl.full([1, 1], 3072.0, tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 * tmp12
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp13 * tmp17
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/hc/chcx3323mnomtallxj7zcooqf7wvpgardf5tiwl52szuumrquznu.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub]
# Source node to ATen node mapping:
#   cache_position => iota
#   first_dummy_value => sub
#   getitem => slice_1
#   position_ids => unsqueeze
# Graph fragment:
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 1][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%unsqueeze, 1, 0, 1), kwargs = {})
#   %sub : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, 1), kwargs = {})
#   return %sub
triton_poi_fused_arange_slice_sub_unsqueeze_1 = async_compile.triton('triton_poi_fused_arange_slice_sub_unsqueeze_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=(1,))]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_slice_sub_unsqueeze_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.full([1], -1, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/h4/ch43o6r4xe5etquohdvx6jpldk2zgfvfvt3pe7sjfsl7n3ixixtk.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat]
# Source node to ATen node mapping:
#   cache_position => iota
#   first_dummy_value => sub
#   getitem => slice_1
#   position_diff => cat
#   position_ids => unsqueeze
# Graph fragment:
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 1][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%unsqueeze, 1, 0, 1), kwargs = {})
#   %sub : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, 1), kwargs = {})
#   %cat : Tensor "i64[1, 129][129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sub, %unsqueeze], -1), kwargs = {})
#   return %buf2
triton_poi_fused_arange_cat_slice_sub_unsqueeze_2 = async_compile.triton('triton_poi_fused_arange_cat_slice_sub_unsqueeze_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(1,), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_cat_slice_sub_unsqueeze_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 2048}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_cat_slice_sub_unsqueeze_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/7h/c7h526mo7k5anzfnhnxr2t3v6ywc7auacflis3huigyfqu3pu5w5.py
# Topologically Sorted Source Nodes: [position_diff, ne, packed_sequence_mask], Original ATen: [aten.slice, aten.sub, aten.ne, aten.cumsum]
# Source node to ATen node mapping:
#   ne => ne
#   packed_sequence_mask => cumsum
#   position_diff => slice_2, slice_3, sub_1
# Graph fragment:
#   %cat : Tensor "i64[1, 129][129, 1]cuda:0" = PlaceHolder[target=cat]
#   %slice_2 : Tensor "i64[1, 128][129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 0, 128), kwargs = {})
#   %slice_3 : Tensor "i64[1, 128][129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 1, 129), kwargs = {})
#   %sub_1 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_3, %slice_2), kwargs = {})
#   %ne : Tensor "b8[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%sub_1, 1), kwargs = {})
#   %cumsum : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%ne, -1), kwargs = {})
#   return %cumsum
triton_per_fused_cumsum_ne_slice_sub_3 = async_compile.triton('triton_per_fused_cumsum_ne_slice_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=(2,))]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_cumsum_ne_slice_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'r0_': 4096}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused_cumsum_ne_slice_sub_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1 + r0_0), None)
    tmp1 = tl.load(in_ptr0 + (r0_0), None)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp8, = tl.associative_scan((tmp7,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp8, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/tg/ctg24u2iwglqnyqfixcibuut2avcyooq5h2rprvv37rxgacwnbuc.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear, view, query_states, cos_3, sin_3, mul_4, x1, x2, neg, cat_1, mul_5, q_embed, matmul_1, permute_1360], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
# Source node to ATen node mapping:
#   cache_position => iota
#   cat_1 => cat_1
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_10
#   expand => expand_1
#   freqs => permute
#   getitem_1 => unsqueeze_1, unsqueeze_2
#   getitem_2 => unsqueeze_3
#   linear => view_12
#   matmul => mul
#   matmul_1 => view_22
#   mul_4 => mul_5
#   mul_5 => mul_6
#   neg => neg
#   permute_1360 => permute_1360
#   position_ids => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_embed => add_2
#   query_states => permute_2
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_6
#   view => view_13
#   x1 => slice_4
#   x2 => slice_5
# Graph fragment:
#   %mm : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %expand_7 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0" = PlaceHolder[target=expand_7]
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_2, [1, -1, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_3, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 128, 64][8192, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 128, 1, 64][8192, 1, 8192, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 128, 2, 64][8192, 1, 0, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 128, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 128, 2, 64][16384, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_10 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 128, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_10,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %sin : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_10,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %view_12 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 128, 3072]), kwargs = {})
#   %view_13 : Tensor "bf16[1, 128, 24, 128][393216, 3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_12, [1, 128, -1, 128]), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 24, 128, 128][393216, 128, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 24, 128, 128][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_5), kwargs = {})
#   %slice_4 : Tensor "bf16[1, 24, 128, 64][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 64), kwargs = {})
#   %slice_5 : Tensor "bf16[1, 24, 128, 64][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 64, 9223372036854775807), kwargs = {})
#   %neg : Tensor "bf16[1, 24, 128, 64][196608, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %cat_1 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_6), kwargs = {})
#   %add_2 : Tensor "bf16[1, 24, 128, 128][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %view_22 : Tensor "bf16[24, 128, 128][128, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_7, [24, 128, 128]), kwargs = {})
#   %permute_1360 : Tensor "bf16[24, 128, 128][128, 1, 3072]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1]), kwargs = {})
#   return %expand_7,%permute_1360
triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 5505280}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    x2 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 3072*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + ((x4 % 64)), None, eviction_policy='evict_last')
    tmp2 = x1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl_math.cos(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 * tmp8
    tmp10 = x0
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 64, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tl.load(in_ptr0 + (64 + 128*x2 + 3072*x1 + (x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tmp10 >= tmp13
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp10 < tmp20
    tmp22 = tl.load(in_ptr0 + (128*x2 + 3072*x1 + ((-64) + x0)), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp14, tmp18, tmp22)
    tmp24 = tl_math.sin(tmp4)
    tmp25 = tmp24 * tmp6
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp23 * tmp26
    tmp28 = tmp9 + tmp27
    tl.store(out_ptr0 + (x4), tmp28, None)
    tl.store(out_ptr1 + (x0 + 128*x2 + 3072*x1), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/6j/c6jzpmrcyemrm6uy6tjvkojtxjnztrsm7wcrggmxjruhxhgg3yrs.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear_1, view_1, key_states, cos_3, sin_3, mul_6, x1_1, x2_1, neg_1, cat_2, mul_7, k_embed, getitem_7, hidden_states_3, key_states_1], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
# Source node to ATen node mapping:
#   cache_position => iota
#   cat_2 => cat_2
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_10
#   expand => expand_1
#   freqs => permute
#   getitem_1 => unsqueeze_1, unsqueeze_2
#   getitem_2 => unsqueeze_3
#   getitem_7 => unsqueeze_7
#   hidden_states_3 => expand_5
#   k_embed => add_3
#   key_states => permute_4
#   key_states_1 => clone_2
#   linear_1 => view_15
#   matmul => mul
#   mul_6 => mul_7
#   mul_7 => mul_8
#   neg_1 => neg_1
#   position_ids => unsqueeze
#   position_ids_expanded => convert_element_type
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_6
#   view_1 => view_16
#   x1_1 => slice_6
#   x2_1 => slice_7
# Graph fragment:
#   %mm_1 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_2, [1, -1, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_3, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 128, 64][8192, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 128, 1, 64][8192, 1, 8192, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 128, 2, 64][8192, 1, 0, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 128, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 128, 2, 64][16384, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_10 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 128, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_10,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %sin : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_10,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %view_15 : Tensor "bf16[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 128, 1024]), kwargs = {})
#   %view_16 : Tensor "bf16[1, 128, 8, 128][131072, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_15, [1, 128, -1, 128]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 8, 128, 128][131072, 128, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_16, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_7 : Tensor "bf16[1, 8, 128, 128][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_5), kwargs = {})
#   %slice_6 : Tensor "bf16[1, 8, 128, 64][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 0, 64), kwargs = {})
#   %slice_7 : Tensor "bf16[1, 8, 128, 64][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 64, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "bf16[1, 8, 128, 64][65536, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_7,), kwargs = {})
#   %cat_2 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_6], -1), kwargs = {})
#   %mul_8 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_2, %unsqueeze_6), kwargs = {})
#   %add_3 : Tensor "bf16[1, 8, 128, 128][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 8, 1, 128, 128][131072, 128, 131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_3, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 3, 128, 128][131072, 128, 0, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_7, [1, 8, 3, 128, 128]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 3, 128, 128][393216, 49152, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 2359552}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    x3 = xindex // 49152
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3 + 1024*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + ((x5 % 64)), None, eviction_policy='evict_last')
    tmp2 = x1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl_math.cos(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 * tmp8
    tmp10 = x0
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 64, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tl.load(in_ptr0 + (64 + 128*x3 + 1024*x1 + (x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tmp10 >= tmp13
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp10 < tmp20
    tmp22 = tl.load(in_ptr0 + (128*x3 + 1024*x1 + ((-64) + x0)), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp14, tmp18, tmp22)
    tmp24 = tl_math.sin(tmp4)
    tmp25 = tmp24 * tmp6
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp23 * tmp26
    tmp28 = tmp9 + tmp27
    tl.store(out_ptr0 + (x5), tmp28, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/oj/cojzxf4wwuwq5jocpxzxas6jveu5iyr55o45juij4f6ef7e6afec.py
# Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_1, attn_weights, attn_weights_1, softmax, attn_weights_2], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
# Source node to ATen node mapping:
#   attn_weights => mul_9
#   attn_weights_1 => add_4
#   attn_weights_2 => convert_element_type_14
#   batch_arange => iota_2
#   batched_outputs_2 => view_6
#   cache_position => iota
#   eq => eq, view_4, view_5
#   index => index, view_2
#   index_1 => index_1
#   kv_arange_1 => add
#   le => le, view
#   mask => full_default_2, where
#   matmul_1 => view_24
#   result_1 => bitwise_and, full_default
#   result_2 => bitwise_and_1
#   softmax => convert_element_type_13, div, exp_default_28, sub_tensor_28
#   tensor => full_default_1
# Graph fragment:
#   %bmm : Tensor "bf16[24, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm]
#   %cumsum : Tensor "i64[1, 128][128, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %add_4 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0" = PlaceHolder[target=add_4]
#   %getitem_56 : Tensor "f32[1, 24, 128, 1][3072, 128, 1, 1]cuda:0" = PlaceHolder[target=getitem_56]
#   %getitem_57 : Tensor "f32[1, 24, 128, 1][3072, 128, 1, 1]cuda:0" = PlaceHolder[target=getitem_57]
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %iota_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [128, 1]), kwargs = {})
#   %le : Tensor "b8[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add, %view), kwargs = {})
#   %full_default : Tensor "b8[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and : Tensor "b8[128, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %view_2 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_2, %iota]), kwargs = {})
#   %index_1 : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_2, %add]), kwargs = {})
#   %view_4 : Tensor "i64[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [1, 128, 1]), kwargs = {})
#   %view_5 : Tensor "i64[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [1, 1, 128]), kwargs = {})
#   %eq : Tensor "b8[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_4, %view_5), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %eq), kwargs = {})
#   %view_6 : Tensor "b8[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and_1, [1, 1, 128, 128]), kwargs = {})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.3895313892515355e+38), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default_2), kwargs = {})
#   %view_24 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 24, 128, 128]), kwargs = {})
#   %mul_9 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_24, 0.08838834764831845), kwargs = {})
#   %add_4 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %where), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_28 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_13, -1), kwargs = {})
#   %sub_tensor_28 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_13, %getitem_56), kwargs = {})
#   %exp_default_28 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_28,), kwargs = {})
#   %div : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_28, %getitem_57), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   return %add_4,%getitem_56,%getitem_57,%expand_9
triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6 = async_compile.triton('triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 4, 'num_reduction': 4, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 50176, 'r0_': 3933184}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3072
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 128*x3), xmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (r0_2), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 0.08838834764831845, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = r0_2
    tmp4 = x0
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1, 1], True, tl.int1)
    tmp7 = tmp6 & tmp5
    tmp10 = tmp8 == tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.full([1, 1], 0.0, tl.float32)
    tmp13 = tl.full([1, 1], -3.3895313892515355e+38, tl.float32)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp2 + tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, float("-inf"))
    tmp22 = triton_helpers.max2(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tmp17 - tmp22
    tmp24 = libdevice.exp(tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tmp16 - tmp22
    tmp30 = libdevice.exp(tmp29)
    tmp31 = (tmp30 / tmp28)
    tmp32 = tmp31.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 128*x3), tmp15, xmask)
    tl.store(out_ptr2 + (r0_2 + 128*x3), tmp32, xmask)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/rs/crseg6x6l3bmscqwjvbzhygdmbnrhtt42idlzgsdo5c5bgs2qpmx.py
# Topologically Sorted Source Nodes: [linear_2, view_2, value_states, getitem_8, hidden_states_4, value_states_1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   getitem_8 => unsqueeze_8
#   hidden_states_4 => expand_6
#   linear_2 => view_18
#   value_states => permute_6
#   value_states_1 => clone_3
#   view_2 => view_19
# Graph fragment:
#   %mm_2 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_18 : Tensor "bf16[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 128, 1024]), kwargs = {})
#   %view_19 : Tensor "bf16[1, 128, 8, 128][131072, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_18, [1, 128, -1, 128]), kwargs = {})
#   %permute_6 : Tensor "bf16[1, 8, 128, 128][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_8 : Tensor "bf16[1, 8, 1, 128, 128][131072, 128, 131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_6, 2), kwargs = {})
#   %expand_6 : Tensor "bf16[1, 8, 3, 128, 128][131072, 128, 0, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, 8, 3, 128, 128]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 3, 128, 128][393216, 49152, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7 = async_compile.triton('triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 1835008}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    x3 = xindex // 49152
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3 + 1024*x1), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/2v/c2vu7lwfruj3vkvyfbxd7byxixj73stfzwxbg5ictjo5qdfbysy4.py
# Topologically Sorted Source Nodes: [attn_output, transpose_5, attn_output_1], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   attn_output => view_27
#   attn_output_1 => clone_5
#   transpose_5 => permute_8
# Graph fragment:
#   %bmm_1 : Tensor "bf16[24, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_27 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 24, 128, 128]), kwargs = {})
#   %permute_8 : Tensor "bf16[1, 128, 24, 128][393216, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_27, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "bf16[1, 128, 24, 128][393216, 3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_clone_transpose_view_8 = async_compile.triton('triton_poi_fused_clone_transpose_view_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_transpose_view_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 2359296}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 24)
    x2 = xindex // 3072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 16384*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/ub/cubpbnvvik6lf5i7g2sfu7shrdsauz3qs53y7ncteelbfmltfnxc.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_5, rsqrt_1, hidden_states_7, to_9, hidden_states_8], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_5 => add_6
#   attn_output_3 => view_30
#   hidden_states_5 => add_5
#   hidden_states_6 => convert_element_type_19
#   hidden_states_7 => mul_10
#   hidden_states_8 => mul_11
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to_9 => convert_element_type_20
#   variance_1 => mean_1
# Graph fragment:
#   %embedding : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=embedding]
#   %mm_3 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %buf22 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=buf22]
#   %primals_9 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_9]
#   %rsqrt_1 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %view_30 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 128, 3072]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_30), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_19, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_6 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_10 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %rsqrt_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   %mul_11 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, %convert_element_type_20), kwargs = {})
#   return %buf22,%rsqrt_1,%mul_11
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 1024, 'r0_': 3151872}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tl.full([1, 1], 3072.0, tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.store(in_out_ptr0 + (x0), tmp12, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 * tmp12
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp13 * tmp19
        tl.store(out_ptr0 + (r0_1 + 3072*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/7r/c7rdpd4zkp3mmex2rhhpwxxr5b5g2kpksiy5omjhmsrqnxuiq7t4.py
# Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   linear_4 => view_32
#   linear_5 => view_34
#   mul_11 => mul_12
#   silu => add_7, convert_element_type_23, convert_element_type_24, div_1, exp_1, neg_2
# Graph fragment:
#   %mm_4 : Tensor "bf16[128, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %mm_5 : Tensor "bf16[128, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %view_32 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [1, 128, 8192]), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_32, torch.float32), kwargs = {})
#   %neg_2 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_23,), kwargs = {})
#   %exp_1 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_7 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_23, %add_7), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   %view_34 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [1, 128, 8192]), kwargs = {})
#   %mul_12 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %view_34), kwargs = {})
#   return %mul_12
triton_poi_fused__unsafe_view_mul_silu_10 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 8388608}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/xt/cxtketxmek5rjakbhi5pqbemmrvbmxmpq6kx6khrirxxw3zh2x3u.py
# Topologically Sorted Source Nodes: [attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, to_11, hidden_states_12], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_7 => add_9
#   attn_output_3 => view_30
#   down_proj => view_36
#   hidden_states_10 => convert_element_type_29
#   hidden_states_11 => mul_13
#   hidden_states_12 => mul_14
#   hidden_states_5 => add_5
#   hidden_states_9 => add_8
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   to_11 => convert_element_type_30
#   variance_2 => mean_2
# Graph fragment:
#   %embedding : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=embedding]
#   %mm_3 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_6 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %add_8 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_8]
#   %buf30 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=buf30]
#   %primals_13 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_13]
#   %rsqrt_2 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_2]
#   %view_30 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 128, 3072]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_30), kwargs = {})
#   %view_36 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 128, 3072]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_36), kwargs = {})
#   %convert_element_type_29 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_29, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_9 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_13 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_29, %rsqrt_2), kwargs = {})
#   %convert_element_type_30 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   %mul_14 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_13, %convert_element_type_30), kwargs = {})
#   return %add_8,%buf30,%rsqrt_2,%mul_14
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 1024, 'r0_': 5511168}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp4, r0_mask & xmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = tl.full([1, 1], 3072.0, tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tl.full([1, 1], 1e-05, tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tl.store(in_out_ptr1 + (x0), tmp14, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 * tmp14
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp15 * tmp19
        tl.store(out_ptr0 + (r0_1 + 3072*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/bn/cbnh73v2dqfs3xdic3wm4vaxiabyuqkfq6l3jzhgss3hzhy35vdk.py
# Topologically Sorted Source Nodes: [attn_output_7, hidden_states_15, hidden_states_16, pow_4, variance_3, add_12, rsqrt_3, hidden_states_17, to_14, hidden_states_18], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_12 => add_14
#   attn_output_7 => view_56
#   hidden_states_15 => add_13
#   hidden_states_16 => convert_element_type_45
#   hidden_states_17 => mul_20
#   hidden_states_18 => mul_21
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   to_14 => convert_element_type_46
#   variance_3 => mean_3
# Graph fragment:
#   %add_8 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_8]
#   %mm_10 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %add_13 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_13]
#   %buf48 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=buf48]
#   %primals_18 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_18]
#   %rsqrt_3 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_3]
#   %view_56 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [1, 128, 3072]), kwargs = {})
#   %add_13 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_56), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_45, 2), kwargs = {})
#   %mean_3 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_14 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_20 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_45, %rsqrt_3), kwargs = {})
#   %convert_element_type_46 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_20, torch.bfloat16), kwargs = {})
#   %mul_21 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, %convert_element_type_46), kwargs = {})
#   return %add_13,%buf48,%rsqrt_3,%mul_21
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 1024, 'r0_': 4724736}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp2, r0_mask & xmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp8 = tl.full([1, 1], 3072.0, tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tl.store(in_out_ptr1 + (x0), tmp12, xmask)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 * tmp12
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp13 * tmp17
        tl.store(out_ptr0 + (r0_1 + 3072*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/ad/cadalypovobnydrxcqgatkpbej3chsl7q2q65gf5tndelrqub6v3.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_182, view_78, query_states_26, linear_183, view_79, key_states_52, mul_264, x1_52, x2_52, neg_52, cat_53, mul_265, q_embed_26, mul_266, x1_53, x2_53, neg_53, cat_54, mul_267, k_embed_26, getitem_189, hidden_states_263, key_states_53, matmul_53, permute_398], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
# Source node to ATen node mapping:
#   cache_position => iota
#   cat_53 => cat_53
#   cat_54 => cat_54
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_10
#   expand => expand_1
#   freqs => permute
#   getitem_1 => unsqueeze_1, unsqueeze_2
#   getitem_189 => unsqueeze_111
#   getitem_2 => unsqueeze_3
#   hidden_states_263 => expand_161
#   k_embed_26 => add_211
#   key_states_52 => permute_316
#   key_states_53 => clone_106
#   linear_182 => view_688
#   linear_183 => view_691
#   matmul => mul
#   matmul_53 => view_698
#   mul_264 => mul_265
#   mul_265 => mul_266
#   mul_266 => mul_267
#   mul_267 => mul_268
#   neg_52 => neg_78
#   neg_53 => neg_79
#   permute_398 => permute_398
#   position_ids => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_embed_26 => add_210
#   query_states_26 => permute_314
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_6
#   view_78 => view_689
#   view_79 => view_692
#   x1_52 => slice_108
#   x1_53 => slice_110
#   x2_52 => slice_109
#   x2_53 => slice_111
# Graph fragment:
#   %mm_182 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_182]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %expand_163 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0" = PlaceHolder[target=expand_163]
#   %mm_183 : Tensor "bf16[128, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_183]
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_2, [1, -1, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_3, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 128][8192, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 128, 64][8192, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 128, 1, 64][8192, 1, 8192, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 128, 2, 64][8192, 1, 0, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 128, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 128, 2, 64][16384, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_10 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 128, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_10,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %sin : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_10,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %view_688 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_182, [1, 128, 3072]), kwargs = {})
#   %view_689 : Tensor "bf16[1, 128, 24, 128][393216, 3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_688, [1, 128, -1, 128]), kwargs = {})
#   %permute_314 : Tensor "bf16[1, 24, 128, 128][393216, 128, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_689, [0, 2, 1, 3]), kwargs = {})
#   %view_691 : Tensor "bf16[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_183, [1, 128, 1024]), kwargs = {})
#   %view_692 : Tensor "bf16[1, 128, 8, 128][131072, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_691, [1, 128, -1, 128]), kwargs = {})
#   %permute_316 : Tensor "bf16[1, 8, 128, 128][131072, 128, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_692, [0, 2, 1, 3]), kwargs = {})
#   %mul_265 : Tensor "bf16[1, 24, 128, 128][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_314, %unsqueeze_5), kwargs = {})
#   %slice_108 : Tensor "bf16[1, 24, 128, 64][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_314, 3, 0, 64), kwargs = {})
#   %slice_109 : Tensor "bf16[1, 24, 128, 64][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_314, 3, 64, 9223372036854775807), kwargs = {})
#   %neg_78 : Tensor "bf16[1, 24, 128, 64][196608, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_109,), kwargs = {})
#   %cat_53 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_78, %slice_108], -1), kwargs = {})
#   %mul_266 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_53, %unsqueeze_6), kwargs = {})
#   %add_210 : Tensor "bf16[1, 24, 128, 128][393216, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_265, %mul_266), kwargs = {})
#   %mul_267 : Tensor "bf16[1, 8, 128, 128][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_316, %unsqueeze_5), kwargs = {})
#   %slice_110 : Tensor "bf16[1, 8, 128, 64][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_316, 3, 0, 64), kwargs = {})
#   %slice_111 : Tensor "bf16[1, 8, 128, 64][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_316, 3, 64, 9223372036854775807), kwargs = {})
#   %neg_79 : Tensor "bf16[1, 8, 128, 64][65536, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_111,), kwargs = {})
#   %cat_54 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_79, %slice_110], -1), kwargs = {})
#   %mul_268 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_54, %unsqueeze_6), kwargs = {})
#   %add_211 : Tensor "bf16[1, 8, 128, 128][131072, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_267, %mul_268), kwargs = {})
#   %unsqueeze_111 : Tensor "bf16[1, 8, 1, 128, 128][131072, 128, 131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_211, 2), kwargs = {})
#   %expand_161 : Tensor "bf16[1, 8, 3, 128, 128][131072, 128, 0, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_111, [1, 8, 3, 128, 128]), kwargs = {})
#   %clone_106 : Tensor "bf16[1, 8, 3, 128, 128][393216, 49152, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_161,), kwargs = {memory_format: torch.contiguous_format})
#   %view_698 : Tensor "bf16[24, 128, 128][128, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%expand_163, [24, 128, 128]), kwargs = {})
#   %permute_398 : Tensor "bf16[24, 128, 128][128, 1, 3072]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_698, [0, 2, 1]), kwargs = {})
#   return %expand_163,%permute_398,%clone_106
triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_13 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 3, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 7864576}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    x2 = xindex // 16384
    x6 = xindex
    x4 = xindex // 49152
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 3072*x1), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + ((x6 % 64)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr2 + (x0 + 128*x4 + 1024*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = x1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl_math.cos(tmp4)
    tmp6 = tl.full([1], 1.0, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 * tmp8
    tmp10 = x0
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tl.full([1], 64, tl.int64)
    tmp14 = tmp10 < tmp13
    tmp15 = tl.load(in_ptr0 + (64 + 128*x2 + 3072*x1 + (x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tmp10 >= tmp13
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp10 < tmp20
    tmp22 = tl.load(in_ptr0 + (128*x2 + 3072*x1 + ((-64) + x0)), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp14, tmp18, tmp22)
    tmp24 = tl_math.sin(tmp4)
    tmp25 = tmp24 * tmp6
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp23 * tmp26
    tmp28 = tmp9 + tmp27
    tmp30 = tmp29 * tmp8
    tmp31 = tl.load(in_ptr2 + (64 + 128*x4 + 1024*x1 + (x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = -tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp14, tmp32, tmp33)
    tmp35 = tl.load(in_ptr2 + (128*x4 + 1024*x1 + ((-64) + x0)), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp14, tmp34, tmp35)
    tmp37 = tmp36 * tmp26
    tmp38 = tmp30 + tmp37
    tl.store(out_ptr0 + (x6), tmp28, None)
    tl.store(out_ptr1 + (x0 + 128*x2 + 3072*x1), tmp28, None)
    tl.store(out_ptr2 + (x6), tmp38, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/2a/c2adsqtgzl72yl6wb5gu5xmc3hgothbd2codrh6dwlyyfb4x7bgf.py
# Topologically Sorted Source Nodes: [labels], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   labels => constant_pad_nd
# Graph fragment:
#   %primals_1 : Tensor "i64[1, 128][128, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %constant_pad_nd : Tensor "i64[1, 129][129, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [0, 1], -100.0), kwargs = {})
#   return %constant_pad_nd
triton_poi_fused_constant_pad_nd_14 = async_compile.triton('triton_poi_fused_constant_pad_nd_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 3088}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 129, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.full([1], -100, tl.int64)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/xh/cxhhvaebvaeupsfivpv2cj4e2wrqi72ynf66bo4uf7via4riazef.py
# Topologically Sorted Source Nodes: [logits, logits_1, logits_2], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
# Graph fragment:
#   %mm_196 : Tensor "bf16[128, 128256][128256, 1]cuda:0" = PlaceHolder[target=mm_196]
#   %view_740 : Tensor "bf16[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 128, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %view_741 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_741, 1), kwargs = {})
#   return %buf737
triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_15 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 2048, 'r0_': 32833536}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_15(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 64128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 64128*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/u6/cu63m2nrdhh5e4rexgqsebuop3zetjxq4vjp6d2c3smrpkvepbfo.py
# Topologically Sorted Source Nodes: [logits, logits_1, logits_2], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
# Graph fragment:
#   %buf737 : Tensor "f32[128, 1, 2][2, 256, 1]cuda:0" = PlaceHolder[target=buf737]
#   %view_740 : Tensor "bf16[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 128, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %view_741 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_741, 1), kwargs = {})
#   return %buf738
triton_per_fused__to_copy__unsafe_view_prepare_softmax_online_view_16 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_prepare_softmax_online_view_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_prepare_softmax_online_view_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 1024, 'r0_': 256}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_prepare_softmax_online_view_16(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 2*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/vk/cvkht2igyf4ejea5agvpkd75kub653wrgmbnun2t3buyehagpxn3.py
# Topologically Sorted Source Nodes: [logits, logits_1, logits_2], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
# Graph fragment:
#   %mm_196 : Tensor "bf16[128, 128256][128256, 1]cuda:0" = PlaceHolder[target=mm_196]
#   %buf738 : Tensor "f32[128, 1][1, 128]cuda:0" = PlaceHolder[target=buf738]
#   %view_740 : Tensor "bf16[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 128, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %view_741 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_741, 1), kwargs = {})
#   return %buf739
triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_17 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 2048, 'r0_': 32833536}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_17(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 256
    r0_numel = 64128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x1 = xindex // 2
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 64128*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp4 = libdevice.exp(tmp3)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/nf/cnfz2br7tbsuj6or7ov52g5tmdzkshrswpep3wernqnsptyhimy3.py
# Topologically Sorted Source Nodes: [logits, logits_1, logits_2, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online, aten._log_softmax]
# Source node to ATen node mapping:
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
#   loss => log
# Graph fragment:
#   %buf739 : Tensor "f32[128, 1, 2][2, 256, 1]cuda:0" = PlaceHolder[target=buf739]
#   %getitem_1 : Tensor "f32[128, 1][1, 128]cuda:0" = PlaceHolder[target=getitem_1]
#   %view_740 : Tensor "bf16[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 128, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %view_741 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_741, 1), kwargs = {})
#   %log : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   return %getitem_1,%log
triton_per_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18 = async_compile.triton('triton_per_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 1024, 'r0_': 256}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 2*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl_math.log(tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/ho/chozazncs2czvjvrxj425nc5lgtdpkjhayynz7d3zvzyrldtjxha.py
# Topologically Sorted Source Nodes: [logits, logits_1, getitem_200, logits_2, shift_labels_1, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.slice, aten.view, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   getitem_200 => slice_116
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
#   loss => convert_element_type_736, div_56, full_default_3, full_default_4, gather, getitem, ne_1, neg_84, squeeze, sub_31, sub_tensor, sum_30, sum_31, unsqueeze_117, where_1, where_2
#   shift_labels_1 => view_742
# Graph fragment:
#   %constant_pad_nd : Tensor "i64[1, 129][129, 1]cuda:0" = PlaceHolder[target=constant_pad_nd]
#   %mm_196 : Tensor "bf16[128, 128256][128256, 1]cuda:0" = PlaceHolder[target=mm_196]
#   %buf738 : Tensor "f32[128, 1][1, 128]cuda:0" = PlaceHolder[target=buf738]
#   %log : Tensor "f32[128, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_30 : Tensor "i64[][]cuda:0" = PlaceHolder[target=sum_30]
#   %sum_31 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_31]
#   %convert_element_type_736 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_736]
#   %view_740 : Tensor "bf16[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 128, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %slice_116 : Tensor "i64[1, 128][129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, 1, 1, 9223372036854775807), kwargs = {})
#   %view_741 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %view_742 : Tensor "i64[128][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_116, [-1]), kwargs = {})
#   %getitem : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=2] = call_function[target=operator.getitem](args = (%prepare_softmax_online_default, 0), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_741, %getitem), kwargs = {})
#   %sub_31 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne_1 : Tensor "b8[128][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_742, -100), kwargs = {})
#   %full_default_3 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %view_742, %full_default_3), kwargs = {})
#   %unsqueeze_117 : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_1, 1), kwargs = {})
#   %gather : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_31, 1, %unsqueeze_117), kwargs = {})
#   %squeeze : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg_84 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg_84, %full_default_4), kwargs = {})
#   %sum_30 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%ne_1,), kwargs = {})
#   %convert_element_type_736 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_30, torch.float32), kwargs = {})
#   %sum_31 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%where_2,), kwargs = {})
#   %div_56 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sum_31, %convert_element_type_736), kwargs = {})
#   return %sum_30,%sum_31,%convert_element_type_736,%div_56
triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19 = async_compile.triton('triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(6,))]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'r0_': 2048}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (1 + r0_0), None)
    tmp16 = tl.load(in_ptr2 + (r0_0), None)
    tmp18 = tl.load(in_ptr3 + (r0_0), None)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int64)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.int64)
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tl.where(tmp2, tmp0, tmp7)
    tmp9 = tl.full([1, 1], 128256, tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert((0 <= tmp12) & (tmp12 < 128256), "index out of bounds: 0 <= tmp12 < 128256")
    tmp14 = tl.load(in_ptr1 + (tmp12 + 128256*r0_0), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 - tmp16
    tmp19 = tmp17 - tmp18
    tmp20 = -tmp19
    tmp21 = tl.full([1, 1], 0.0, tl.float32)
    tmp22 = tl.where(tmp2, tmp20, tmp21)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
    tmp25 = tl.sum(tmp23, 1)[:, None].to(tl.float32)
    tmp26 = tmp6.to(tl.float32)
    tmp27 = (tmp25 / tmp26)
    tl.store(out_ptr1 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp26, None)
    tl.store(in_out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1)), tmp27, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256 = args
        args.clear()
        assert_size_stride(primals_1, (1, 128), (128, 1), 'input')
        assert_size_stride(primals_2, (128256, 3072), (3072, 1), 'input')
        assert_size_stride(primals_4, (3072, ), (1, ), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            primals_1 = copy_if_misaligned(primals_1)
            buf0 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf6 = reinterpret_tensor(buf5, (1, 128, 1), (128, 1, 1), 0); del buf5  # reuse
            buf7 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_6, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(buf6, primals_1, primals_2, primals_4, buf0, buf7, 128, 3072, stream=raw_stream0)
            buf3 = empty_strided_cuda((1, 129), (129, 1), torch.int64)
            buf1 = reinterpret_tensor(buf3, (1, 1), (129, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_slice_sub_unsqueeze_1.run(buf1, 1, stream=raw_stream0)
            buf2 = reinterpret_tensor(buf3, (1, 128), (129, 1), 1)  # alias
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_cat_slice_sub_unsqueeze_2.run(buf2, 128, stream=raw_stream0)
            buf4 = empty_strided_cuda((1, 128), (128, 1), torch.int64)
            # Topologically Sorted Source Nodes: [position_diff, ne, packed_sequence_mask], Original ATen: [aten.slice, aten.sub, aten.ne, aten.cumsum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_cumsum_ne_slice_sub_3.run(buf3, buf4, 1, 128, stream=raw_stream0)
            del buf1
            del buf2
            assert_size_stride(primals_5, (3072, 3072), (3072, 1), 'input')
            buf8 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1, to_6, hidden_states_2, linear], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_5, (3072, 3072), (1, 3072), 0), out=buf8)
            assert_size_stride(primals_6, (1024, 3072), (3072, 1), 'input')
            buf9 = empty_strided_cuda((128, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1, to_6, hidden_states_2, linear, linear_1], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_6, (3072, 1024), (1, 3072), 0), out=buf9)
            assert_size_stride(primals_7, (1024, 3072), (3072, 1), 'input')
            buf10 = empty_strided_cuda((128, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states, hidden_states_1, to_6, hidden_states_2, linear, linear_2], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_7, (3072, 1024), (1, 3072), 0), out=buf10)
            assert_size_stride(primals_3, (64, ), (1, ), 'input')
            buf11 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf772 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear, view, query_states, cos_3, sin_3, mul_4, x1, x2, neg, cat_1, mul_5, q_embed, matmul_1, permute_1360], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf8, primals_3, buf11, buf772, 393216, stream=raw_stream0)
            buf12 = reinterpret_tensor(buf8, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear_1, view_1, key_states, cos_3, sin_3, mul_6, x1_1, x2_1, neg_1, cat_2, mul_7, k_embed, getitem_7, hidden_states_3, key_states_1], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf9, primals_3, buf12, 393216, stream=raw_stream0)
            buf13 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear, view, query_states, linear_1, view_1, key_states, cos_3, sin_3, mul_4, x1, x2, neg, cat_1, mul_5, q_embed, mul_6, x1_1, x2_1, neg_1, cat_2, mul_7, k_embed, getitem_7, hidden_states_3, key_states_1, transpose_4, matmul_1], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf11, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf12, (24, 128, 128), (16384, 1, 128), 0), out=buf13)
            buf14 = reinterpret_tensor(buf13, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf13  # reuse
            buf15 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf16 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf17 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_1, attn_weights, attn_weights_1, softmax, attn_weights_2], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf14, buf4, buf15, buf16, buf17, 3072, 128, stream=raw_stream0)
            buf18 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view_2, value_states, getitem_8, hidden_states_4, value_states_1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf10, buf18, 393216, stream=raw_stream0)
            buf19 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view_2, value_states, getitem_8, hidden_states_4, value_states_1, softmax, attn_weights_2, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf17, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf18, (24, 128, 128), (16384, 128, 1), 0), out=buf19)
            buf20 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output, transpose_5, attn_output_1], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf19, buf20, 393216, stream=raw_stream0)
            assert_size_stride(primals_8, (3072, 3072), (3072, 1), 'input')
            buf21 = reinterpret_tensor(buf19, (128, 3072), (3072, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [attn_output, transpose_5, attn_output_1, reshape_2, attn_output_3], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_8, (3072, 3072), (1, 3072), 0), out=buf21)
            assert_size_stride(primals_9, (3072, ), (1, ), 'input')
            buf22 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf23 = reinterpret_tensor(buf22, (1, 128, 1), (128, 1, 1), 0); del buf22  # reuse
            buf24 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_5, rsqrt_1, hidden_states_7, to_9, hidden_states_8], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9.run(buf23, buf0, buf21, primals_9, buf24, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_10, (8192, 3072), (3072, 1), 'input')
            buf25 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_5, hidden_states_6, hidden_states_7, to_9, hidden_states_8, linear_4], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf24, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_10, (3072, 8192), (1, 3072), 0), out=buf25)
            assert_size_stride(primals_11, (8192, 3072), (3072, 1), 'input')
            buf26 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_5, hidden_states_6, hidden_states_7, to_9, hidden_states_8, linear_4, linear_5], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf24, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_11, (3072, 8192), (1, 3072), 0), out=buf26)
            buf27 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf25, buf26, buf27, 1048576, stream=raw_stream0)
            assert_size_stride(primals_12, (3072, 8192), (8192, 1), 'input')
            buf28 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11, down_proj], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf27, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_12, (8192, 3072), (1, 8192), 0), out=buf28)
            assert_size_stride(primals_13, (3072, ), (1, ), 'input')
            buf29 = reinterpret_tensor(buf28, (1, 128, 3072), (393216, 3072, 1), 0); del buf28  # reuse
            buf30 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf31 = reinterpret_tensor(buf30, (1, 128, 1), (128, 1, 1), 0); del buf30  # reuse
            buf32 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, to_11, hidden_states_12], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf29, buf31, buf0, buf21, primals_13, buf32, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_14, (3072, 3072), (3072, 1), 'input')
            buf33 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, to_11, hidden_states_12, linear_7], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_14, (3072, 3072), (1, 3072), 0), out=buf33)
            assert_size_stride(primals_15, (1024, 3072), (3072, 1), 'input')
            buf34 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, to_11, hidden_states_12, linear_7, linear_8], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_15, (3072, 1024), (1, 3072), 0), out=buf34)
            assert_size_stride(primals_16, (1024, 3072), (3072, 1), 'input')
            buf35 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, to_11, hidden_states_12, linear_7, linear_9], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_16, (3072, 1024), (1, 3072), 0), out=buf35)
            buf36 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf771 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_7, view_3, query_states_1, mul_14, x1_2, x2_2, neg_2, cat_3, mul_15, q_embed_1, matmul_3, permute_1323], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf33, primals_3, buf36, buf771, 393216, stream=raw_stream0)
            buf37 = reinterpret_tensor(buf33, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_8, view_4, key_states_2, mul_16, x1_3, x2_3, neg_3, cat_4, mul_17, k_embed_1, getitem_14, hidden_states_13, key_states_3], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf34, primals_3, buf37, 393216, stream=raw_stream0)
            buf38 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_7, view_3, query_states_1, linear_8, view_4, key_states_2, mul_14, x1_2, x2_2, neg_2, cat_3, mul_15, q_embed_1, mul_16, x1_3, x2_3, neg_3, cat_4, mul_17, k_embed_1, getitem_14, hidden_states_13, key_states_3, transpose_9, matmul_3], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf36, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf37, (24, 128, 128), (16384, 1, 128), 0), out=buf38)
            buf39 = reinterpret_tensor(buf38, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf38  # reuse
            buf40 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf41 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf42 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_3, attn_weights_4, attn_weights_5, softmax_1, attn_weights_6], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf39, buf4, buf40, buf41, buf42, 3072, 128, stream=raw_stream0)
            buf43 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, view_5, value_states_2, getitem_15, hidden_states_14, value_states_3], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf35, buf43, 393216, stream=raw_stream0)
            buf44 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, view_5, value_states_2, getitem_15, hidden_states_14, value_states_3, softmax_1, attn_weights_6, attn_output_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf42, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf43, (24, 128, 128), (16384, 128, 1), 0), out=buf44)
            buf45 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_4, transpose_10, attn_output_5], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf44, buf45, 393216, stream=raw_stream0)
            assert_size_stride(primals_17, (3072, 3072), (3072, 1), 'input')
            buf46 = reinterpret_tensor(buf44, (128, 3072), (3072, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [attn_output_4, transpose_10, attn_output_5, reshape_5, attn_output_7], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf45, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_17, (3072, 3072), (1, 3072), 0), out=buf46)
            assert_size_stride(primals_18, (3072, ), (1, ), 'input')
            buf47 = reinterpret_tensor(buf46, (1, 128, 3072), (393216, 3072, 1), 0); del buf46  # reuse
            buf48 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf49 = reinterpret_tensor(buf48, (1, 128, 1), (128, 1, 1), 0); del buf48  # reuse
            buf50 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_7, hidden_states_15, hidden_states_16, pow_4, variance_3, add_12, rsqrt_3, hidden_states_17, to_14, hidden_states_18], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf47, buf49, buf29, primals_18, buf50, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_19, (8192, 3072), (3072, 1), 'input')
            buf51 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17, to_14, hidden_states_18, linear_11], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf50, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_19, (3072, 8192), (1, 3072), 0), out=buf51)
            assert_size_stride(primals_20, (8192, 3072), (3072, 1), 'input')
            buf52 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17, to_14, hidden_states_18, linear_11, linear_12], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf50, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_20, (3072, 8192), (1, 3072), 0), out=buf52)
            buf53 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_21], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf51, buf52, buf53, 1048576, stream=raw_stream0)
            assert_size_stride(primals_21, (3072, 8192), (8192, 1), 'input')
            buf54 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_21, down_proj_1], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf53, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_21, (8192, 3072), (1, 8192), 0), out=buf54)
            assert_size_stride(primals_22, (3072, ), (1, ), 'input')
            buf55 = reinterpret_tensor(buf54, (1, 128, 3072), (393216, 3072, 1), 0); del buf54  # reuse
            buf56 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf57 = reinterpret_tensor(buf56, (1, 128, 1), (128, 1, 1), 0); del buf56  # reuse
            buf58 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, hidden_states_20, pow_5, variance_4, add_14, rsqrt_4, hidden_states_21, to_16, hidden_states_22], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf55, buf57, buf47, primals_22, buf58, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_23, (3072, 3072), (3072, 1), 'input')
            buf59 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21, to_16, hidden_states_22, linear_14], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf58, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_23, (3072, 3072), (1, 3072), 0), out=buf59)
            assert_size_stride(primals_24, (1024, 3072), (3072, 1), 'input')
            buf60 = buf35; del buf35  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21, to_16, hidden_states_22, linear_14, linear_15], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf58, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_24, (3072, 1024), (1, 3072), 0), out=buf60)
            assert_size_stride(primals_25, (1024, 3072), (3072, 1), 'input')
            buf61 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21, to_16, hidden_states_22, linear_14, linear_16], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf58, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_25, (3072, 1024), (1, 3072), 0), out=buf61)
            buf62 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf770 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_14, view_6, query_states_2, mul_24, x1_4, x2_4, neg_4, cat_5, mul_25, q_embed_2, matmul_5, permute_1286], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf59, primals_3, buf62, buf770, 393216, stream=raw_stream0)
            buf63 = reinterpret_tensor(buf59, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_15, view_7, key_states_4, mul_26, x1_5, x2_5, neg_5, cat_6, mul_27, k_embed_2, getitem_21, hidden_states_23, key_states_5], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf60, primals_3, buf63, 393216, stream=raw_stream0)
            buf64 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_14, view_6, query_states_2, linear_15, view_7, key_states_4, mul_24, x1_4, x2_4, neg_4, cat_5, mul_25, q_embed_2, mul_26, x1_5, x2_5, neg_5, cat_6, mul_27, k_embed_2, getitem_21, hidden_states_23, key_states_5, transpose_14, matmul_5], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf62, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf63, (24, 128, 128), (16384, 1, 128), 0), out=buf64)
            buf65 = reinterpret_tensor(buf64, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf64  # reuse
            buf66 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf67 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf68 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_5, attn_weights_8, attn_weights_9, softmax_2, attn_weights_10], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf65, buf4, buf66, buf67, buf68, 3072, 128, stream=raw_stream0)
            buf69 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_16, view_8, value_states_4, getitem_22, hidden_states_24, value_states_5], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf61, buf69, 393216, stream=raw_stream0)
            buf70 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_16, view_8, value_states_4, getitem_22, hidden_states_24, value_states_5, softmax_2, attn_weights_10, attn_output_8], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf68, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf69, (24, 128, 128), (16384, 128, 1), 0), out=buf70)
            buf71 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_8, transpose_15, attn_output_9], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf70, buf71, 393216, stream=raw_stream0)
            assert_size_stride(primals_26, (3072, 3072), (3072, 1), 'input')
            buf72 = reinterpret_tensor(buf70, (128, 3072), (3072, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [attn_output_8, transpose_15, attn_output_9, reshape_8, attn_output_11], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf71, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_26, (3072, 3072), (1, 3072), 0), out=buf72)
            assert_size_stride(primals_27, (3072, ), (1, ), 'input')
            buf73 = reinterpret_tensor(buf72, (1, 128, 3072), (393216, 3072, 1), 0); del buf72  # reuse
            buf74 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf75 = reinterpret_tensor(buf74, (1, 128, 1), (128, 1, 1), 0); del buf74  # reuse
            buf76 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_11, hidden_states_25, hidden_states_26, pow_6, variance_5, add_19, rsqrt_5, hidden_states_27, to_19, hidden_states_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf73, buf75, buf55, primals_27, buf76, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_28, (8192, 3072), (3072, 1), 'input')
            buf77 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27, to_19, hidden_states_28, linear_18], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf76, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_28, (3072, 8192), (1, 3072), 0), out=buf77)
            assert_size_stride(primals_29, (8192, 3072), (3072, 1), 'input')
            buf78 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27, to_19, hidden_states_28, linear_18, linear_19], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf76, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_29, (3072, 8192), (1, 3072), 0), out=buf78)
            buf79 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_31], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf77, buf78, buf79, 1048576, stream=raw_stream0)
            assert_size_stride(primals_30, (3072, 8192), (8192, 1), 'input')
            buf80 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_31, down_proj_2], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf79, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_30, (8192, 3072), (1, 8192), 0), out=buf80)
            assert_size_stride(primals_31, (3072, ), (1, ), 'input')
            buf81 = reinterpret_tensor(buf80, (1, 128, 3072), (393216, 3072, 1), 0); del buf80  # reuse
            buf82 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf83 = reinterpret_tensor(buf82, (1, 128, 1), (128, 1, 1), 0); del buf82  # reuse
            buf84 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_2, hidden_states_29, hidden_states_30, pow_7, variance_6, add_21, rsqrt_6, hidden_states_31, to_21, hidden_states_32], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf81, buf83, buf73, primals_31, buf84, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_32, (3072, 3072), (3072, 1), 'input')
            buf85 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31, to_21, hidden_states_32, linear_21], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf84, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_32, (3072, 3072), (1, 3072), 0), out=buf85)
            assert_size_stride(primals_33, (1024, 3072), (3072, 1), 'input')
            buf86 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31, to_21, hidden_states_32, linear_21, linear_22], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf84, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_33, (3072, 1024), (1, 3072), 0), out=buf86)
            assert_size_stride(primals_34, (1024, 3072), (3072, 1), 'input')
            buf87 = buf60; del buf60  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31, to_21, hidden_states_32, linear_21, linear_23], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf84, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_34, (3072, 1024), (1, 3072), 0), out=buf87)
            buf88 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf769 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_21, view_9, query_states_3, mul_34, x1_6, x2_6, neg_6, cat_7, mul_35, q_embed_3, matmul_7, permute_1249], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf85, primals_3, buf88, buf769, 393216, stream=raw_stream0)
            buf89 = reinterpret_tensor(buf85, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_22, view_10, key_states_6, mul_36, x1_7, x2_7, neg_7, cat_8, mul_37, k_embed_3, getitem_28, hidden_states_33, key_states_7], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf86, primals_3, buf89, 393216, stream=raw_stream0)
            buf90 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_21, view_9, query_states_3, linear_22, view_10, key_states_6, mul_34, x1_6, x2_6, neg_6, cat_7, mul_35, q_embed_3, mul_36, x1_7, x2_7, neg_7, cat_8, mul_37, k_embed_3, getitem_28, hidden_states_33, key_states_7, transpose_19, matmul_7], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf88, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf89, (24, 128, 128), (16384, 1, 128), 0), out=buf90)
            buf91 = reinterpret_tensor(buf90, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf90  # reuse
            buf92 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf93 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf94 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_7, attn_weights_12, attn_weights_13, softmax_3, attn_weights_14], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf91, buf4, buf92, buf93, buf94, 3072, 128, stream=raw_stream0)
            buf95 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_23, view_11, value_states_6, getitem_29, hidden_states_34, value_states_7], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf87, buf95, 393216, stream=raw_stream0)
            buf96 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_23, view_11, value_states_6, getitem_29, hidden_states_34, value_states_7, softmax_3, attn_weights_14, attn_output_12], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf94, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf95, (24, 128, 128), (16384, 128, 1), 0), out=buf96)
            buf97 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_12, transpose_20, attn_output_13], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf96, buf97, 393216, stream=raw_stream0)
            assert_size_stride(primals_35, (3072, 3072), (3072, 1), 'input')
            buf98 = reinterpret_tensor(buf96, (128, 3072), (3072, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, transpose_20, attn_output_13, reshape_11, attn_output_15], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf97, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_35, (3072, 3072), (1, 3072), 0), out=buf98)
            assert_size_stride(primals_36, (3072, ), (1, ), 'input')
            buf99 = reinterpret_tensor(buf98, (1, 128, 3072), (393216, 3072, 1), 0); del buf98  # reuse
            buf100 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf101 = reinterpret_tensor(buf100, (1, 128, 1), (128, 1, 1), 0); del buf100  # reuse
            buf102 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_15, hidden_states_35, hidden_states_36, pow_8, variance_7, add_26, rsqrt_7, hidden_states_37, to_24, hidden_states_38], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf99, buf101, buf81, primals_36, buf102, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_37, (8192, 3072), (3072, 1), 'input')
            buf103 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_37, to_24, hidden_states_38, linear_25], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf102, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_37, (3072, 8192), (1, 3072), 0), out=buf103)
            assert_size_stride(primals_38, (8192, 3072), (3072, 1), 'input')
            buf104 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_37, to_24, hidden_states_38, linear_25, linear_26], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf102, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_38, (3072, 8192), (1, 3072), 0), out=buf104)
            buf105 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_41], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf103, buf104, buf105, 1048576, stream=raw_stream0)
            assert_size_stride(primals_39, (3072, 8192), (8192, 1), 'input')
            buf106 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_41, down_proj_3], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf105, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_39, (8192, 3072), (1, 8192), 0), out=buf106)
            assert_size_stride(primals_40, (3072, ), (1, ), 'input')
            buf107 = reinterpret_tensor(buf106, (1, 128, 3072), (393216, 3072, 1), 0); del buf106  # reuse
            buf108 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf109 = reinterpret_tensor(buf108, (1, 128, 1), (128, 1, 1), 0); del buf108  # reuse
            buf110 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, hidden_states_40, pow_9, variance_8, add_28, rsqrt_8, hidden_states_41, to_26, hidden_states_42], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf107, buf109, buf99, primals_40, buf110, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_41, (3072, 3072), (3072, 1), 'input')
            buf111 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41, to_26, hidden_states_42, linear_28], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf110, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_41, (3072, 3072), (1, 3072), 0), out=buf111)
            assert_size_stride(primals_42, (1024, 3072), (3072, 1), 'input')
            buf112 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41, to_26, hidden_states_42, linear_28, linear_29], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf110, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_42, (3072, 1024), (1, 3072), 0), out=buf112)
            assert_size_stride(primals_43, (1024, 3072), (3072, 1), 'input')
            buf113 = buf86; del buf86  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41, to_26, hidden_states_42, linear_28, linear_30], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf110, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_43, (3072, 1024), (1, 3072), 0), out=buf113)
            buf114 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf768 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_28, view_12, query_states_4, mul_44, x1_8, x2_8, neg_8, cat_9, mul_45, q_embed_4, matmul_9, permute_1212], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf111, primals_3, buf114, buf768, 393216, stream=raw_stream0)
            buf115 = reinterpret_tensor(buf111, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf111  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_29, view_13, key_states_8, mul_46, x1_9, x2_9, neg_9, cat_10, mul_47, k_embed_4, getitem_35, hidden_states_43, key_states_9], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf112, primals_3, buf115, 393216, stream=raw_stream0)
            buf116 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_28, view_12, query_states_4, linear_29, view_13, key_states_8, mul_44, x1_8, x2_8, neg_8, cat_9, mul_45, q_embed_4, mul_46, x1_9, x2_9, neg_9, cat_10, mul_47, k_embed_4, getitem_35, hidden_states_43, key_states_9, transpose_24, matmul_9], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf114, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf115, (24, 128, 128), (16384, 1, 128), 0), out=buf116)
            buf117 = reinterpret_tensor(buf116, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf116  # reuse
            buf118 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf119 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf120 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_9, attn_weights_16, attn_weights_17, softmax_4, attn_weights_18], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf117, buf4, buf118, buf119, buf120, 3072, 128, stream=raw_stream0)
            buf121 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30, view_14, value_states_8, getitem_36, hidden_states_44, value_states_9], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf113, buf121, 393216, stream=raw_stream0)
            buf122 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30, view_14, value_states_8, getitem_36, hidden_states_44, value_states_9, softmax_4, attn_weights_18, attn_output_16], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf120, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf121, (24, 128, 128), (16384, 128, 1), 0), out=buf122)
            buf123 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_16, transpose_25, attn_output_17], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf122, buf123, 393216, stream=raw_stream0)
            assert_size_stride(primals_44, (3072, 3072), (3072, 1), 'input')
            buf124 = reinterpret_tensor(buf122, (128, 3072), (3072, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [attn_output_16, transpose_25, attn_output_17, reshape_14, attn_output_19], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf123, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_44, (3072, 3072), (1, 3072), 0), out=buf124)
            assert_size_stride(primals_45, (3072, ), (1, ), 'input')
            buf125 = reinterpret_tensor(buf124, (1, 128, 3072), (393216, 3072, 1), 0); del buf124  # reuse
            buf126 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf127 = reinterpret_tensor(buf126, (1, 128, 1), (128, 1, 1), 0); del buf126  # reuse
            buf128 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_19, hidden_states_45, hidden_states_46, pow_10, variance_9, add_33, rsqrt_9, hidden_states_47, to_29, hidden_states_48], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf125, buf127, buf107, primals_45, buf128, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_46, (8192, 3072), (3072, 1), 'input')
            buf129 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47, to_29, hidden_states_48, linear_32], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf128, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_46, (3072, 8192), (1, 3072), 0), out=buf129)
            assert_size_stride(primals_47, (8192, 3072), (3072, 1), 'input')
            buf130 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47, to_29, hidden_states_48, linear_32, linear_33], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf128, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_47, (3072, 8192), (1, 3072), 0), out=buf130)
            buf131 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_51], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf129, buf130, buf131, 1048576, stream=raw_stream0)
            assert_size_stride(primals_48, (3072, 8192), (8192, 1), 'input')
            buf132 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_51, down_proj_4], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf131, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_48, (8192, 3072), (1, 8192), 0), out=buf132)
            assert_size_stride(primals_49, (3072, ), (1, ), 'input')
            buf133 = reinterpret_tensor(buf132, (1, 128, 3072), (393216, 3072, 1), 0); del buf132  # reuse
            buf134 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf135 = reinterpret_tensor(buf134, (1, 128, 1), (128, 1, 1), 0); del buf134  # reuse
            buf136 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_4, hidden_states_49, hidden_states_50, pow_11, variance_10, add_35, rsqrt_10, hidden_states_51, to_31, hidden_states_52], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf133, buf135, buf125, primals_49, buf136, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_50, (3072, 3072), (3072, 1), 'input')
            buf137 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51, to_31, hidden_states_52, linear_35], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf136, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_50, (3072, 3072), (1, 3072), 0), out=buf137)
            assert_size_stride(primals_51, (1024, 3072), (3072, 1), 'input')
            buf138 = buf113; del buf113  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51, to_31, hidden_states_52, linear_35, linear_36], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf136, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_51, (3072, 1024), (1, 3072), 0), out=buf138)
            assert_size_stride(primals_52, (1024, 3072), (3072, 1), 'input')
            buf139 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51, to_31, hidden_states_52, linear_35, linear_37], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf136, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_52, (3072, 1024), (1, 3072), 0), out=buf139)
            buf140 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf767 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_35, view_15, query_states_5, mul_54, x1_10, x2_10, neg_10, cat_11, mul_55, q_embed_5, matmul_11, permute_1175], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf137, primals_3, buf140, buf767, 393216, stream=raw_stream0)
            buf141 = reinterpret_tensor(buf137, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_36, view_16, key_states_10, mul_56, x1_11, x2_11, neg_11, cat_12, mul_57, k_embed_5, getitem_42, hidden_states_53, key_states_11], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf138, primals_3, buf141, 393216, stream=raw_stream0)
            buf142 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_35, view_15, query_states_5, linear_36, view_16, key_states_10, mul_54, x1_10, x2_10, neg_10, cat_11, mul_55, q_embed_5, mul_56, x1_11, x2_11, neg_11, cat_12, mul_57, k_embed_5, getitem_42, hidden_states_53, key_states_11, transpose_29, matmul_11], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf140, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf141, (24, 128, 128), (16384, 1, 128), 0), out=buf142)
            buf143 = reinterpret_tensor(buf142, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf142  # reuse
            buf144 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf145 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf146 = buf140; del buf140  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_11, attn_weights_20, attn_weights_21, softmax_5, attn_weights_22], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf143, buf4, buf144, buf145, buf146, 3072, 128, stream=raw_stream0)
            buf147 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_37, view_17, value_states_10, getitem_43, hidden_states_54, value_states_11], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf139, buf147, 393216, stream=raw_stream0)
            buf148 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_37, view_17, value_states_10, getitem_43, hidden_states_54, value_states_11, softmax_5, attn_weights_22, attn_output_20], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf146, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf147, (24, 128, 128), (16384, 128, 1), 0), out=buf148)
            buf149 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_20, transpose_30, attn_output_21], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf148, buf149, 393216, stream=raw_stream0)
            assert_size_stride(primals_53, (3072, 3072), (3072, 1), 'input')
            buf150 = reinterpret_tensor(buf148, (128, 3072), (3072, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [attn_output_20, transpose_30, attn_output_21, reshape_17, attn_output_23], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf149, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_53, (3072, 3072), (1, 3072), 0), out=buf150)
            assert_size_stride(primals_54, (3072, ), (1, ), 'input')
            buf151 = reinterpret_tensor(buf150, (1, 128, 3072), (393216, 3072, 1), 0); del buf150  # reuse
            buf152 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf153 = reinterpret_tensor(buf152, (1, 128, 1), (128, 1, 1), 0); del buf152  # reuse
            buf154 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_23, hidden_states_55, hidden_states_56, pow_12, variance_11, add_40, rsqrt_11, hidden_states_57, to_34, hidden_states_58], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf151, buf153, buf133, primals_54, buf154, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_55, (8192, 3072), (3072, 1), 'input')
            buf155 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57, to_34, hidden_states_58, linear_39], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_55, (3072, 8192), (1, 3072), 0), out=buf155)
            assert_size_stride(primals_56, (8192, 3072), (3072, 1), 'input')
            buf156 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57, to_34, hidden_states_58, linear_39, linear_40], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_56, (3072, 8192), (1, 3072), 0), out=buf156)
            buf157 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_61], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf155, buf156, buf157, 1048576, stream=raw_stream0)
            assert_size_stride(primals_57, (3072, 8192), (8192, 1), 'input')
            buf158 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_61, down_proj_5], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf157, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_57, (8192, 3072), (1, 8192), 0), out=buf158)
            assert_size_stride(primals_58, (3072, ), (1, ), 'input')
            buf159 = reinterpret_tensor(buf158, (1, 128, 3072), (393216, 3072, 1), 0); del buf158  # reuse
            buf160 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf161 = reinterpret_tensor(buf160, (1, 128, 1), (128, 1, 1), 0); del buf160  # reuse
            buf162 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, hidden_states_60, pow_13, variance_12, add_42, rsqrt_12, hidden_states_61, to_36, hidden_states_62], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf159, buf161, buf151, primals_58, buf162, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_59, (3072, 3072), (3072, 1), 'input')
            buf163 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_60, hidden_states_61, to_36, hidden_states_62, linear_42], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf162, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 3072), (1, 3072), 0), out=buf163)
            assert_size_stride(primals_60, (1024, 3072), (3072, 1), 'input')
            buf164 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_60, hidden_states_61, to_36, hidden_states_62, linear_42, linear_43], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf162, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_60, (3072, 1024), (1, 3072), 0), out=buf164)
            assert_size_stride(primals_61, (1024, 3072), (3072, 1), 'input')
            buf165 = buf138; del buf138  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_60, hidden_states_61, to_36, hidden_states_62, linear_42, linear_44], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf162, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_61, (3072, 1024), (1, 3072), 0), out=buf165)
            buf166 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf766 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_42, view_18, query_states_6, mul_64, x1_12, x2_12, neg_12, cat_13, mul_65, q_embed_6, matmul_13, permute_1138], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf163, primals_3, buf166, buf766, 393216, stream=raw_stream0)
            buf167 = reinterpret_tensor(buf163, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf163  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_43, view_19, key_states_12, mul_66, x1_13, x2_13, neg_13, cat_14, mul_67, k_embed_6, getitem_49, hidden_states_63, key_states_13], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf164, primals_3, buf167, 393216, stream=raw_stream0)
            buf168 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_42, view_18, query_states_6, linear_43, view_19, key_states_12, mul_64, x1_12, x2_12, neg_12, cat_13, mul_65, q_embed_6, mul_66, x1_13, x2_13, neg_13, cat_14, mul_67, k_embed_6, getitem_49, hidden_states_63, key_states_13, transpose_34, matmul_13], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf166, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf167, (24, 128, 128), (16384, 1, 128), 0), out=buf168)
            buf169 = reinterpret_tensor(buf168, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf168  # reuse
            buf170 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf171 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf172 = buf166; del buf166  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_13, attn_weights_24, attn_weights_25, softmax_6, attn_weights_26], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf169, buf4, buf170, buf171, buf172, 3072, 128, stream=raw_stream0)
            buf173 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_44, view_20, value_states_12, getitem_50, hidden_states_64, value_states_13], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf165, buf173, 393216, stream=raw_stream0)
            buf174 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_44, view_20, value_states_12, getitem_50, hidden_states_64, value_states_13, softmax_6, attn_weights_26, attn_output_24], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf172, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf173, (24, 128, 128), (16384, 128, 1), 0), out=buf174)
            buf175 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_24, transpose_35, attn_output_25], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf174, buf175, 393216, stream=raw_stream0)
            assert_size_stride(primals_62, (3072, 3072), (3072, 1), 'input')
            buf176 = reinterpret_tensor(buf174, (128, 3072), (3072, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, transpose_35, attn_output_25, reshape_20, attn_output_27], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf175, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_62, (3072, 3072), (1, 3072), 0), out=buf176)
            assert_size_stride(primals_63, (3072, ), (1, ), 'input')
            buf177 = reinterpret_tensor(buf176, (1, 128, 3072), (393216, 3072, 1), 0); del buf176  # reuse
            buf178 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf179 = reinterpret_tensor(buf178, (1, 128, 1), (128, 1, 1), 0); del buf178  # reuse
            buf180 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_27, hidden_states_65, hidden_states_66, pow_14, variance_13, add_47, rsqrt_13, hidden_states_67, to_39, hidden_states_68], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf177, buf179, buf159, primals_63, buf180, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_64, (8192, 3072), (3072, 1), 'input')
            buf181 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, to_39, hidden_states_68, linear_46], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf180, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_64, (3072, 8192), (1, 3072), 0), out=buf181)
            assert_size_stride(primals_65, (8192, 3072), (3072, 1), 'input')
            buf182 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, to_39, hidden_states_68, linear_46, linear_47], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf180, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_65, (3072, 8192), (1, 3072), 0), out=buf182)
            buf183 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_71], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf181, buf182, buf183, 1048576, stream=raw_stream0)
            assert_size_stride(primals_66, (3072, 8192), (8192, 1), 'input')
            buf184 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_71, down_proj_6], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf183, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_66, (8192, 3072), (1, 8192), 0), out=buf184)
            assert_size_stride(primals_67, (3072, ), (1, ), 'input')
            buf185 = reinterpret_tensor(buf184, (1, 128, 3072), (393216, 3072, 1), 0); del buf184  # reuse
            buf186 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf187 = reinterpret_tensor(buf186, (1, 128, 1), (128, 1, 1), 0); del buf186  # reuse
            buf188 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_6, hidden_states_69, hidden_states_70, pow_15, variance_14, add_49, rsqrt_14, hidden_states_71, to_41, hidden_states_72], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf185, buf187, buf177, primals_67, buf188, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_68, (3072, 3072), (3072, 1), 'input')
            buf189 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71, to_41, hidden_states_72, linear_49], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf188, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_68, (3072, 3072), (1, 3072), 0), out=buf189)
            assert_size_stride(primals_69, (1024, 3072), (3072, 1), 'input')
            buf190 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71, to_41, hidden_states_72, linear_49, linear_50], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf188, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_69, (3072, 1024), (1, 3072), 0), out=buf190)
            assert_size_stride(primals_70, (1024, 3072), (3072, 1), 'input')
            buf191 = buf164; del buf164  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71, to_41, hidden_states_72, linear_49, linear_51], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf188, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_70, (3072, 1024), (1, 3072), 0), out=buf191)
            buf192 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf765 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_49, view_21, query_states_7, mul_74, x1_14, x2_14, neg_14, cat_15, mul_75, q_embed_7, matmul_15, permute_1101], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf189, primals_3, buf192, buf765, 393216, stream=raw_stream0)
            buf193 = reinterpret_tensor(buf189, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_50, view_22, key_states_14, mul_76, x1_15, x2_15, neg_15, cat_16, mul_77, k_embed_7, getitem_56, hidden_states_73, key_states_15], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf190, primals_3, buf193, 393216, stream=raw_stream0)
            buf194 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_49, view_21, query_states_7, linear_50, view_22, key_states_14, mul_74, x1_14, x2_14, neg_14, cat_15, mul_75, q_embed_7, mul_76, x1_15, x2_15, neg_15, cat_16, mul_77, k_embed_7, getitem_56, hidden_states_73, key_states_15, transpose_39, matmul_15], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf192, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf193, (24, 128, 128), (16384, 1, 128), 0), out=buf194)
            buf195 = reinterpret_tensor(buf194, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf194  # reuse
            buf196 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf197 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf198 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_15, attn_weights_28, attn_weights_29, softmax_7, attn_weights_30], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf195, buf4, buf196, buf197, buf198, 3072, 128, stream=raw_stream0)
            buf199 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_51, view_23, value_states_14, getitem_57, hidden_states_74, value_states_15], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf191, buf199, 393216, stream=raw_stream0)
            buf200 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_51, view_23, value_states_14, getitem_57, hidden_states_74, value_states_15, softmax_7, attn_weights_30, attn_output_28], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf198, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf199, (24, 128, 128), (16384, 128, 1), 0), out=buf200)
            buf201 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_28, transpose_40, attn_output_29], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf200, buf201, 393216, stream=raw_stream0)
            assert_size_stride(primals_71, (3072, 3072), (3072, 1), 'input')
            buf202 = reinterpret_tensor(buf200, (128, 3072), (3072, 1), 0); del buf200  # reuse
            # Topologically Sorted Source Nodes: [attn_output_28, transpose_40, attn_output_29, reshape_23, attn_output_31], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf201, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 3072), (1, 3072), 0), out=buf202)
            assert_size_stride(primals_72, (3072, ), (1, ), 'input')
            buf203 = reinterpret_tensor(buf202, (1, 128, 3072), (393216, 3072, 1), 0); del buf202  # reuse
            buf204 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf205 = reinterpret_tensor(buf204, (1, 128, 1), (128, 1, 1), 0); del buf204  # reuse
            buf206 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_31, hidden_states_75, hidden_states_76, pow_16, variance_15, add_54, rsqrt_15, hidden_states_77, to_44, hidden_states_78], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf203, buf205, buf185, primals_72, buf206, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_73, (8192, 3072), (3072, 1), 'input')
            buf207 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77, to_44, hidden_states_78, linear_53], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf206, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_73, (3072, 8192), (1, 3072), 0), out=buf207)
            assert_size_stride(primals_74, (8192, 3072), (3072, 1), 'input')
            buf208 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77, to_44, hidden_states_78, linear_53, linear_54], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf206, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_74, (3072, 8192), (1, 3072), 0), out=buf208)
            buf209 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_81], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf207, buf208, buf209, 1048576, stream=raw_stream0)
            assert_size_stride(primals_75, (3072, 8192), (8192, 1), 'input')
            buf210 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_81, down_proj_7], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf209, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_75, (8192, 3072), (1, 8192), 0), out=buf210)
            assert_size_stride(primals_76, (3072, ), (1, ), 'input')
            buf211 = reinterpret_tensor(buf210, (1, 128, 3072), (393216, 3072, 1), 0); del buf210  # reuse
            buf212 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf213 = reinterpret_tensor(buf212, (1, 128, 1), (128, 1, 1), 0); del buf212  # reuse
            buf214 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, hidden_states_80, pow_17, variance_16, add_56, rsqrt_16, hidden_states_81, to_46, hidden_states_82], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf211, buf213, buf203, primals_76, buf214, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_77, (3072, 3072), (3072, 1), 'input')
            buf215 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, to_46, hidden_states_82, linear_56], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_77, (3072, 3072), (1, 3072), 0), out=buf215)
            assert_size_stride(primals_78, (1024, 3072), (3072, 1), 'input')
            buf216 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, to_46, hidden_states_82, linear_56, linear_57], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_78, (3072, 1024), (1, 3072), 0), out=buf216)
            assert_size_stride(primals_79, (1024, 3072), (3072, 1), 'input')
            buf217 = buf190; del buf190  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, to_46, hidden_states_82, linear_56, linear_58], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_79, (3072, 1024), (1, 3072), 0), out=buf217)
            buf218 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf764 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_56, view_24, query_states_8, mul_84, x1_16, x2_16, neg_16, cat_17, mul_85, q_embed_8, matmul_17, permute_1064], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf215, primals_3, buf218, buf764, 393216, stream=raw_stream0)
            buf219 = reinterpret_tensor(buf215, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_57, view_25, key_states_16, mul_86, x1_17, x2_17, neg_17, cat_18, mul_87, k_embed_8, getitem_63, hidden_states_83, key_states_17], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf216, primals_3, buf219, 393216, stream=raw_stream0)
            buf220 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_56, view_24, query_states_8, linear_57, view_25, key_states_16, mul_84, x1_16, x2_16, neg_16, cat_17, mul_85, q_embed_8, mul_86, x1_17, x2_17, neg_17, cat_18, mul_87, k_embed_8, getitem_63, hidden_states_83, key_states_17, transpose_44, matmul_17], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf218, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf219, (24, 128, 128), (16384, 1, 128), 0), out=buf220)
            buf221 = reinterpret_tensor(buf220, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf220  # reuse
            buf222 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf223 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf224 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_17, attn_weights_32, attn_weights_33, softmax_8, attn_weights_34], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf221, buf4, buf222, buf223, buf224, 3072, 128, stream=raw_stream0)
            buf225 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_58, view_26, value_states_16, getitem_64, hidden_states_84, value_states_17], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf217, buf225, 393216, stream=raw_stream0)
            buf226 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_58, view_26, value_states_16, getitem_64, hidden_states_84, value_states_17, softmax_8, attn_weights_34, attn_output_32], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf224, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf225, (24, 128, 128), (16384, 128, 1), 0), out=buf226)
            buf227 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_32, transpose_45, attn_output_33], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf226, buf227, 393216, stream=raw_stream0)
            assert_size_stride(primals_80, (3072, 3072), (3072, 1), 'input')
            buf228 = reinterpret_tensor(buf226, (128, 3072), (3072, 1), 0); del buf226  # reuse
            # Topologically Sorted Source Nodes: [attn_output_32, transpose_45, attn_output_33, reshape_26, attn_output_35], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf227, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 3072), (1, 3072), 0), out=buf228)
            assert_size_stride(primals_81, (3072, ), (1, ), 'input')
            buf229 = reinterpret_tensor(buf228, (1, 128, 3072), (393216, 3072, 1), 0); del buf228  # reuse
            buf230 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf231 = reinterpret_tensor(buf230, (1, 128, 1), (128, 1, 1), 0); del buf230  # reuse
            buf232 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_35, hidden_states_85, hidden_states_86, pow_18, variance_17, add_61, rsqrt_17, hidden_states_87, to_49, hidden_states_88], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf229, buf231, buf211, primals_81, buf232, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_82, (8192, 3072), (3072, 1), 'input')
            buf233 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87, to_49, hidden_states_88, linear_60], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_82, (3072, 8192), (1, 3072), 0), out=buf233)
            assert_size_stride(primals_83, (8192, 3072), (3072, 1), 'input')
            buf234 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87, to_49, hidden_states_88, linear_60, linear_61], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 8192), (1, 3072), 0), out=buf234)
            buf235 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_91], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf233, buf234, buf235, 1048576, stream=raw_stream0)
            assert_size_stride(primals_84, (3072, 8192), (8192, 1), 'input')
            buf236 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_91, down_proj_8], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf235, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_84, (8192, 3072), (1, 8192), 0), out=buf236)
            assert_size_stride(primals_85, (3072, ), (1, ), 'input')
            buf237 = reinterpret_tensor(buf236, (1, 128, 3072), (393216, 3072, 1), 0); del buf236  # reuse
            buf238 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf239 = reinterpret_tensor(buf238, (1, 128, 1), (128, 1, 1), 0); del buf238  # reuse
            buf240 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_8, hidden_states_89, hidden_states_90, pow_19, variance_18, add_63, rsqrt_18, hidden_states_91, to_51, hidden_states_92], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf237, buf239, buf229, primals_85, buf240, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_86, (3072, 3072), (3072, 1), 'input')
            buf241 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91, to_51, hidden_states_92, linear_63], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_86, (3072, 3072), (1, 3072), 0), out=buf241)
            assert_size_stride(primals_87, (1024, 3072), (3072, 1), 'input')
            buf242 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91, to_51, hidden_states_92, linear_63, linear_64], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_87, (3072, 1024), (1, 3072), 0), out=buf242)
            assert_size_stride(primals_88, (1024, 3072), (3072, 1), 'input')
            buf243 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91, to_51, hidden_states_92, linear_63, linear_65], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_88, (3072, 1024), (1, 3072), 0), out=buf243)
            buf244 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf763 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_63, view_27, query_states_9, mul_94, x1_18, x2_18, neg_18, cat_19, mul_95, q_embed_9, matmul_19, permute_1027], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf241, primals_3, buf244, buf763, 393216, stream=raw_stream0)
            buf245 = reinterpret_tensor(buf241, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_64, view_28, key_states_18, mul_96, x1_19, x2_19, neg_19, cat_20, mul_97, k_embed_9, getitem_70, hidden_states_93, key_states_19], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf242, primals_3, buf245, 393216, stream=raw_stream0)
            buf246 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_63, view_27, query_states_9, linear_64, view_28, key_states_18, mul_94, x1_18, x2_18, neg_18, cat_19, mul_95, q_embed_9, mul_96, x1_19, x2_19, neg_19, cat_20, mul_97, k_embed_9, getitem_70, hidden_states_93, key_states_19, transpose_49, matmul_19], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf244, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf245, (24, 128, 128), (16384, 1, 128), 0), out=buf246)
            buf247 = reinterpret_tensor(buf246, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf246  # reuse
            buf248 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf249 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf250 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_19, attn_weights_36, attn_weights_37, softmax_9, attn_weights_38], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf247, buf4, buf248, buf249, buf250, 3072, 128, stream=raw_stream0)
            buf251 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_65, view_29, value_states_18, getitem_71, hidden_states_94, value_states_19], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf243, buf251, 393216, stream=raw_stream0)
            buf252 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_65, view_29, value_states_18, getitem_71, hidden_states_94, value_states_19, softmax_9, attn_weights_38, attn_output_36], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf250, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf251, (24, 128, 128), (16384, 128, 1), 0), out=buf252)
            buf253 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_36, transpose_50, attn_output_37], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf252, buf253, 393216, stream=raw_stream0)
            assert_size_stride(primals_89, (3072, 3072), (3072, 1), 'input')
            buf254 = reinterpret_tensor(buf252, (128, 3072), (3072, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [attn_output_36, transpose_50, attn_output_37, reshape_29, attn_output_39], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf253, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_89, (3072, 3072), (1, 3072), 0), out=buf254)
            assert_size_stride(primals_90, (3072, ), (1, ), 'input')
            buf255 = reinterpret_tensor(buf254, (1, 128, 3072), (393216, 3072, 1), 0); del buf254  # reuse
            buf256 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf257 = reinterpret_tensor(buf256, (1, 128, 1), (128, 1, 1), 0); del buf256  # reuse
            buf258 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_39, hidden_states_95, hidden_states_96, pow_20, variance_19, add_68, rsqrt_19, hidden_states_97, to_54, hidden_states_98], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf255, buf257, buf237, primals_90, buf258, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_91, (8192, 3072), (3072, 1), 'input')
            buf259 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97, to_54, hidden_states_98, linear_67], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf258, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 8192), (1, 3072), 0), out=buf259)
            assert_size_stride(primals_92, (8192, 3072), (3072, 1), 'input')
            buf260 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97, to_54, hidden_states_98, linear_67, linear_68], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf258, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_92, (3072, 8192), (1, 3072), 0), out=buf260)
            buf261 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_101], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf259, buf260, buf261, 1048576, stream=raw_stream0)
            assert_size_stride(primals_93, (3072, 8192), (8192, 1), 'input')
            buf262 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_101, down_proj_9], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf261, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_93, (8192, 3072), (1, 8192), 0), out=buf262)
            assert_size_stride(primals_94, (3072, ), (1, ), 'input')
            buf263 = reinterpret_tensor(buf262, (1, 128, 3072), (393216, 3072, 1), 0); del buf262  # reuse
            buf264 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf265 = reinterpret_tensor(buf264, (1, 128, 1), (128, 1, 1), 0); del buf264  # reuse
            buf266 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, hidden_states_100, pow_21, variance_20, add_70, rsqrt_20, hidden_states_101, to_56, hidden_states_102], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf263, buf265, buf255, primals_94, buf266, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_95, (3072, 3072), (3072, 1), 'input')
            buf267 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101, to_56, hidden_states_102, linear_70], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf266, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 3072), (1, 3072), 0), out=buf267)
            assert_size_stride(primals_96, (1024, 3072), (3072, 1), 'input')
            buf268 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101, to_56, hidden_states_102, linear_70, linear_71], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf266, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_96, (3072, 1024), (1, 3072), 0), out=buf268)
            assert_size_stride(primals_97, (1024, 3072), (3072, 1), 'input')
            buf269 = buf242; del buf242  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101, to_56, hidden_states_102, linear_70, linear_72], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf266, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_97, (3072, 1024), (1, 3072), 0), out=buf269)
            buf270 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf762 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_70, view_30, query_states_10, mul_104, x1_20, x2_20, neg_20, cat_21, mul_105, q_embed_10, matmul_21, permute_990], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf267, primals_3, buf270, buf762, 393216, stream=raw_stream0)
            buf271 = reinterpret_tensor(buf267, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_71, view_31, key_states_20, mul_106, x1_21, x2_21, neg_21, cat_22, mul_107, k_embed_10, getitem_77, hidden_states_103, key_states_21], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf268, primals_3, buf271, 393216, stream=raw_stream0)
            buf272 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_70, view_30, query_states_10, linear_71, view_31, key_states_20, mul_104, x1_20, x2_20, neg_20, cat_21, mul_105, q_embed_10, mul_106, x1_21, x2_21, neg_21, cat_22, mul_107, k_embed_10, getitem_77, hidden_states_103, key_states_21, transpose_54, matmul_21], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf270, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf271, (24, 128, 128), (16384, 1, 128), 0), out=buf272)
            buf273 = reinterpret_tensor(buf272, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf272  # reuse
            buf274 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf275 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf276 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_21, attn_weights_40, attn_weights_41, softmax_10, attn_weights_42], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf273, buf4, buf274, buf275, buf276, 3072, 128, stream=raw_stream0)
            buf277 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_72, view_32, value_states_20, getitem_78, hidden_states_104, value_states_21], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf269, buf277, 393216, stream=raw_stream0)
            buf278 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_72, view_32, value_states_20, getitem_78, hidden_states_104, value_states_21, softmax_10, attn_weights_42, attn_output_40], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf276, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf277, (24, 128, 128), (16384, 128, 1), 0), out=buf278)
            buf279 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_40, transpose_55, attn_output_41], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf278, buf279, 393216, stream=raw_stream0)
            assert_size_stride(primals_98, (3072, 3072), (3072, 1), 'input')
            buf280 = reinterpret_tensor(buf278, (128, 3072), (3072, 1), 0); del buf278  # reuse
            # Topologically Sorted Source Nodes: [attn_output_40, transpose_55, attn_output_41, reshape_32, attn_output_43], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf279, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_98, (3072, 3072), (1, 3072), 0), out=buf280)
            assert_size_stride(primals_99, (3072, ), (1, ), 'input')
            buf281 = reinterpret_tensor(buf280, (1, 128, 3072), (393216, 3072, 1), 0); del buf280  # reuse
            buf282 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf283 = reinterpret_tensor(buf282, (1, 128, 1), (128, 1, 1), 0); del buf282  # reuse
            buf284 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_43, hidden_states_105, hidden_states_106, pow_22, variance_21, add_75, rsqrt_21, hidden_states_107, to_59, hidden_states_108], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf281, buf283, buf263, primals_99, buf284, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_100, (8192, 3072), (3072, 1), 'input')
            buf285 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_106, hidden_states_107, to_59, hidden_states_108, linear_74], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf284, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_100, (3072, 8192), (1, 3072), 0), out=buf285)
            assert_size_stride(primals_101, (8192, 3072), (3072, 1), 'input')
            buf286 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_106, hidden_states_107, to_59, hidden_states_108, linear_74, linear_75], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf284, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_101, (3072, 8192), (1, 3072), 0), out=buf286)
            buf287 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_111], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf285, buf286, buf287, 1048576, stream=raw_stream0)
            assert_size_stride(primals_102, (3072, 8192), (8192, 1), 'input')
            buf288 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_111, down_proj_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf287, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_102, (8192, 3072), (1, 8192), 0), out=buf288)
            assert_size_stride(primals_103, (3072, ), (1, ), 'input')
            buf289 = reinterpret_tensor(buf288, (1, 128, 3072), (393216, 3072, 1), 0); del buf288  # reuse
            buf290 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf291 = reinterpret_tensor(buf290, (1, 128, 1), (128, 1, 1), 0); del buf290  # reuse
            buf292 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_10, hidden_states_109, hidden_states_110, pow_23, variance_22, add_77, rsqrt_22, hidden_states_111, to_61, hidden_states_112], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf289, buf291, buf281, primals_103, buf292, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_104, (3072, 3072), (3072, 1), 'input')
            buf293 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111, to_61, hidden_states_112, linear_77], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf292, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_104, (3072, 3072), (1, 3072), 0), out=buf293)
            assert_size_stride(primals_105, (1024, 3072), (3072, 1), 'input')
            buf294 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111, to_61, hidden_states_112, linear_77, linear_78], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf292, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_105, (3072, 1024), (1, 3072), 0), out=buf294)
            assert_size_stride(primals_106, (1024, 3072), (3072, 1), 'input')
            buf295 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111, to_61, hidden_states_112, linear_77, linear_79], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf292, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_106, (3072, 1024), (1, 3072), 0), out=buf295)
            buf296 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf761 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_77, view_33, query_states_11, mul_114, x1_22, x2_22, neg_22, cat_23, mul_115, q_embed_11, matmul_23, permute_953], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf293, primals_3, buf296, buf761, 393216, stream=raw_stream0)
            buf297 = reinterpret_tensor(buf293, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_78, view_34, key_states_22, mul_116, x1_23, x2_23, neg_23, cat_24, mul_117, k_embed_11, getitem_84, hidden_states_113, key_states_23], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf294, primals_3, buf297, 393216, stream=raw_stream0)
            buf298 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_77, view_33, query_states_11, linear_78, view_34, key_states_22, mul_114, x1_22, x2_22, neg_22, cat_23, mul_115, q_embed_11, mul_116, x1_23, x2_23, neg_23, cat_24, mul_117, k_embed_11, getitem_84, hidden_states_113, key_states_23, transpose_59, matmul_23], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf296, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf297, (24, 128, 128), (16384, 1, 128), 0), out=buf298)
            buf299 = reinterpret_tensor(buf298, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf298  # reuse
            buf300 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf301 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf302 = buf296; del buf296  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_23, attn_weights_44, attn_weights_45, softmax_11, attn_weights_46], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf299, buf4, buf300, buf301, buf302, 3072, 128, stream=raw_stream0)
            buf303 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_79, view_35, value_states_22, getitem_85, hidden_states_114, value_states_23], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf295, buf303, 393216, stream=raw_stream0)
            buf304 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_79, view_35, value_states_22, getitem_85, hidden_states_114, value_states_23, softmax_11, attn_weights_46, attn_output_44], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf302, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf303, (24, 128, 128), (16384, 128, 1), 0), out=buf304)
            buf305 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_44, transpose_60, attn_output_45], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf304, buf305, 393216, stream=raw_stream0)
            assert_size_stride(primals_107, (3072, 3072), (3072, 1), 'input')
            buf306 = reinterpret_tensor(buf304, (128, 3072), (3072, 1), 0); del buf304  # reuse
            # Topologically Sorted Source Nodes: [attn_output_44, transpose_60, attn_output_45, reshape_35, attn_output_47], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf305, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 3072), (1, 3072), 0), out=buf306)
            assert_size_stride(primals_108, (3072, ), (1, ), 'input')
            buf307 = reinterpret_tensor(buf306, (1, 128, 3072), (393216, 3072, 1), 0); del buf306  # reuse
            buf308 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf309 = reinterpret_tensor(buf308, (1, 128, 1), (128, 1, 1), 0); del buf308  # reuse
            buf310 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_47, hidden_states_115, hidden_states_116, pow_24, variance_23, add_82, rsqrt_23, hidden_states_117, to_64, hidden_states_118], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf307, buf309, buf289, primals_108, buf310, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_109, (8192, 3072), (3072, 1), 'input')
            buf311 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_117, to_64, hidden_states_118, linear_81], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf310, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_109, (3072, 8192), (1, 3072), 0), out=buf311)
            assert_size_stride(primals_110, (8192, 3072), (3072, 1), 'input')
            buf312 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_117, to_64, hidden_states_118, linear_81, linear_82], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf310, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_110, (3072, 8192), (1, 3072), 0), out=buf312)
            buf313 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_121], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf311, buf312, buf313, 1048576, stream=raw_stream0)
            assert_size_stride(primals_111, (3072, 8192), (8192, 1), 'input')
            buf314 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_121, down_proj_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf313, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_111, (8192, 3072), (1, 8192), 0), out=buf314)
            assert_size_stride(primals_112, (3072, ), (1, ), 'input')
            buf315 = reinterpret_tensor(buf314, (1, 128, 3072), (393216, 3072, 1), 0); del buf314  # reuse
            buf316 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf317 = reinterpret_tensor(buf316, (1, 128, 1), (128, 1, 1), 0); del buf316  # reuse
            buf318 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, hidden_states_120, pow_25, variance_24, add_84, rsqrt_24, hidden_states_121, to_66, hidden_states_122], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf315, buf317, buf307, primals_112, buf318, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_113, (3072, 3072), (3072, 1), 'input')
            buf319 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_120, hidden_states_121, to_66, hidden_states_122, linear_84], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf318, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_113, (3072, 3072), (1, 3072), 0), out=buf319)
            assert_size_stride(primals_114, (1024, 3072), (3072, 1), 'input')
            buf320 = buf295; del buf295  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_120, hidden_states_121, to_66, hidden_states_122, linear_84, linear_85], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf318, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_114, (3072, 1024), (1, 3072), 0), out=buf320)
            assert_size_stride(primals_115, (1024, 3072), (3072, 1), 'input')
            buf321 = buf294; del buf294  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_120, hidden_states_121, to_66, hidden_states_122, linear_84, linear_86], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf318, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_115, (3072, 1024), (1, 3072), 0), out=buf321)
            buf322 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf760 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_84, view_36, query_states_12, mul_124, x1_24, x2_24, neg_24, cat_25, mul_125, q_embed_12, matmul_25, permute_916], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf319, primals_3, buf322, buf760, 393216, stream=raw_stream0)
            buf323 = reinterpret_tensor(buf319, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_85, view_37, key_states_24, mul_126, x1_25, x2_25, neg_25, cat_26, mul_127, k_embed_12, getitem_91, hidden_states_123, key_states_25], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf320, primals_3, buf323, 393216, stream=raw_stream0)
            buf324 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_84, view_36, query_states_12, linear_85, view_37, key_states_24, mul_124, x1_24, x2_24, neg_24, cat_25, mul_125, q_embed_12, mul_126, x1_25, x2_25, neg_25, cat_26, mul_127, k_embed_12, getitem_91, hidden_states_123, key_states_25, transpose_64, matmul_25], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf322, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf323, (24, 128, 128), (16384, 1, 128), 0), out=buf324)
            buf325 = reinterpret_tensor(buf324, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf324  # reuse
            buf326 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf327 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf328 = buf322; del buf322  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_25, attn_weights_48, attn_weights_49, softmax_12, attn_weights_50], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf325, buf4, buf326, buf327, buf328, 3072, 128, stream=raw_stream0)
            buf329 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_86, view_38, value_states_24, getitem_92, hidden_states_124, value_states_25], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf321, buf329, 393216, stream=raw_stream0)
            buf330 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_86, view_38, value_states_24, getitem_92, hidden_states_124, value_states_25, softmax_12, attn_weights_50, attn_output_48], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf328, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf329, (24, 128, 128), (16384, 128, 1), 0), out=buf330)
            buf331 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_48, transpose_65, attn_output_49], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf330, buf331, 393216, stream=raw_stream0)
            assert_size_stride(primals_116, (3072, 3072), (3072, 1), 'input')
            buf332 = reinterpret_tensor(buf330, (128, 3072), (3072, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [attn_output_48, transpose_65, attn_output_49, reshape_38, attn_output_51], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf331, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_116, (3072, 3072), (1, 3072), 0), out=buf332)
            assert_size_stride(primals_117, (3072, ), (1, ), 'input')
            buf333 = reinterpret_tensor(buf332, (1, 128, 3072), (393216, 3072, 1), 0); del buf332  # reuse
            buf334 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf335 = reinterpret_tensor(buf334, (1, 128, 1), (128, 1, 1), 0); del buf334  # reuse
            buf336 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_51, hidden_states_125, hidden_states_126, pow_26, variance_25, add_89, rsqrt_25, hidden_states_127, to_69, hidden_states_128], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf333, buf335, buf315, primals_117, buf336, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_118, (8192, 3072), (3072, 1), 'input')
            buf337 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_127, to_69, hidden_states_128, linear_88], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf336, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_118, (3072, 8192), (1, 3072), 0), out=buf337)
            assert_size_stride(primals_119, (8192, 3072), (3072, 1), 'input')
            buf338 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_127, to_69, hidden_states_128, linear_88, linear_89], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf336, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_119, (3072, 8192), (1, 3072), 0), out=buf338)
            buf339 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_131], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf337, buf338, buf339, 1048576, stream=raw_stream0)
            assert_size_stride(primals_120, (3072, 8192), (8192, 1), 'input')
            buf340 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_131, down_proj_12], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf339, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_120, (8192, 3072), (1, 8192), 0), out=buf340)
            assert_size_stride(primals_121, (3072, ), (1, ), 'input')
            buf341 = reinterpret_tensor(buf340, (1, 128, 3072), (393216, 3072, 1), 0); del buf340  # reuse
            buf342 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf343 = reinterpret_tensor(buf342, (1, 128, 1), (128, 1, 1), 0); del buf342  # reuse
            buf344 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_12, hidden_states_129, hidden_states_130, pow_27, variance_26, add_91, rsqrt_26, hidden_states_131, to_71, hidden_states_132], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf341, buf343, buf333, primals_121, buf344, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_122, (3072, 3072), (3072, 1), 'input')
            buf345 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131, to_71, hidden_states_132, linear_91], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf344, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 3072), (1, 3072), 0), out=buf345)
            assert_size_stride(primals_123, (1024, 3072), (3072, 1), 'input')
            buf346 = buf321; del buf321  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131, to_71, hidden_states_132, linear_91, linear_92], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf344, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_123, (3072, 1024), (1, 3072), 0), out=buf346)
            assert_size_stride(primals_124, (1024, 3072), (3072, 1), 'input')
            buf347 = buf320; del buf320  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131, to_71, hidden_states_132, linear_91, linear_93], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf344, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_124, (3072, 1024), (1, 3072), 0), out=buf347)
            buf348 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf759 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_91, view_39, query_states_13, mul_134, x1_26, x2_26, neg_26, cat_27, mul_135, q_embed_13, matmul_27, permute_879], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf345, primals_3, buf348, buf759, 393216, stream=raw_stream0)
            buf349 = reinterpret_tensor(buf345, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf345  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_92, view_40, key_states_26, mul_136, x1_27, x2_27, neg_27, cat_28, mul_137, k_embed_13, getitem_98, hidden_states_133, key_states_27], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf346, primals_3, buf349, 393216, stream=raw_stream0)
            buf350 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_91, view_39, query_states_13, linear_92, view_40, key_states_26, mul_134, x1_26, x2_26, neg_26, cat_27, mul_135, q_embed_13, mul_136, x1_27, x2_27, neg_27, cat_28, mul_137, k_embed_13, getitem_98, hidden_states_133, key_states_27, transpose_69, matmul_27], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf348, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf349, (24, 128, 128), (16384, 1, 128), 0), out=buf350)
            buf351 = reinterpret_tensor(buf350, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf350  # reuse
            buf352 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf353 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf354 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_27, attn_weights_52, attn_weights_53, softmax_13, attn_weights_54], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf351, buf4, buf352, buf353, buf354, 3072, 128, stream=raw_stream0)
            buf355 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_93, view_41, value_states_26, getitem_99, hidden_states_134, value_states_27], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf347, buf355, 393216, stream=raw_stream0)
            buf356 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_93, view_41, value_states_26, getitem_99, hidden_states_134, value_states_27, softmax_13, attn_weights_54, attn_output_52], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf354, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf355, (24, 128, 128), (16384, 128, 1), 0), out=buf356)
            buf357 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_52, transpose_70, attn_output_53], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf356, buf357, 393216, stream=raw_stream0)
            assert_size_stride(primals_125, (3072, 3072), (3072, 1), 'input')
            buf358 = reinterpret_tensor(buf356, (128, 3072), (3072, 1), 0); del buf356  # reuse
            # Topologically Sorted Source Nodes: [attn_output_52, transpose_70, attn_output_53, reshape_41, attn_output_55], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf357, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_125, (3072, 3072), (1, 3072), 0), out=buf358)
            assert_size_stride(primals_126, (3072, ), (1, ), 'input')
            buf359 = reinterpret_tensor(buf358, (1, 128, 3072), (393216, 3072, 1), 0); del buf358  # reuse
            buf360 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf361 = reinterpret_tensor(buf360, (1, 128, 1), (128, 1, 1), 0); del buf360  # reuse
            buf362 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_55, hidden_states_135, hidden_states_136, pow_28, variance_27, add_96, rsqrt_27, hidden_states_137, to_74, hidden_states_138], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf359, buf361, buf341, primals_126, buf362, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_127, (8192, 3072), (3072, 1), 'input')
            buf363 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137, to_74, hidden_states_138, linear_95], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf362, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_127, (3072, 8192), (1, 3072), 0), out=buf363)
            assert_size_stride(primals_128, (8192, 3072), (3072, 1), 'input')
            buf364 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137, to_74, hidden_states_138, linear_95, linear_96], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf362, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_128, (3072, 8192), (1, 3072), 0), out=buf364)
            buf365 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_141], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf363, buf364, buf365, 1048576, stream=raw_stream0)
            assert_size_stride(primals_129, (3072, 8192), (8192, 1), 'input')
            buf366 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_141, down_proj_13], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf365, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_129, (8192, 3072), (1, 8192), 0), out=buf366)
            assert_size_stride(primals_130, (3072, ), (1, ), 'input')
            buf367 = reinterpret_tensor(buf366, (1, 128, 3072), (393216, 3072, 1), 0); del buf366  # reuse
            buf368 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf369 = reinterpret_tensor(buf368, (1, 128, 1), (128, 1, 1), 0); del buf368  # reuse
            buf370 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, hidden_states_140, pow_29, variance_28, add_98, rsqrt_28, hidden_states_141, to_76, hidden_states_142], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf367, buf369, buf359, primals_130, buf370, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_131, (3072, 3072), (3072, 1), 'input')
            buf371 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_140, hidden_states_141, to_76, hidden_states_142, linear_98], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 3072), (1, 3072), 0), out=buf371)
            assert_size_stride(primals_132, (1024, 3072), (3072, 1), 'input')
            buf372 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_140, hidden_states_141, to_76, hidden_states_142, linear_98, linear_99], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_132, (3072, 1024), (1, 3072), 0), out=buf372)
            assert_size_stride(primals_133, (1024, 3072), (3072, 1), 'input')
            buf373 = buf346; del buf346  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_140, hidden_states_141, to_76, hidden_states_142, linear_98, linear_100], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_133, (3072, 1024), (1, 3072), 0), out=buf373)
            buf374 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf758 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_98, view_42, query_states_14, mul_144, x1_28, x2_28, neg_28, cat_29, mul_145, q_embed_14, matmul_29, permute_842], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf371, primals_3, buf374, buf758, 393216, stream=raw_stream0)
            buf375 = reinterpret_tensor(buf371, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_99, view_43, key_states_28, mul_146, x1_29, x2_29, neg_29, cat_30, mul_147, k_embed_14, getitem_105, hidden_states_143, key_states_29], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf372, primals_3, buf375, 393216, stream=raw_stream0)
            buf376 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_98, view_42, query_states_14, linear_99, view_43, key_states_28, mul_144, x1_28, x2_28, neg_28, cat_29, mul_145, q_embed_14, mul_146, x1_29, x2_29, neg_29, cat_30, mul_147, k_embed_14, getitem_105, hidden_states_143, key_states_29, transpose_74, matmul_29], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf374, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf375, (24, 128, 128), (16384, 1, 128), 0), out=buf376)
            buf377 = reinterpret_tensor(buf376, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf376  # reuse
            buf378 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf379 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf380 = buf374; del buf374  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_29, attn_weights_56, attn_weights_57, softmax_14, attn_weights_58], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf377, buf4, buf378, buf379, buf380, 3072, 128, stream=raw_stream0)
            buf381 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_100, view_44, value_states_28, getitem_106, hidden_states_144, value_states_29], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf373, buf381, 393216, stream=raw_stream0)
            buf382 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_100, view_44, value_states_28, getitem_106, hidden_states_144, value_states_29, softmax_14, attn_weights_58, attn_output_56], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf380, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf381, (24, 128, 128), (16384, 128, 1), 0), out=buf382)
            buf383 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_56, transpose_75, attn_output_57], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf382, buf383, 393216, stream=raw_stream0)
            assert_size_stride(primals_134, (3072, 3072), (3072, 1), 'input')
            buf384 = reinterpret_tensor(buf382, (128, 3072), (3072, 1), 0); del buf382  # reuse
            # Topologically Sorted Source Nodes: [attn_output_56, transpose_75, attn_output_57, reshape_44, attn_output_59], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf383, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_134, (3072, 3072), (1, 3072), 0), out=buf384)
            assert_size_stride(primals_135, (3072, ), (1, ), 'input')
            buf385 = reinterpret_tensor(buf384, (1, 128, 3072), (393216, 3072, 1), 0); del buf384  # reuse
            buf386 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf387 = reinterpret_tensor(buf386, (1, 128, 1), (128, 1, 1), 0); del buf386  # reuse
            buf388 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_59, hidden_states_145, hidden_states_146, pow_30, variance_29, add_103, rsqrt_29, hidden_states_147, to_79, hidden_states_148], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf385, buf387, buf367, primals_135, buf388, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_136, (8192, 3072), (3072, 1), 'input')
            buf389 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_146, hidden_states_147, to_79, hidden_states_148, linear_102], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_136, (3072, 8192), (1, 3072), 0), out=buf389)
            assert_size_stride(primals_137, (8192, 3072), (3072, 1), 'input')
            buf390 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_146, hidden_states_147, to_79, hidden_states_148, linear_102, linear_103], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_137, (3072, 8192), (1, 3072), 0), out=buf390)
            buf391 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_151], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf389, buf390, buf391, 1048576, stream=raw_stream0)
            assert_size_stride(primals_138, (3072, 8192), (8192, 1), 'input')
            buf392 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_151, down_proj_14], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf391, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_138, (8192, 3072), (1, 8192), 0), out=buf392)
            assert_size_stride(primals_139, (3072, ), (1, ), 'input')
            buf393 = reinterpret_tensor(buf392, (1, 128, 3072), (393216, 3072, 1), 0); del buf392  # reuse
            buf394 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf395 = reinterpret_tensor(buf394, (1, 128, 1), (128, 1, 1), 0); del buf394  # reuse
            buf396 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_14, hidden_states_149, hidden_states_150, pow_31, variance_30, add_105, rsqrt_30, hidden_states_151, to_81, hidden_states_152], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf393, buf395, buf385, primals_139, buf396, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_140, (3072, 3072), (3072, 1), 'input')
            buf397 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, to_81, hidden_states_152, linear_105], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf396, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_140, (3072, 3072), (1, 3072), 0), out=buf397)
            assert_size_stride(primals_141, (1024, 3072), (3072, 1), 'input')
            buf398 = buf373; del buf373  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, to_81, hidden_states_152, linear_105, linear_106], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf396, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_141, (3072, 1024), (1, 3072), 0), out=buf398)
            assert_size_stride(primals_142, (1024, 3072), (3072, 1), 'input')
            buf399 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, to_81, hidden_states_152, linear_105, linear_107], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf396, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_142, (3072, 1024), (1, 3072), 0), out=buf399)
            buf400 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf757 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_105, view_45, query_states_15, mul_154, x1_30, x2_30, neg_30, cat_31, mul_155, q_embed_15, matmul_31, permute_805], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf397, primals_3, buf400, buf757, 393216, stream=raw_stream0)
            buf401 = reinterpret_tensor(buf397, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf397  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_106, view_46, key_states_30, mul_156, x1_31, x2_31, neg_31, cat_32, mul_157, k_embed_15, getitem_112, hidden_states_153, key_states_31], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf398, primals_3, buf401, 393216, stream=raw_stream0)
            buf402 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_105, view_45, query_states_15, linear_106, view_46, key_states_30, mul_154, x1_30, x2_30, neg_30, cat_31, mul_155, q_embed_15, mul_156, x1_31, x2_31, neg_31, cat_32, mul_157, k_embed_15, getitem_112, hidden_states_153, key_states_31, transpose_79, matmul_31], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf400, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf401, (24, 128, 128), (16384, 1, 128), 0), out=buf402)
            buf403 = reinterpret_tensor(buf402, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf402  # reuse
            buf404 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf405 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf406 = buf400; del buf400  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_31, attn_weights_60, attn_weights_61, softmax_15, attn_weights_62], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf403, buf4, buf404, buf405, buf406, 3072, 128, stream=raw_stream0)
            buf407 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_107, view_47, value_states_30, getitem_113, hidden_states_154, value_states_31], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf399, buf407, 393216, stream=raw_stream0)
            buf408 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_107, view_47, value_states_30, getitem_113, hidden_states_154, value_states_31, softmax_15, attn_weights_62, attn_output_60], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf406, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf407, (24, 128, 128), (16384, 128, 1), 0), out=buf408)
            buf409 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_60, transpose_80, attn_output_61], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf408, buf409, 393216, stream=raw_stream0)
            assert_size_stride(primals_143, (3072, 3072), (3072, 1), 'input')
            buf410 = reinterpret_tensor(buf408, (128, 3072), (3072, 1), 0); del buf408  # reuse
            # Topologically Sorted Source Nodes: [attn_output_60, transpose_80, attn_output_61, reshape_47, attn_output_63], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf409, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 3072), (1, 3072), 0), out=buf410)
            assert_size_stride(primals_144, (3072, ), (1, ), 'input')
            buf411 = reinterpret_tensor(buf410, (1, 128, 3072), (393216, 3072, 1), 0); del buf410  # reuse
            buf412 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf413 = reinterpret_tensor(buf412, (1, 128, 1), (128, 1, 1), 0); del buf412  # reuse
            buf414 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_63, hidden_states_155, hidden_states_156, pow_32, variance_31, add_110, rsqrt_31, hidden_states_157, to_84, hidden_states_158], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf411, buf413, buf393, primals_144, buf414, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_145, (8192, 3072), (3072, 1), 'input')
            buf415 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_157, to_84, hidden_states_158, linear_109], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf414, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_145, (3072, 8192), (1, 3072), 0), out=buf415)
            assert_size_stride(primals_146, (8192, 3072), (3072, 1), 'input')
            buf416 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_157, to_84, hidden_states_158, linear_109, linear_110], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf414, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_146, (3072, 8192), (1, 3072), 0), out=buf416)
            buf417 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_161], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf415, buf416, buf417, 1048576, stream=raw_stream0)
            assert_size_stride(primals_147, (3072, 8192), (8192, 1), 'input')
            buf418 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_161, down_proj_15], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf417, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_147, (8192, 3072), (1, 8192), 0), out=buf418)
            assert_size_stride(primals_148, (3072, ), (1, ), 'input')
            buf419 = reinterpret_tensor(buf418, (1, 128, 3072), (393216, 3072, 1), 0); del buf418  # reuse
            buf420 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf421 = reinterpret_tensor(buf420, (1, 128, 1), (128, 1, 1), 0); del buf420  # reuse
            buf422 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, hidden_states_160, pow_33, variance_32, add_112, rsqrt_32, hidden_states_161, to_86, hidden_states_162], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf419, buf421, buf411, primals_148, buf422, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_149, (3072, 3072), (3072, 1), 'input')
            buf423 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161, to_86, hidden_states_162, linear_112], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf422, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_149, (3072, 3072), (1, 3072), 0), out=buf423)
            assert_size_stride(primals_150, (1024, 3072), (3072, 1), 'input')
            buf424 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161, to_86, hidden_states_162, linear_112, linear_113], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf422, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_150, (3072, 1024), (1, 3072), 0), out=buf424)
            assert_size_stride(primals_151, (1024, 3072), (3072, 1), 'input')
            buf425 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161, to_86, hidden_states_162, linear_112, linear_114], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf422, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_151, (3072, 1024), (1, 3072), 0), out=buf425)
            buf426 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf756 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_112, view_48, query_states_16, mul_164, x1_32, x2_32, neg_32, cat_33, mul_165, q_embed_16, matmul_33, permute_768], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf423, primals_3, buf426, buf756, 393216, stream=raw_stream0)
            buf427 = reinterpret_tensor(buf423, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf423  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_113, view_49, key_states_32, mul_166, x1_33, x2_33, neg_33, cat_34, mul_167, k_embed_16, getitem_119, hidden_states_163, key_states_33], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf424, primals_3, buf427, 393216, stream=raw_stream0)
            buf428 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_112, view_48, query_states_16, linear_113, view_49, key_states_32, mul_164, x1_32, x2_32, neg_32, cat_33, mul_165, q_embed_16, mul_166, x1_33, x2_33, neg_33, cat_34, mul_167, k_embed_16, getitem_119, hidden_states_163, key_states_33, transpose_84, matmul_33], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf426, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf427, (24, 128, 128), (16384, 1, 128), 0), out=buf428)
            buf429 = reinterpret_tensor(buf428, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf428  # reuse
            buf430 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf431 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf432 = buf426; del buf426  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_33, attn_weights_64, attn_weights_65, softmax_16, attn_weights_66], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf429, buf4, buf430, buf431, buf432, 3072, 128, stream=raw_stream0)
            buf433 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_114, view_50, value_states_32, getitem_120, hidden_states_164, value_states_33], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf425, buf433, 393216, stream=raw_stream0)
            buf434 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_114, view_50, value_states_32, getitem_120, hidden_states_164, value_states_33, softmax_16, attn_weights_66, attn_output_64], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf432, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf433, (24, 128, 128), (16384, 128, 1), 0), out=buf434)
            buf435 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_64, transpose_85, attn_output_65], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf434, buf435, 393216, stream=raw_stream0)
            assert_size_stride(primals_152, (3072, 3072), (3072, 1), 'input')
            buf436 = reinterpret_tensor(buf434, (128, 3072), (3072, 1), 0); del buf434  # reuse
            # Topologically Sorted Source Nodes: [attn_output_64, transpose_85, attn_output_65, reshape_50, attn_output_67], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf435, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_152, (3072, 3072), (1, 3072), 0), out=buf436)
            assert_size_stride(primals_153, (3072, ), (1, ), 'input')
            buf437 = reinterpret_tensor(buf436, (1, 128, 3072), (393216, 3072, 1), 0); del buf436  # reuse
            buf438 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf439 = reinterpret_tensor(buf438, (1, 128, 1), (128, 1, 1), 0); del buf438  # reuse
            buf440 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_67, hidden_states_165, hidden_states_166, pow_34, variance_33, add_117, rsqrt_33, hidden_states_167, to_89, hidden_states_168], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf437, buf439, buf419, primals_153, buf440, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_154, (8192, 3072), (3072, 1), 'input')
            buf441 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_167, to_89, hidden_states_168, linear_116], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf440, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_154, (3072, 8192), (1, 3072), 0), out=buf441)
            assert_size_stride(primals_155, (8192, 3072), (3072, 1), 'input')
            buf442 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_167, to_89, hidden_states_168, linear_116, linear_117], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf440, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_155, (3072, 8192), (1, 3072), 0), out=buf442)
            buf443 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_171], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf441, buf442, buf443, 1048576, stream=raw_stream0)
            assert_size_stride(primals_156, (3072, 8192), (8192, 1), 'input')
            buf444 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_171, down_proj_16], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf443, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_156, (8192, 3072), (1, 8192), 0), out=buf444)
            assert_size_stride(primals_157, (3072, ), (1, ), 'input')
            buf445 = reinterpret_tensor(buf444, (1, 128, 3072), (393216, 3072, 1), 0); del buf444  # reuse
            buf446 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf447 = reinterpret_tensor(buf446, (1, 128, 1), (128, 1, 1), 0); del buf446  # reuse
            buf448 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_16, hidden_states_169, hidden_states_170, pow_35, variance_34, add_119, rsqrt_34, hidden_states_171, to_91, hidden_states_172], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf445, buf447, buf437, primals_157, buf448, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_158, (3072, 3072), (3072, 1), 'input')
            buf449 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171, to_91, hidden_states_172, linear_119], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf448, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_158, (3072, 3072), (1, 3072), 0), out=buf449)
            assert_size_stride(primals_159, (1024, 3072), (3072, 1), 'input')
            buf450 = buf425; del buf425  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171, to_91, hidden_states_172, linear_119, linear_120], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf448, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_159, (3072, 1024), (1, 3072), 0), out=buf450)
            assert_size_stride(primals_160, (1024, 3072), (3072, 1), 'input')
            buf451 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171, to_91, hidden_states_172, linear_119, linear_121], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf448, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_160, (3072, 1024), (1, 3072), 0), out=buf451)
            buf452 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf755 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_119, view_51, query_states_17, mul_174, x1_34, x2_34, neg_34, cat_35, mul_175, q_embed_17, matmul_35, permute_731], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf449, primals_3, buf452, buf755, 393216, stream=raw_stream0)
            buf453 = reinterpret_tensor(buf449, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf449  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_120, view_52, key_states_34, mul_176, x1_35, x2_35, neg_35, cat_36, mul_177, k_embed_17, getitem_126, hidden_states_173, key_states_35], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf450, primals_3, buf453, 393216, stream=raw_stream0)
            buf454 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_119, view_51, query_states_17, linear_120, view_52, key_states_34, mul_174, x1_34, x2_34, neg_34, cat_35, mul_175, q_embed_17, mul_176, x1_35, x2_35, neg_35, cat_36, mul_177, k_embed_17, getitem_126, hidden_states_173, key_states_35, transpose_89, matmul_35], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf452, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf453, (24, 128, 128), (16384, 1, 128), 0), out=buf454)
            buf455 = reinterpret_tensor(buf454, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf454  # reuse
            buf456 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf457 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf458 = buf452; del buf452  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_35, attn_weights_68, attn_weights_69, softmax_17, attn_weights_70], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf455, buf4, buf456, buf457, buf458, 3072, 128, stream=raw_stream0)
            buf459 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_121, view_53, value_states_34, getitem_127, hidden_states_174, value_states_35], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf451, buf459, 393216, stream=raw_stream0)
            buf460 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_121, view_53, value_states_34, getitem_127, hidden_states_174, value_states_35, softmax_17, attn_weights_70, attn_output_68], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf458, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf459, (24, 128, 128), (16384, 128, 1), 0), out=buf460)
            buf461 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_68, transpose_90, attn_output_69], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf460, buf461, 393216, stream=raw_stream0)
            assert_size_stride(primals_161, (3072, 3072), (3072, 1), 'input')
            buf462 = reinterpret_tensor(buf460, (128, 3072), (3072, 1), 0); del buf460  # reuse
            # Topologically Sorted Source Nodes: [attn_output_68, transpose_90, attn_output_69, reshape_53, attn_output_71], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf461, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_161, (3072, 3072), (1, 3072), 0), out=buf462)
            assert_size_stride(primals_162, (3072, ), (1, ), 'input')
            buf463 = reinterpret_tensor(buf462, (1, 128, 3072), (393216, 3072, 1), 0); del buf462  # reuse
            buf464 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf465 = reinterpret_tensor(buf464, (1, 128, 1), (128, 1, 1), 0); del buf464  # reuse
            buf466 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_71, hidden_states_175, hidden_states_176, pow_36, variance_35, add_124, rsqrt_35, hidden_states_177, to_94, hidden_states_178], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf463, buf465, buf445, primals_162, buf466, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_163, (8192, 3072), (3072, 1), 'input')
            buf467 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_176, hidden_states_177, to_94, hidden_states_178, linear_123], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf466, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_163, (3072, 8192), (1, 3072), 0), out=buf467)
            assert_size_stride(primals_164, (8192, 3072), (3072, 1), 'input')
            buf468 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_176, hidden_states_177, to_94, hidden_states_178, linear_123, linear_124], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf466, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_164, (3072, 8192), (1, 3072), 0), out=buf468)
            buf469 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_181], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf467, buf468, buf469, 1048576, stream=raw_stream0)
            assert_size_stride(primals_165, (3072, 8192), (8192, 1), 'input')
            buf470 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_181, down_proj_17], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf469, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_165, (8192, 3072), (1, 8192), 0), out=buf470)
            assert_size_stride(primals_166, (3072, ), (1, ), 'input')
            buf471 = reinterpret_tensor(buf470, (1, 128, 3072), (393216, 3072, 1), 0); del buf470  # reuse
            buf472 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf473 = reinterpret_tensor(buf472, (1, 128, 1), (128, 1, 1), 0); del buf472  # reuse
            buf474 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, hidden_states_180, pow_37, variance_36, add_126, rsqrt_36, hidden_states_181, to_96, hidden_states_182], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf471, buf473, buf463, primals_166, buf474, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_167, (3072, 3072), (3072, 1), 'input')
            buf475 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181, to_96, hidden_states_182, linear_126], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf474, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_167, (3072, 3072), (1, 3072), 0), out=buf475)
            assert_size_stride(primals_168, (1024, 3072), (3072, 1), 'input')
            buf476 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181, to_96, hidden_states_182, linear_126, linear_127], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf474, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_168, (3072, 1024), (1, 3072), 0), out=buf476)
            assert_size_stride(primals_169, (1024, 3072), (3072, 1), 'input')
            buf477 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181, to_96, hidden_states_182, linear_126, linear_128], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf474, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_169, (3072, 1024), (1, 3072), 0), out=buf477)
            buf478 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf754 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_126, view_54, query_states_18, mul_184, x1_36, x2_36, neg_36, cat_37, mul_185, q_embed_18, matmul_37, permute_694], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf475, primals_3, buf478, buf754, 393216, stream=raw_stream0)
            buf479 = reinterpret_tensor(buf475, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf475  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_127, view_55, key_states_36, mul_186, x1_37, x2_37, neg_37, cat_38, mul_187, k_embed_18, getitem_133, hidden_states_183, key_states_37], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf476, primals_3, buf479, 393216, stream=raw_stream0)
            buf480 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_126, view_54, query_states_18, linear_127, view_55, key_states_36, mul_184, x1_36, x2_36, neg_36, cat_37, mul_185, q_embed_18, mul_186, x1_37, x2_37, neg_37, cat_38, mul_187, k_embed_18, getitem_133, hidden_states_183, key_states_37, transpose_94, matmul_37], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf478, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf479, (24, 128, 128), (16384, 1, 128), 0), out=buf480)
            buf481 = reinterpret_tensor(buf480, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf480  # reuse
            buf482 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf483 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf484 = buf478; del buf478  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_37, attn_weights_72, attn_weights_73, softmax_18, attn_weights_74], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf481, buf4, buf482, buf483, buf484, 3072, 128, stream=raw_stream0)
            buf485 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_128, view_56, value_states_36, getitem_134, hidden_states_184, value_states_37], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf477, buf485, 393216, stream=raw_stream0)
            buf486 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_128, view_56, value_states_36, getitem_134, hidden_states_184, value_states_37, softmax_18, attn_weights_74, attn_output_72], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf484, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf485, (24, 128, 128), (16384, 128, 1), 0), out=buf486)
            buf487 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_72, transpose_95, attn_output_73], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf486, buf487, 393216, stream=raw_stream0)
            assert_size_stride(primals_170, (3072, 3072), (3072, 1), 'input')
            buf488 = reinterpret_tensor(buf486, (128, 3072), (3072, 1), 0); del buf486  # reuse
            # Topologically Sorted Source Nodes: [attn_output_72, transpose_95, attn_output_73, reshape_56, attn_output_75], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf487, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_170, (3072, 3072), (1, 3072), 0), out=buf488)
            assert_size_stride(primals_171, (3072, ), (1, ), 'input')
            buf489 = reinterpret_tensor(buf488, (1, 128, 3072), (393216, 3072, 1), 0); del buf488  # reuse
            buf490 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf491 = reinterpret_tensor(buf490, (1, 128, 1), (128, 1, 1), 0); del buf490  # reuse
            buf492 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_75, hidden_states_185, hidden_states_186, pow_38, variance_37, add_131, rsqrt_37, hidden_states_187, to_99, hidden_states_188], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf489, buf491, buf471, primals_171, buf492, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_172, (8192, 3072), (3072, 1), 'input')
            buf493 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_186, hidden_states_187, to_99, hidden_states_188, linear_130], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf492, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_172, (3072, 8192), (1, 3072), 0), out=buf493)
            assert_size_stride(primals_173, (8192, 3072), (3072, 1), 'input')
            buf494 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_186, hidden_states_187, to_99, hidden_states_188, linear_130, linear_131], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf492, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_173, (3072, 8192), (1, 3072), 0), out=buf494)
            buf495 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_191], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf493, buf494, buf495, 1048576, stream=raw_stream0)
            assert_size_stride(primals_174, (3072, 8192), (8192, 1), 'input')
            buf496 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_191, down_proj_18], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf495, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_174, (8192, 3072), (1, 8192), 0), out=buf496)
            assert_size_stride(primals_175, (3072, ), (1, ), 'input')
            buf497 = reinterpret_tensor(buf496, (1, 128, 3072), (393216, 3072, 1), 0); del buf496  # reuse
            buf498 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf499 = reinterpret_tensor(buf498, (1, 128, 1), (128, 1, 1), 0); del buf498  # reuse
            buf500 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_18, hidden_states_189, hidden_states_190, pow_39, variance_38, add_133, rsqrt_38, hidden_states_191, to_101, hidden_states_192], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf497, buf499, buf489, primals_175, buf500, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_176, (3072, 3072), (3072, 1), 'input')
            buf501 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191, to_101, hidden_states_192, linear_133], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf500, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_176, (3072, 3072), (1, 3072), 0), out=buf501)
            assert_size_stride(primals_177, (1024, 3072), (3072, 1), 'input')
            buf502 = buf477; del buf477  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191, to_101, hidden_states_192, linear_133, linear_134], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf500, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_177, (3072, 1024), (1, 3072), 0), out=buf502)
            assert_size_stride(primals_178, (1024, 3072), (3072, 1), 'input')
            buf503 = buf476; del buf476  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191, to_101, hidden_states_192, linear_133, linear_135], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf500, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_178, (3072, 1024), (1, 3072), 0), out=buf503)
            buf504 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf753 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_133, view_57, query_states_19, mul_194, x1_38, x2_38, neg_38, cat_39, mul_195, q_embed_19, matmul_39, permute_657], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf501, primals_3, buf504, buf753, 393216, stream=raw_stream0)
            buf505 = reinterpret_tensor(buf501, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf501  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_134, view_58, key_states_38, mul_196, x1_39, x2_39, neg_39, cat_40, mul_197, k_embed_19, getitem_140, hidden_states_193, key_states_39], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf502, primals_3, buf505, 393216, stream=raw_stream0)
            buf506 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_133, view_57, query_states_19, linear_134, view_58, key_states_38, mul_194, x1_38, x2_38, neg_38, cat_39, mul_195, q_embed_19, mul_196, x1_39, x2_39, neg_39, cat_40, mul_197, k_embed_19, getitem_140, hidden_states_193, key_states_39, transpose_99, matmul_39], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf504, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf505, (24, 128, 128), (16384, 1, 128), 0), out=buf506)
            buf507 = reinterpret_tensor(buf506, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf506  # reuse
            buf508 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf509 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf510 = buf504; del buf504  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_39, attn_weights_76, attn_weights_77, softmax_19, attn_weights_78], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf507, buf4, buf508, buf509, buf510, 3072, 128, stream=raw_stream0)
            buf511 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_135, view_59, value_states_38, getitem_141, hidden_states_194, value_states_39], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf503, buf511, 393216, stream=raw_stream0)
            buf512 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_135, view_59, value_states_38, getitem_141, hidden_states_194, value_states_39, softmax_19, attn_weights_78, attn_output_76], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf510, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf511, (24, 128, 128), (16384, 128, 1), 0), out=buf512)
            buf513 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_76, transpose_100, attn_output_77], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf512, buf513, 393216, stream=raw_stream0)
            assert_size_stride(primals_179, (3072, 3072), (3072, 1), 'input')
            buf514 = reinterpret_tensor(buf512, (128, 3072), (3072, 1), 0); del buf512  # reuse
            # Topologically Sorted Source Nodes: [attn_output_76, transpose_100, attn_output_77, reshape_59, attn_output_79], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf513, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_179, (3072, 3072), (1, 3072), 0), out=buf514)
            assert_size_stride(primals_180, (3072, ), (1, ), 'input')
            buf515 = reinterpret_tensor(buf514, (1, 128, 3072), (393216, 3072, 1), 0); del buf514  # reuse
            buf516 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf517 = reinterpret_tensor(buf516, (1, 128, 1), (128, 1, 1), 0); del buf516  # reuse
            buf518 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_79, hidden_states_195, hidden_states_196, pow_40, variance_39, add_138, rsqrt_39, hidden_states_197, to_104, hidden_states_198], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf515, buf517, buf497, primals_180, buf518, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_181, (8192, 3072), (3072, 1), 'input')
            buf519 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_197, to_104, hidden_states_198, linear_137], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf518, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_181, (3072, 8192), (1, 3072), 0), out=buf519)
            assert_size_stride(primals_182, (8192, 3072), (3072, 1), 'input')
            buf520 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_197, to_104, hidden_states_198, linear_137, linear_138], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf518, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_182, (3072, 8192), (1, 3072), 0), out=buf520)
            buf521 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_201], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf519, buf520, buf521, 1048576, stream=raw_stream0)
            assert_size_stride(primals_183, (3072, 8192), (8192, 1), 'input')
            buf522 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_201, down_proj_19], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf521, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_183, (8192, 3072), (1, 8192), 0), out=buf522)
            assert_size_stride(primals_184, (3072, ), (1, ), 'input')
            buf523 = reinterpret_tensor(buf522, (1, 128, 3072), (393216, 3072, 1), 0); del buf522  # reuse
            buf524 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf525 = reinterpret_tensor(buf524, (1, 128, 1), (128, 1, 1), 0); del buf524  # reuse
            buf526 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, hidden_states_200, pow_41, variance_40, add_140, rsqrt_40, hidden_states_201, to_106, hidden_states_202], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf523, buf525, buf515, primals_184, buf526, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_185, (3072, 3072), (3072, 1), 'input')
            buf527 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_201, to_106, hidden_states_202, linear_140], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_185, (3072, 3072), (1, 3072), 0), out=buf527)
            assert_size_stride(primals_186, (1024, 3072), (3072, 1), 'input')
            buf528 = buf503; del buf503  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_201, to_106, hidden_states_202, linear_140, linear_141], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_186, (3072, 1024), (1, 3072), 0), out=buf528)
            assert_size_stride(primals_187, (1024, 3072), (3072, 1), 'input')
            buf529 = buf502; del buf502  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_201, to_106, hidden_states_202, linear_140, linear_142], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_187, (3072, 1024), (1, 3072), 0), out=buf529)
            buf530 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf752 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_140, view_60, query_states_20, mul_204, x1_40, x2_40, neg_40, cat_41, mul_205, q_embed_20, matmul_41, permute_620], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf527, primals_3, buf530, buf752, 393216, stream=raw_stream0)
            buf531 = reinterpret_tensor(buf527, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf527  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_141, view_61, key_states_40, mul_206, x1_41, x2_41, neg_41, cat_42, mul_207, k_embed_20, getitem_147, hidden_states_203, key_states_41], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf528, primals_3, buf531, 393216, stream=raw_stream0)
            buf532 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_140, view_60, query_states_20, linear_141, view_61, key_states_40, mul_204, x1_40, x2_40, neg_40, cat_41, mul_205, q_embed_20, mul_206, x1_41, x2_41, neg_41, cat_42, mul_207, k_embed_20, getitem_147, hidden_states_203, key_states_41, transpose_104, matmul_41], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf530, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf531, (24, 128, 128), (16384, 1, 128), 0), out=buf532)
            buf533 = reinterpret_tensor(buf532, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf532  # reuse
            buf534 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf535 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf536 = buf530; del buf530  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_41, attn_weights_80, attn_weights_81, softmax_20, attn_weights_82], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf533, buf4, buf534, buf535, buf536, 3072, 128, stream=raw_stream0)
            buf537 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_142, view_62, value_states_40, getitem_148, hidden_states_204, value_states_41], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf529, buf537, 393216, stream=raw_stream0)
            buf538 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_142, view_62, value_states_40, getitem_148, hidden_states_204, value_states_41, softmax_20, attn_weights_82, attn_output_80], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf536, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf537, (24, 128, 128), (16384, 128, 1), 0), out=buf538)
            buf539 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_80, transpose_105, attn_output_81], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf538, buf539, 393216, stream=raw_stream0)
            assert_size_stride(primals_188, (3072, 3072), (3072, 1), 'input')
            buf540 = reinterpret_tensor(buf538, (128, 3072), (3072, 1), 0); del buf538  # reuse
            # Topologically Sorted Source Nodes: [attn_output_80, transpose_105, attn_output_81, reshape_62, attn_output_83], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf539, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_188, (3072, 3072), (1, 3072), 0), out=buf540)
            assert_size_stride(primals_189, (3072, ), (1, ), 'input')
            buf541 = reinterpret_tensor(buf540, (1, 128, 3072), (393216, 3072, 1), 0); del buf540  # reuse
            buf542 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf543 = reinterpret_tensor(buf542, (1, 128, 1), (128, 1, 1), 0); del buf542  # reuse
            buf544 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_83, hidden_states_205, hidden_states_206, pow_42, variance_41, add_145, rsqrt_41, hidden_states_207, to_109, hidden_states_208], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf541, buf543, buf523, primals_189, buf544, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_190, (8192, 3072), (3072, 1), 'input')
            buf545 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_206, hidden_states_207, to_109, hidden_states_208, linear_144], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf544, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_190, (3072, 8192), (1, 3072), 0), out=buf545)
            assert_size_stride(primals_191, (8192, 3072), (3072, 1), 'input')
            buf546 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_206, hidden_states_207, to_109, hidden_states_208, linear_144, linear_145], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf544, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_191, (3072, 8192), (1, 3072), 0), out=buf546)
            buf547 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_211], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf545, buf546, buf547, 1048576, stream=raw_stream0)
            assert_size_stride(primals_192, (3072, 8192), (8192, 1), 'input')
            buf548 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_211, down_proj_20], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf547, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_192, (8192, 3072), (1, 8192), 0), out=buf548)
            assert_size_stride(primals_193, (3072, ), (1, ), 'input')
            buf549 = reinterpret_tensor(buf548, (1, 128, 3072), (393216, 3072, 1), 0); del buf548  # reuse
            buf550 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf551 = reinterpret_tensor(buf550, (1, 128, 1), (128, 1, 1), 0); del buf550  # reuse
            buf552 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_20, hidden_states_209, hidden_states_210, pow_43, variance_42, add_147, rsqrt_42, hidden_states_211, to_111, hidden_states_212], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf549, buf551, buf541, primals_193, buf552, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_194, (3072, 3072), (3072, 1), 'input')
            buf553 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211, to_111, hidden_states_212, linear_147], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf552, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_194, (3072, 3072), (1, 3072), 0), out=buf553)
            assert_size_stride(primals_195, (1024, 3072), (3072, 1), 'input')
            buf554 = buf529; del buf529  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211, to_111, hidden_states_212, linear_147, linear_148], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf552, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_195, (3072, 1024), (1, 3072), 0), out=buf554)
            assert_size_stride(primals_196, (1024, 3072), (3072, 1), 'input')
            buf555 = buf528; del buf528  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211, to_111, hidden_states_212, linear_147, linear_149], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf552, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_196, (3072, 1024), (1, 3072), 0), out=buf555)
            buf556 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf751 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_147, view_63, query_states_21, mul_214, x1_42, x2_42, neg_42, cat_43, mul_215, q_embed_21, matmul_43, permute_583], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf553, primals_3, buf556, buf751, 393216, stream=raw_stream0)
            buf557 = reinterpret_tensor(buf553, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf553  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_148, view_64, key_states_42, mul_216, x1_43, x2_43, neg_43, cat_44, mul_217, k_embed_21, getitem_154, hidden_states_213, key_states_43], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf554, primals_3, buf557, 393216, stream=raw_stream0)
            buf558 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_147, view_63, query_states_21, linear_148, view_64, key_states_42, mul_214, x1_42, x2_42, neg_42, cat_43, mul_215, q_embed_21, mul_216, x1_43, x2_43, neg_43, cat_44, mul_217, k_embed_21, getitem_154, hidden_states_213, key_states_43, transpose_109, matmul_43], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf556, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf557, (24, 128, 128), (16384, 1, 128), 0), out=buf558)
            buf559 = reinterpret_tensor(buf558, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf558  # reuse
            buf560 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf561 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf562 = buf556; del buf556  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_43, attn_weights_84, attn_weights_85, softmax_21, attn_weights_86], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf559, buf4, buf560, buf561, buf562, 3072, 128, stream=raw_stream0)
            buf563 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_149, view_65, value_states_42, getitem_155, hidden_states_214, value_states_43], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf555, buf563, 393216, stream=raw_stream0)
            buf564 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_149, view_65, value_states_42, getitem_155, hidden_states_214, value_states_43, softmax_21, attn_weights_86, attn_output_84], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf562, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf563, (24, 128, 128), (16384, 128, 1), 0), out=buf564)
            buf565 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_84, transpose_110, attn_output_85], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf564, buf565, 393216, stream=raw_stream0)
            assert_size_stride(primals_197, (3072, 3072), (3072, 1), 'input')
            buf566 = reinterpret_tensor(buf564, (128, 3072), (3072, 1), 0); del buf564  # reuse
            # Topologically Sorted Source Nodes: [attn_output_84, transpose_110, attn_output_85, reshape_65, attn_output_87], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf565, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_197, (3072, 3072), (1, 3072), 0), out=buf566)
            assert_size_stride(primals_198, (3072, ), (1, ), 'input')
            buf567 = reinterpret_tensor(buf566, (1, 128, 3072), (393216, 3072, 1), 0); del buf566  # reuse
            buf568 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf569 = reinterpret_tensor(buf568, (1, 128, 1), (128, 1, 1), 0); del buf568  # reuse
            buf570 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_87, hidden_states_215, hidden_states_216, pow_44, variance_43, add_152, rsqrt_43, hidden_states_217, to_114, hidden_states_218], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf567, buf569, buf549, primals_198, buf570, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_199, (8192, 3072), (3072, 1), 'input')
            buf571 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_216, hidden_states_217, to_114, hidden_states_218, linear_151], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf570, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_199, (3072, 8192), (1, 3072), 0), out=buf571)
            assert_size_stride(primals_200, (8192, 3072), (3072, 1), 'input')
            buf572 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_216, hidden_states_217, to_114, hidden_states_218, linear_151, linear_152], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf570, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_200, (3072, 8192), (1, 3072), 0), out=buf572)
            buf573 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_221], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf571, buf572, buf573, 1048576, stream=raw_stream0)
            assert_size_stride(primals_201, (3072, 8192), (8192, 1), 'input')
            buf574 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_221, down_proj_21], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf573, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_201, (8192, 3072), (1, 8192), 0), out=buf574)
            assert_size_stride(primals_202, (3072, ), (1, ), 'input')
            buf575 = reinterpret_tensor(buf574, (1, 128, 3072), (393216, 3072, 1), 0); del buf574  # reuse
            buf576 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf577 = reinterpret_tensor(buf576, (1, 128, 1), (128, 1, 1), 0); del buf576  # reuse
            buf578 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, hidden_states_220, pow_45, variance_44, add_154, rsqrt_44, hidden_states_221, to_116, hidden_states_222], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf575, buf577, buf567, primals_202, buf578, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_203, (3072, 3072), (3072, 1), 'input')
            buf579 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221, to_116, hidden_states_222, linear_154], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf578, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_203, (3072, 3072), (1, 3072), 0), out=buf579)
            assert_size_stride(primals_204, (1024, 3072), (3072, 1), 'input')
            buf580 = buf555; del buf555  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221, to_116, hidden_states_222, linear_154, linear_155], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf578, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_204, (3072, 1024), (1, 3072), 0), out=buf580)
            assert_size_stride(primals_205, (1024, 3072), (3072, 1), 'input')
            buf581 = buf554; del buf554  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221, to_116, hidden_states_222, linear_154, linear_156], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf578, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_205, (3072, 1024), (1, 3072), 0), out=buf581)
            buf582 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf750 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_154, view_66, query_states_22, mul_224, x1_44, x2_44, neg_44, cat_45, mul_225, q_embed_22, matmul_45, permute_546], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf579, primals_3, buf582, buf750, 393216, stream=raw_stream0)
            buf583 = reinterpret_tensor(buf579, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf579  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_155, view_67, key_states_44, mul_226, x1_45, x2_45, neg_45, cat_46, mul_227, k_embed_22, getitem_161, hidden_states_223, key_states_45], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf580, primals_3, buf583, 393216, stream=raw_stream0)
            buf584 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_154, view_66, query_states_22, linear_155, view_67, key_states_44, mul_224, x1_44, x2_44, neg_44, cat_45, mul_225, q_embed_22, mul_226, x1_45, x2_45, neg_45, cat_46, mul_227, k_embed_22, getitem_161, hidden_states_223, key_states_45, transpose_114, matmul_45], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf582, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf583, (24, 128, 128), (16384, 1, 128), 0), out=buf584)
            buf585 = reinterpret_tensor(buf584, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf584  # reuse
            buf586 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf587 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf588 = buf582; del buf582  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_45, attn_weights_88, attn_weights_89, softmax_22, attn_weights_90], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf585, buf4, buf586, buf587, buf588, 3072, 128, stream=raw_stream0)
            buf589 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_156, view_68, value_states_44, getitem_162, hidden_states_224, value_states_45], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf581, buf589, 393216, stream=raw_stream0)
            buf590 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_156, view_68, value_states_44, getitem_162, hidden_states_224, value_states_45, softmax_22, attn_weights_90, attn_output_88], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf588, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf589, (24, 128, 128), (16384, 128, 1), 0), out=buf590)
            buf591 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_88, transpose_115, attn_output_89], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf590, buf591, 393216, stream=raw_stream0)
            assert_size_stride(primals_206, (3072, 3072), (3072, 1), 'input')
            buf592 = reinterpret_tensor(buf590, (128, 3072), (3072, 1), 0); del buf590  # reuse
            # Topologically Sorted Source Nodes: [attn_output_88, transpose_115, attn_output_89, reshape_68, attn_output_91], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf591, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_206, (3072, 3072), (1, 3072), 0), out=buf592)
            assert_size_stride(primals_207, (3072, ), (1, ), 'input')
            buf593 = reinterpret_tensor(buf592, (1, 128, 3072), (393216, 3072, 1), 0); del buf592  # reuse
            buf594 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf595 = reinterpret_tensor(buf594, (1, 128, 1), (128, 1, 1), 0); del buf594  # reuse
            buf596 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_91, hidden_states_225, hidden_states_226, pow_46, variance_45, add_159, rsqrt_45, hidden_states_227, to_119, hidden_states_228], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf593, buf595, buf575, primals_207, buf596, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_208, (8192, 3072), (3072, 1), 'input')
            buf597 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_227, to_119, hidden_states_228, linear_158], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf596, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_208, (3072, 8192), (1, 3072), 0), out=buf597)
            assert_size_stride(primals_209, (8192, 3072), (3072, 1), 'input')
            buf598 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_227, to_119, hidden_states_228, linear_158, linear_159], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf596, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_209, (3072, 8192), (1, 3072), 0), out=buf598)
            buf599 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_231], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf597, buf598, buf599, 1048576, stream=raw_stream0)
            assert_size_stride(primals_210, (3072, 8192), (8192, 1), 'input')
            buf600 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_231, down_proj_22], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf599, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_210, (8192, 3072), (1, 8192), 0), out=buf600)
            assert_size_stride(primals_211, (3072, ), (1, ), 'input')
            buf601 = reinterpret_tensor(buf600, (1, 128, 3072), (393216, 3072, 1), 0); del buf600  # reuse
            buf602 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf603 = reinterpret_tensor(buf602, (1, 128, 1), (128, 1, 1), 0); del buf602  # reuse
            buf604 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_22, hidden_states_229, hidden_states_230, pow_47, variance_46, add_161, rsqrt_46, hidden_states_231, to_121, hidden_states_232], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf601, buf603, buf593, primals_211, buf604, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_212, (3072, 3072), (3072, 1), 'input')
            buf605 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_230, hidden_states_231, to_121, hidden_states_232, linear_161], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf604, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_212, (3072, 3072), (1, 3072), 0), out=buf605)
            assert_size_stride(primals_213, (1024, 3072), (3072, 1), 'input')
            buf606 = buf581; del buf581  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_230, hidden_states_231, to_121, hidden_states_232, linear_161, linear_162], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf604, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_213, (3072, 1024), (1, 3072), 0), out=buf606)
            assert_size_stride(primals_214, (1024, 3072), (3072, 1), 'input')
            buf607 = buf580; del buf580  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_230, hidden_states_231, to_121, hidden_states_232, linear_161, linear_163], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf604, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_214, (3072, 1024), (1, 3072), 0), out=buf607)
            buf608 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf749 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_161, view_69, query_states_23, mul_234, x1_46, x2_46, neg_46, cat_47, mul_235, q_embed_23, matmul_47, permute_509], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf605, primals_3, buf608, buf749, 393216, stream=raw_stream0)
            buf609 = reinterpret_tensor(buf605, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf605  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_162, view_70, key_states_46, mul_236, x1_47, x2_47, neg_47, cat_48, mul_237, k_embed_23, getitem_168, hidden_states_233, key_states_47], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf606, primals_3, buf609, 393216, stream=raw_stream0)
            buf610 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_161, view_69, query_states_23, linear_162, view_70, key_states_46, mul_234, x1_46, x2_46, neg_46, cat_47, mul_235, q_embed_23, mul_236, x1_47, x2_47, neg_47, cat_48, mul_237, k_embed_23, getitem_168, hidden_states_233, key_states_47, transpose_119, matmul_47], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf608, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf609, (24, 128, 128), (16384, 1, 128), 0), out=buf610)
            buf611 = reinterpret_tensor(buf610, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf610  # reuse
            buf612 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf613 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf614 = buf608; del buf608  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_47, attn_weights_92, attn_weights_93, softmax_23, attn_weights_94], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf611, buf4, buf612, buf613, buf614, 3072, 128, stream=raw_stream0)
            buf615 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_163, view_71, value_states_46, getitem_169, hidden_states_234, value_states_47], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf607, buf615, 393216, stream=raw_stream0)
            buf616 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_163, view_71, value_states_46, getitem_169, hidden_states_234, value_states_47, softmax_23, attn_weights_94, attn_output_92], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf614, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf615, (24, 128, 128), (16384, 128, 1), 0), out=buf616)
            buf617 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_92, transpose_120, attn_output_93], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf616, buf617, 393216, stream=raw_stream0)
            assert_size_stride(primals_215, (3072, 3072), (3072, 1), 'input')
            buf618 = reinterpret_tensor(buf616, (128, 3072), (3072, 1), 0); del buf616  # reuse
            # Topologically Sorted Source Nodes: [attn_output_92, transpose_120, attn_output_93, reshape_71, attn_output_95], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf617, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_215, (3072, 3072), (1, 3072), 0), out=buf618)
            assert_size_stride(primals_216, (3072, ), (1, ), 'input')
            buf619 = reinterpret_tensor(buf618, (1, 128, 3072), (393216, 3072, 1), 0); del buf618  # reuse
            buf620 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf621 = reinterpret_tensor(buf620, (1, 128, 1), (128, 1, 1), 0); del buf620  # reuse
            buf622 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_95, hidden_states_235, hidden_states_236, pow_48, variance_47, add_166, rsqrt_47, hidden_states_237, to_124, hidden_states_238], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf619, buf621, buf601, primals_216, buf622, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_217, (8192, 3072), (3072, 1), 'input')
            buf623 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_236, hidden_states_237, to_124, hidden_states_238, linear_165], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf622, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_217, (3072, 8192), (1, 3072), 0), out=buf623)
            assert_size_stride(primals_218, (8192, 3072), (3072, 1), 'input')
            buf624 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_236, hidden_states_237, to_124, hidden_states_238, linear_165, linear_166], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf622, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_218, (3072, 8192), (1, 3072), 0), out=buf624)
            buf625 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_241], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf623, buf624, buf625, 1048576, stream=raw_stream0)
            assert_size_stride(primals_219, (3072, 8192), (8192, 1), 'input')
            buf626 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_241, down_proj_23], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf625, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_219, (8192, 3072), (1, 8192), 0), out=buf626)
            assert_size_stride(primals_220, (3072, ), (1, ), 'input')
            buf627 = reinterpret_tensor(buf626, (1, 128, 3072), (393216, 3072, 1), 0); del buf626  # reuse
            buf628 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf629 = reinterpret_tensor(buf628, (1, 128, 1), (128, 1, 1), 0); del buf628  # reuse
            buf630 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, hidden_states_240, pow_49, variance_48, add_168, rsqrt_48, hidden_states_241, to_126, hidden_states_242], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf627, buf629, buf619, primals_220, buf630, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_221, (3072, 3072), (3072, 1), 'input')
            buf631 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_240, hidden_states_241, to_126, hidden_states_242, linear_168], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf630, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_221, (3072, 3072), (1, 3072), 0), out=buf631)
            assert_size_stride(primals_222, (1024, 3072), (3072, 1), 'input')
            buf632 = buf607; del buf607  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_240, hidden_states_241, to_126, hidden_states_242, linear_168, linear_169], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf630, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_222, (3072, 1024), (1, 3072), 0), out=buf632)
            assert_size_stride(primals_223, (1024, 3072), (3072, 1), 'input')
            buf633 = buf606; del buf606  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_240, hidden_states_241, to_126, hidden_states_242, linear_168, linear_170], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf630, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_223, (3072, 1024), (1, 3072), 0), out=buf633)
            buf634 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf748 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_168, view_72, query_states_24, mul_244, x1_48, x2_48, neg_48, cat_49, mul_245, q_embed_24, matmul_49, permute_472], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf631, primals_3, buf634, buf748, 393216, stream=raw_stream0)
            buf635 = reinterpret_tensor(buf631, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf631  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_169, view_73, key_states_48, mul_246, x1_49, x2_49, neg_49, cat_50, mul_247, k_embed_24, getitem_175, hidden_states_243, key_states_49], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf632, primals_3, buf635, 393216, stream=raw_stream0)
            buf636 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_168, view_72, query_states_24, linear_169, view_73, key_states_48, mul_244, x1_48, x2_48, neg_48, cat_49, mul_245, q_embed_24, mul_246, x1_49, x2_49, neg_49, cat_50, mul_247, k_embed_24, getitem_175, hidden_states_243, key_states_49, transpose_124, matmul_49], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf634, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf635, (24, 128, 128), (16384, 1, 128), 0), out=buf636)
            buf637 = reinterpret_tensor(buf636, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf636  # reuse
            buf638 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf639 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf640 = buf634; del buf634  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_49, attn_weights_96, attn_weights_97, softmax_24, attn_weights_98], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf637, buf4, buf638, buf639, buf640, 3072, 128, stream=raw_stream0)
            buf641 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_170, view_74, value_states_48, getitem_176, hidden_states_244, value_states_49], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf633, buf641, 393216, stream=raw_stream0)
            buf642 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_170, view_74, value_states_48, getitem_176, hidden_states_244, value_states_49, softmax_24, attn_weights_98, attn_output_96], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf640, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf641, (24, 128, 128), (16384, 128, 1), 0), out=buf642)
            buf643 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_96, transpose_125, attn_output_97], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf642, buf643, 393216, stream=raw_stream0)
            assert_size_stride(primals_224, (3072, 3072), (3072, 1), 'input')
            buf644 = reinterpret_tensor(buf642, (128, 3072), (3072, 1), 0); del buf642  # reuse
            # Topologically Sorted Source Nodes: [attn_output_96, transpose_125, attn_output_97, reshape_74, attn_output_99], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf643, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_224, (3072, 3072), (1, 3072), 0), out=buf644)
            assert_size_stride(primals_225, (3072, ), (1, ), 'input')
            buf645 = reinterpret_tensor(buf644, (1, 128, 3072), (393216, 3072, 1), 0); del buf644  # reuse
            buf646 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf647 = reinterpret_tensor(buf646, (1, 128, 1), (128, 1, 1), 0); del buf646  # reuse
            buf648 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_99, hidden_states_245, hidden_states_246, pow_50, variance_49, add_173, rsqrt_49, hidden_states_247, to_129, hidden_states_248], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf645, buf647, buf627, primals_225, buf648, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_226, (8192, 3072), (3072, 1), 'input')
            buf649 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_246, hidden_states_247, to_129, hidden_states_248, linear_172], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf648, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_226, (3072, 8192), (1, 3072), 0), out=buf649)
            assert_size_stride(primals_227, (8192, 3072), (3072, 1), 'input')
            buf650 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_246, hidden_states_247, to_129, hidden_states_248, linear_172, linear_173], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf648, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_227, (3072, 8192), (1, 3072), 0), out=buf650)
            buf651 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_251], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf649, buf650, buf651, 1048576, stream=raw_stream0)
            assert_size_stride(primals_228, (3072, 8192), (8192, 1), 'input')
            buf652 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_251, down_proj_24], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf651, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_228, (8192, 3072), (1, 8192), 0), out=buf652)
            assert_size_stride(primals_229, (3072, ), (1, ), 'input')
            buf653 = reinterpret_tensor(buf652, (1, 128, 3072), (393216, 3072, 1), 0); del buf652  # reuse
            buf654 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf655 = reinterpret_tensor(buf654, (1, 128, 1), (128, 1, 1), 0); del buf654  # reuse
            buf656 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_24, hidden_states_249, hidden_states_250, pow_51, variance_50, add_175, rsqrt_50, hidden_states_251, to_131, hidden_states_252], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf653, buf655, buf645, primals_229, buf656, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_230, (3072, 3072), (3072, 1), 'input')
            buf657 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_251, to_131, hidden_states_252, linear_175], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf656, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_230, (3072, 3072), (1, 3072), 0), out=buf657)
            assert_size_stride(primals_231, (1024, 3072), (3072, 1), 'input')
            buf658 = buf633; del buf633  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_251, to_131, hidden_states_252, linear_175, linear_176], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf656, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_231, (3072, 1024), (1, 3072), 0), out=buf658)
            assert_size_stride(primals_232, (1024, 3072), (3072, 1), 'input')
            buf659 = buf632; del buf632  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_250, hidden_states_251, to_131, hidden_states_252, linear_175, linear_177], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf656, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_232, (3072, 1024), (1, 3072), 0), out=buf659)
            buf660 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf747 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_175, view_75, query_states_25, mul_254, x1_50, x2_50, neg_50, cat_51, mul_255, q_embed_25, matmul_51, permute_435], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf657, primals_3, buf660, buf747, 393216, stream=raw_stream0)
            buf661 = reinterpret_tensor(buf657, (1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), 0); del buf657  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_176, view_76, key_states_50, mul_256, x1_51, x2_51, neg_51, cat_52, mul_257, k_embed_25, getitem_182, hidden_states_253, key_states_51], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_5.run(buf658, primals_3, buf661, 393216, stream=raw_stream0)
            buf662 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_175, view_75, query_states_25, linear_176, view_76, key_states_50, mul_254, x1_50, x2_50, neg_50, cat_51, mul_255, q_embed_25, mul_256, x1_51, x2_51, neg_51, cat_52, mul_257, k_embed_25, getitem_182, hidden_states_253, key_states_51, transpose_129, matmul_51], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf660, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf661, (24, 128, 128), (16384, 1, 128), 0), out=buf662)
            buf663 = reinterpret_tensor(buf662, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf662  # reuse
            buf664 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf665 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf666 = buf660; del buf660  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_51, attn_weights_100, attn_weights_101, softmax_25, attn_weights_102], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf663, buf4, buf664, buf665, buf666, 3072, 128, stream=raw_stream0)
            buf667 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_177, view_77, value_states_50, getitem_183, hidden_states_254, value_states_51], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf659, buf667, 393216, stream=raw_stream0)
            buf668 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_177, view_77, value_states_50, getitem_183, hidden_states_254, value_states_51, softmax_25, attn_weights_102, attn_output_100], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf666, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf667, (24, 128, 128), (16384, 128, 1), 0), out=buf668)
            buf669 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_100, transpose_130, attn_output_101], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf668, buf669, 393216, stream=raw_stream0)
            assert_size_stride(primals_233, (3072, 3072), (3072, 1), 'input')
            buf670 = reinterpret_tensor(buf668, (128, 3072), (3072, 1), 0); del buf668  # reuse
            # Topologically Sorted Source Nodes: [attn_output_100, transpose_130, attn_output_101, reshape_77, attn_output_103], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf669, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_233, (3072, 3072), (1, 3072), 0), out=buf670)
            assert_size_stride(primals_234, (3072, ), (1, ), 'input')
            buf671 = reinterpret_tensor(buf670, (1, 128, 3072), (393216, 3072, 1), 0); del buf670  # reuse
            buf672 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf673 = reinterpret_tensor(buf672, (1, 128, 1), (128, 1, 1), 0); del buf672  # reuse
            buf674 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_103, hidden_states_255, hidden_states_256, pow_52, variance_51, add_180, rsqrt_51, hidden_states_257, to_134, hidden_states_258], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf671, buf673, buf653, primals_234, buf674, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_235, (8192, 3072), (3072, 1), 'input')
            buf675 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_256, hidden_states_257, to_134, hidden_states_258, linear_179], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf674, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_235, (3072, 8192), (1, 3072), 0), out=buf675)
            assert_size_stride(primals_236, (8192, 3072), (3072, 1), 'input')
            buf676 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_256, hidden_states_257, to_134, hidden_states_258, linear_179, linear_180], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf674, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_236, (3072, 8192), (1, 3072), 0), out=buf676)
            buf677 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_261], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf675, buf676, buf677, 1048576, stream=raw_stream0)
            assert_size_stride(primals_237, (3072, 8192), (8192, 1), 'input')
            buf678 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_261, down_proj_25], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf677, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_237, (8192, 3072), (1, 8192), 0), out=buf678)
            assert_size_stride(primals_238, (3072, ), (1, ), 'input')
            buf679 = reinterpret_tensor(buf678, (1, 128, 3072), (393216, 3072, 1), 0); del buf678  # reuse
            buf680 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf681 = reinterpret_tensor(buf680, (1, 128, 1), (128, 1, 1), 0); del buf680  # reuse
            buf682 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, hidden_states_260, pow_53, variance_52, add_182, rsqrt_52, hidden_states_261, to_136, hidden_states_262], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf679, buf681, buf671, primals_238, buf682, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_239, (3072, 3072), (3072, 1), 'input')
            buf683 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_260, hidden_states_261, to_136, hidden_states_262, linear_182], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf682, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_239, (3072, 3072), (1, 3072), 0), out=buf683)
            assert_size_stride(primals_240, (1024, 3072), (3072, 1), 'input')
            buf684 = buf659; del buf659  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_260, hidden_states_261, to_136, hidden_states_262, linear_182, linear_183], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf682, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_240, (3072, 1024), (1, 3072), 0), out=buf684)
            assert_size_stride(primals_241, (1024, 3072), (3072, 1), 'input')
            buf685 = buf658; del buf658  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_260, hidden_states_261, to_136, hidden_states_262, linear_182, linear_184], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf682, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_241, (3072, 1024), (1, 3072), 0), out=buf685)
            buf686 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf746 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            buf687 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_182, view_78, query_states_26, linear_183, view_79, key_states_52, mul_264, x1_52, x2_52, neg_52, cat_53, mul_265, q_embed_26, mul_266, x1_53, x2_53, neg_53, cat_54, mul_267, k_embed_26, getitem_189, hidden_states_263, key_states_53, matmul_53, permute_398], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_13.run(buf683, primals_3, buf684, buf686, buf746, buf687, 393216, stream=raw_stream0)
            buf688 = reinterpret_tensor(buf683, (24, 128, 128), (16384, 128, 1), 0); del buf683  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_182, view_78, query_states_26, linear_183, view_79, key_states_52, mul_264, x1_52, x2_52, neg_52, cat_53, mul_265, q_embed_26, mul_266, x1_53, x2_53, neg_53, cat_54, mul_267, k_embed_26, getitem_189, hidden_states_263, key_states_53, transpose_134, matmul_53], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf686, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf687, (24, 128, 128), (16384, 1, 128), 0), out=buf688)
            buf689 = reinterpret_tensor(buf688, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf688  # reuse
            buf690 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf691 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf692 = buf686; del buf686  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_53, attn_weights_104, attn_weights_105, softmax_26, attn_weights_106], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf689, buf4, buf690, buf691, buf692, 3072, 128, stream=raw_stream0)
            buf693 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_184, view_80, value_states_52, getitem_190, hidden_states_264, value_states_53], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf685, buf693, 393216, stream=raw_stream0)
            buf694 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_184, view_80, value_states_52, getitem_190, hidden_states_264, value_states_53, softmax_26, attn_weights_106, attn_output_104], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf692, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf693, (24, 128, 128), (16384, 128, 1), 0), out=buf694)
            buf695 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_104, transpose_135, attn_output_105], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf694, buf695, 393216, stream=raw_stream0)
            assert_size_stride(primals_242, (3072, 3072), (3072, 1), 'input')
            buf696 = reinterpret_tensor(buf694, (128, 3072), (3072, 1), 0); del buf694  # reuse
            # Topologically Sorted Source Nodes: [attn_output_104, transpose_135, attn_output_105, reshape_80, attn_output_107], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf695, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_242, (3072, 3072), (1, 3072), 0), out=buf696)
            assert_size_stride(primals_243, (3072, ), (1, ), 'input')
            buf697 = reinterpret_tensor(buf696, (1, 128, 3072), (393216, 3072, 1), 0); del buf696  # reuse
            buf698 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf699 = reinterpret_tensor(buf698, (1, 128, 1), (128, 1, 1), 0); del buf698  # reuse
            buf700 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_107, hidden_states_265, hidden_states_266, pow_54, variance_53, add_187, rsqrt_53, hidden_states_267, to_139, hidden_states_268], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf697, buf699, buf679, primals_243, buf700, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_244, (8192, 3072), (3072, 1), 'input')
            buf701 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_266, hidden_states_267, to_139, hidden_states_268, linear_186], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf700, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_244, (3072, 8192), (1, 3072), 0), out=buf701)
            assert_size_stride(primals_245, (8192, 3072), (3072, 1), 'input')
            buf702 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_266, hidden_states_267, to_139, hidden_states_268, linear_186, linear_187], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf700, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_245, (3072, 8192), (1, 3072), 0), out=buf702)
            buf703 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_271], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf701, buf702, buf703, 1048576, stream=raw_stream0)
            assert_size_stride(primals_246, (3072, 8192), (8192, 1), 'input')
            buf704 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_271, down_proj_26], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf703, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_246, (8192, 3072), (1, 8192), 0), out=buf704)
            assert_size_stride(primals_247, (3072, ), (1, ), 'input')
            buf705 = reinterpret_tensor(buf704, (1, 128, 3072), (393216, 3072, 1), 0); del buf704  # reuse
            buf706 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf707 = reinterpret_tensor(buf706, (1, 128, 1), (128, 1, 1), 0); del buf706  # reuse
            buf708 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_26, hidden_states_269, hidden_states_270, pow_55, variance_54, add_189, rsqrt_54, hidden_states_271, to_141, hidden_states_272], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf705, buf707, buf697, primals_247, buf708, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_248, (3072, 3072), (3072, 1), 'input')
            buf709 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_270, hidden_states_271, to_141, hidden_states_272, linear_189], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf708, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_248, (3072, 3072), (1, 3072), 0), out=buf709)
            assert_size_stride(primals_249, (1024, 3072), (3072, 1), 'input')
            buf710 = buf685; del buf685  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_270, hidden_states_271, to_141, hidden_states_272, linear_189, linear_190], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf708, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_249, (3072, 1024), (1, 3072), 0), out=buf710)
            assert_size_stride(primals_250, (1024, 3072), (3072, 1), 'input')
            buf711 = buf684; del buf684  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_270, hidden_states_271, to_141, hidden_states_272, linear_189, linear_191], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf708, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_250, (3072, 1024), (1, 3072), 0), out=buf711)
            buf712 = empty_strided_cuda((1, 24, 128, 128), (393216, 16384, 128, 1), torch.bfloat16)
            buf745 = empty_strided_cuda((24, 128, 128), (128, 1, 3072), torch.bfloat16)
            buf713 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_189, view_81, query_states_27, linear_190, view_82, key_states_54, mul_274, x1_54, x2_54, neg_54, cat_55, mul_275, q_embed_27, mul_276, x1_55, x2_55, neg_55, cat_56, mul_277, k_embed_27, getitem_196, hidden_states_273, key_states_55, matmul_55, permute_361], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_13.run(buf709, primals_3, buf710, buf712, buf745, buf713, 393216, stream=raw_stream0)
            del buf710
            buf714 = reinterpret_tensor(buf709, (24, 128, 128), (16384, 128, 1), 0); del buf709  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_189, view_81, query_states_27, linear_190, view_82, key_states_54, mul_274, x1_54, x2_54, neg_54, cat_55, mul_275, q_embed_27, mul_276, x1_55, x2_55, neg_55, cat_56, mul_277, k_embed_27, getitem_196, hidden_states_273, key_states_55, transpose_139, matmul_55], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf712, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf713, (24, 128, 128), (16384, 1, 128), 0), out=buf714)
            buf715 = reinterpret_tensor(buf714, (1, 24, 128, 128), (393216, 16384, 128, 1), 0); del buf714  # reuse
            buf716 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf717 = empty_strided_cuda((1, 24, 128, 1), (3072, 128, 1, 1), torch.float32)
            buf718 = buf712; del buf712  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_55, attn_weights_108, attn_weights_109, softmax_27, attn_weights_110], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_6.run(buf715, buf4, buf716, buf717, buf718, 3072, 128, stream=raw_stream0)
            del buf4
            buf719 = empty_strided_cuda((1, 8, 3, 128, 128), (393216, 49152, 16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_191, view_83, value_states_54, getitem_197, hidden_states_274, value_states_55], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_7.run(buf711, buf719, 393216, stream=raw_stream0)
            del buf711
            buf720 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_191, view_83, value_states_54, getitem_197, hidden_states_274, value_states_55, softmax_27, attn_weights_110, attn_output_108], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf718, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf719, (24, 128, 128), (16384, 128, 1), 0), out=buf720)
            buf721 = empty_strided_cuda((1, 128, 24, 128), (393216, 3072, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_108, transpose_140, attn_output_109], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_8.run(buf720, buf721, 393216, stream=raw_stream0)
            assert_size_stride(primals_251, (3072, 3072), (3072, 1), 'input')
            buf722 = reinterpret_tensor(buf720, (128, 3072), (3072, 1), 0); del buf720  # reuse
            # Topologically Sorted Source Nodes: [attn_output_108, transpose_140, attn_output_109, reshape_83, attn_output_111], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf721, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_251, (3072, 3072), (1, 3072), 0), out=buf722)
            assert_size_stride(primals_252, (3072, ), (1, ), 'input')
            buf723 = reinterpret_tensor(buf722, (1, 128, 3072), (393216, 3072, 1), 0); del buf722  # reuse
            buf724 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf725 = reinterpret_tensor(buf724, (1, 128, 1), (128, 1, 1), 0); del buf724  # reuse
            buf726 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [attn_output_111, hidden_states_275, hidden_states_276, pow_56, variance_55, add_194, rsqrt_55, hidden_states_277, to_144, hidden_states_278], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf723, buf725, buf705, primals_252, buf726, 128, 3072, stream=raw_stream0)
            assert_size_stride(primals_253, (8192, 3072), (3072, 1), 'input')
            buf727 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_276, hidden_states_277, to_144, hidden_states_278, linear_193], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf726, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_253, (3072, 8192), (1, 3072), 0), out=buf727)
            assert_size_stride(primals_254, (8192, 3072), (3072, 1), 'input')
            buf728 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_276, hidden_states_277, to_144, hidden_states_278, linear_193, linear_194], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf726, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_254, (3072, 8192), (1, 3072), 0), out=buf728)
            buf729 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_281], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_10.run(buf727, buf728, buf729, 1048576, stream=raw_stream0)
            assert_size_stride(primals_255, (3072, 8192), (8192, 1), 'input')
            buf730 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_281, down_proj_27], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf729, (128, 8192), (8192, 1), 0), reinterpret_tensor(primals_255, (8192, 3072), (1, 8192), 0), out=buf730)
            assert_size_stride(primals_256, (3072, ), (1, ), 'input')
            buf731 = reinterpret_tensor(buf730, (1, 128, 3072), (393216, 3072, 1), 0); del buf730  # reuse
            buf732 = empty_strided_cuda((1, 128, 1), (128, 1, 128), torch.float32)
            buf733 = reinterpret_tensor(buf732, (1, 128, 1), (128, 1, 1), 0); del buf732  # reuse
            buf734 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_27, hidden_states_279, hidden_states_280, pow_57, variance_56, add_196, rsqrt_56, hidden_states_281, to_146, hidden_states_282], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf731, buf733, buf723, primals_256, buf734, 128, 3072, stream=raw_stream0)
            buf735 = empty_strided_cuda((128, 128256), (128256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_280, hidden_states_281, to_146, hidden_states_282, logits], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf734, (128, 3072), (3072, 1), 0), reinterpret_tensor(primals_2, (3072, 128256), (1, 3072), 0), out=buf735)
            buf736 = buf3; del buf3  # reuse
            # Topologically Sorted Source Nodes: [labels], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_14.run(primals_1, buf736, 129, stream=raw_stream0)
            buf737 = empty_strided_cuda((128, 1, 2), (2, 256, 1), torch.float32)
            # Topologically Sorted Source Nodes: [logits, logits_1, logits_2], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_15.run(buf735, buf737, 256, 64128, stream=raw_stream0)
            buf738 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
            # Topologically Sorted Source Nodes: [logits, logits_1, logits_2], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_prepare_softmax_online_view_16.run(buf737, buf738, 128, 2, stream=raw_stream0)
            buf739 = buf737; del buf737  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_1, logits_2], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_prepare_softmax_online_view_17.run(buf735, buf738, buf739, 256, 64128, stream=raw_stream0)
            buf740 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
            buf741 = reinterpret_tensor(buf740, (128, 1), (1, 1), 0); del buf740  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_1, logits_2, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online, aten._log_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18.run(buf741, buf739, 128, 2, stream=raw_stream0)
            del buf739
            buf744 = empty_strided_cuda((), (), torch.float32)
            buf743 = empty_strided_cuda((), (), torch.float32)
            buf773 = buf744; del buf744  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_1, getitem_200, logits_2, shift_labels_1, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.slice, aten.view, aten._log_softmax, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19.run(buf773, buf736, buf735, buf738, buf741, buf743, 1, 128, stream=raw_stream0)
        return (buf773, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, buf0, buf6, reinterpret_tensor(buf7, (128, 3072), (3072, 1), 0), buf14, buf15, buf16, reinterpret_tensor(buf20, (128, 3072), (3072, 1), 0), buf21, buf23, reinterpret_tensor(buf24, (128, 3072), (3072, 1), 0), buf25, buf26, reinterpret_tensor(buf27, (128, 8192), (8192, 1), 0), buf29, buf31, reinterpret_tensor(buf32, (128, 3072), (3072, 1), 0), buf39, buf40, buf41, reinterpret_tensor(buf45, (128, 3072), (3072, 1), 0), buf47, buf49, reinterpret_tensor(buf50, (128, 3072), (3072, 1), 0), buf51, buf52, reinterpret_tensor(buf53, (128, 8192), (8192, 1), 0), buf55, buf57, reinterpret_tensor(buf58, (128, 3072), (3072, 1), 0), buf65, buf66, buf67, reinterpret_tensor(buf71, (128, 3072), (3072, 1), 0), buf73, buf75, reinterpret_tensor(buf76, (128, 3072), (3072, 1), 0), buf77, buf78, reinterpret_tensor(buf79, (128, 8192), (8192, 1), 0), buf81, buf83, reinterpret_tensor(buf84, (128, 3072), (3072, 1), 0), buf91, buf92, buf93, reinterpret_tensor(buf97, (128, 3072), (3072, 1), 0), buf99, buf101, reinterpret_tensor(buf102, (128, 3072), (3072, 1), 0), buf103, buf104, reinterpret_tensor(buf105, (128, 8192), (8192, 1), 0), buf107, buf109, reinterpret_tensor(buf110, (128, 3072), (3072, 1), 0), buf117, buf118, buf119, reinterpret_tensor(buf123, (128, 3072), (3072, 1), 0), buf125, buf127, reinterpret_tensor(buf128, (128, 3072), (3072, 1), 0), buf129, buf130, reinterpret_tensor(buf131, (128, 8192), (8192, 1), 0), buf133, buf135, reinterpret_tensor(buf136, (128, 3072), (3072, 1), 0), buf143, buf144, buf145, reinterpret_tensor(buf149, (128, 3072), (3072, 1), 0), buf151, buf153, reinterpret_tensor(buf154, (128, 3072), (3072, 1), 0), buf155, buf156, reinterpret_tensor(buf157, (128, 8192), (8192, 1), 0), buf159, buf161, reinterpret_tensor(buf162, (128, 3072), (3072, 1), 0), buf169, buf170, buf171, reinterpret_tensor(buf175, (128, 3072), (3072, 1), 0), buf177, buf179, reinterpret_tensor(buf180, (128, 3072), (3072, 1), 0), buf181, buf182, reinterpret_tensor(buf183, (128, 8192), (8192, 1), 0), buf185, buf187, reinterpret_tensor(buf188, (128, 3072), (3072, 1), 0), buf195, buf196, buf197, reinterpret_tensor(buf201, (128, 3072), (3072, 1), 0), buf203, buf205, reinterpret_tensor(buf206, (128, 3072), (3072, 1), 0), buf207, buf208, reinterpret_tensor(buf209, (128, 8192), (8192, 1), 0), buf211, buf213, reinterpret_tensor(buf214, (128, 3072), (3072, 1), 0), buf221, buf222, buf223, reinterpret_tensor(buf227, (128, 3072), (3072, 1), 0), buf229, buf231, reinterpret_tensor(buf232, (128, 3072), (3072, 1), 0), buf233, buf234, reinterpret_tensor(buf235, (128, 8192), (8192, 1), 0), buf237, buf239, reinterpret_tensor(buf240, (128, 3072), (3072, 1), 0), buf247, buf248, buf249, reinterpret_tensor(buf253, (128, 3072), (3072, 1), 0), buf255, buf257, reinterpret_tensor(buf258, (128, 3072), (3072, 1), 0), buf259, buf260, reinterpret_tensor(buf261, (128, 8192), (8192, 1), 0), buf263, buf265, reinterpret_tensor(buf266, (128, 3072), (3072, 1), 0), buf273, buf274, buf275, reinterpret_tensor(buf279, (128, 3072), (3072, 1), 0), buf281, buf283, reinterpret_tensor(buf284, (128, 3072), (3072, 1), 0), buf285, buf286, reinterpret_tensor(buf287, (128, 8192), (8192, 1), 0), buf289, buf291, reinterpret_tensor(buf292, (128, 3072), (3072, 1), 0), buf299, buf300, buf301, reinterpret_tensor(buf305, (128, 3072), (3072, 1), 0), buf307, buf309, reinterpret_tensor(buf310, (128, 3072), (3072, 1), 0), buf311, buf312, reinterpret_tensor(buf313, (128, 8192), (8192, 1), 0), buf315, buf317, reinterpret_tensor(buf318, (128, 3072), (3072, 1), 0), buf325, buf326, buf327, reinterpret_tensor(buf331, (128, 3072), (3072, 1), 0), buf333, buf335, reinterpret_tensor(buf336, (128, 3072), (3072, 1), 0), buf337, buf338, reinterpret_tensor(buf339, (128, 8192), (8192, 1), 0), buf341, buf343, reinterpret_tensor(buf344, (128, 3072), (3072, 1), 0), buf351, buf352, buf353, reinterpret_tensor(buf357, (128, 3072), (3072, 1), 0), buf359, buf361, reinterpret_tensor(buf362, (128, 3072), (3072, 1), 0), buf363, buf364, reinterpret_tensor(buf365, (128, 8192), (8192, 1), 0), buf367, buf369, reinterpret_tensor(buf370, (128, 3072), (3072, 1), 0), buf377, buf378, buf379, reinterpret_tensor(buf383, (128, 3072), (3072, 1), 0), buf385, buf387, reinterpret_tensor(buf388, (128, 3072), (3072, 1), 0), buf389, buf390, reinterpret_tensor(buf391, (128, 8192), (8192, 1), 0), buf393, buf395, reinterpret_tensor(buf396, (128, 3072), (3072, 1), 0), buf403, buf404, buf405, reinterpret_tensor(buf409, (128, 3072), (3072, 1), 0), buf411, buf413, reinterpret_tensor(buf414, (128, 3072), (3072, 1), 0), buf415, buf416, reinterpret_tensor(buf417, (128, 8192), (8192, 1), 0), buf419, buf421, reinterpret_tensor(buf422, (128, 3072), (3072, 1), 0), buf429, buf430, buf431, reinterpret_tensor(buf435, (128, 3072), (3072, 1), 0), buf437, buf439, reinterpret_tensor(buf440, (128, 3072), (3072, 1), 0), buf441, buf442, reinterpret_tensor(buf443, (128, 8192), (8192, 1), 0), buf445, buf447, reinterpret_tensor(buf448, (128, 3072), (3072, 1), 0), buf455, buf456, buf457, reinterpret_tensor(buf461, (128, 3072), (3072, 1), 0), buf463, buf465, reinterpret_tensor(buf466, (128, 3072), (3072, 1), 0), buf467, buf468, reinterpret_tensor(buf469, (128, 8192), (8192, 1), 0), buf471, buf473, reinterpret_tensor(buf474, (128, 3072), (3072, 1), 0), buf481, buf482, buf483, reinterpret_tensor(buf487, (128, 3072), (3072, 1), 0), buf489, buf491, reinterpret_tensor(buf492, (128, 3072), (3072, 1), 0), buf493, buf494, reinterpret_tensor(buf495, (128, 8192), (8192, 1), 0), buf497, buf499, reinterpret_tensor(buf500, (128, 3072), (3072, 1), 0), buf507, buf508, buf509, reinterpret_tensor(buf513, (128, 3072), (3072, 1), 0), buf515, buf517, reinterpret_tensor(buf518, (128, 3072), (3072, 1), 0), buf519, buf520, reinterpret_tensor(buf521, (128, 8192), (8192, 1), 0), buf523, buf525, reinterpret_tensor(buf526, (128, 3072), (3072, 1), 0), buf533, buf534, buf535, reinterpret_tensor(buf539, (128, 3072), (3072, 1), 0), buf541, buf543, reinterpret_tensor(buf544, (128, 3072), (3072, 1), 0), buf545, buf546, reinterpret_tensor(buf547, (128, 8192), (8192, 1), 0), buf549, buf551, reinterpret_tensor(buf552, (128, 3072), (3072, 1), 0), buf559, buf560, buf561, reinterpret_tensor(buf565, (128, 3072), (3072, 1), 0), buf567, buf569, reinterpret_tensor(buf570, (128, 3072), (3072, 1), 0), buf571, buf572, reinterpret_tensor(buf573, (128, 8192), (8192, 1), 0), buf575, buf577, reinterpret_tensor(buf578, (128, 3072), (3072, 1), 0), buf585, buf586, buf587, reinterpret_tensor(buf591, (128, 3072), (3072, 1), 0), buf593, buf595, reinterpret_tensor(buf596, (128, 3072), (3072, 1), 0), buf597, buf598, reinterpret_tensor(buf599, (128, 8192), (8192, 1), 0), buf601, buf603, reinterpret_tensor(buf604, (128, 3072), (3072, 1), 0), buf611, buf612, buf613, reinterpret_tensor(buf617, (128, 3072), (3072, 1), 0), buf619, buf621, reinterpret_tensor(buf622, (128, 3072), (3072, 1), 0), buf623, buf624, reinterpret_tensor(buf625, (128, 8192), (8192, 1), 0), buf627, buf629, reinterpret_tensor(buf630, (128, 3072), (3072, 1), 0), buf637, buf638, buf639, reinterpret_tensor(buf643, (128, 3072), (3072, 1), 0), buf645, buf647, reinterpret_tensor(buf648, (128, 3072), (3072, 1), 0), buf649, buf650, reinterpret_tensor(buf651, (128, 8192), (8192, 1), 0), buf653, buf655, reinterpret_tensor(buf656, (128, 3072), (3072, 1), 0), buf663, buf664, buf665, reinterpret_tensor(buf669, (128, 3072), (3072, 1), 0), buf671, buf673, reinterpret_tensor(buf674, (128, 3072), (3072, 1), 0), buf675, buf676, reinterpret_tensor(buf677, (128, 8192), (8192, 1), 0), buf679, buf681, reinterpret_tensor(buf682, (128, 3072), (3072, 1), 0), buf689, buf690, buf691, reinterpret_tensor(buf695, (128, 3072), (3072, 1), 0), buf697, buf699, reinterpret_tensor(buf700, (128, 3072), (3072, 1), 0), buf701, buf702, reinterpret_tensor(buf703, (128, 8192), (8192, 1), 0), buf705, buf707, reinterpret_tensor(buf708, (128, 3072), (3072, 1), 0), buf715, buf716, buf717, reinterpret_tensor(buf721, (128, 3072), (3072, 1), 0), buf723, buf725, reinterpret_tensor(buf726, (128, 3072), (3072, 1), 0), buf727, buf728, reinterpret_tensor(buf729, (128, 8192), (8192, 1), 0), buf731, buf733, reinterpret_tensor(buf734, (128, 3072), (3072, 1), 0), buf735, buf736, reinterpret_tensor(buf738, (128, 1), (1, 1), 0), buf741, buf743, reinterpret_tensor(buf718, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf719, (24, 128, 128), (16384, 1, 128), 0), buf745, reinterpret_tensor(buf713, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf692, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf693, (24, 128, 128), (16384, 1, 128), 0), buf746, reinterpret_tensor(buf687, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf666, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf667, (24, 128, 128), (16384, 1, 128), 0), buf747, reinterpret_tensor(buf661, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf640, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf641, (24, 128, 128), (16384, 1, 128), 0), buf748, reinterpret_tensor(buf635, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf614, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf615, (24, 128, 128), (16384, 1, 128), 0), buf749, reinterpret_tensor(buf609, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf588, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf589, (24, 128, 128), (16384, 1, 128), 0), buf750, reinterpret_tensor(buf583, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf562, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf563, (24, 128, 128), (16384, 1, 128), 0), buf751, reinterpret_tensor(buf557, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf536, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf537, (24, 128, 128), (16384, 1, 128), 0), buf752, reinterpret_tensor(buf531, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf510, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf511, (24, 128, 128), (16384, 1, 128), 0), buf753, reinterpret_tensor(buf505, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf484, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf485, (24, 128, 128), (16384, 1, 128), 0), buf754, reinterpret_tensor(buf479, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf458, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf459, (24, 128, 128), (16384, 1, 128), 0), buf755, reinterpret_tensor(buf453, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf432, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf433, (24, 128, 128), (16384, 1, 128), 0), buf756, reinterpret_tensor(buf427, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf406, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf407, (24, 128, 128), (16384, 1, 128), 0), buf757, reinterpret_tensor(buf401, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf380, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf381, (24, 128, 128), (16384, 1, 128), 0), buf758, reinterpret_tensor(buf375, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf354, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf355, (24, 128, 128), (16384, 1, 128), 0), buf759, reinterpret_tensor(buf349, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf328, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf329, (24, 128, 128), (16384, 1, 128), 0), buf760, reinterpret_tensor(buf323, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf302, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf303, (24, 128, 128), (16384, 1, 128), 0), buf761, reinterpret_tensor(buf297, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf276, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf277, (24, 128, 128), (16384, 1, 128), 0), buf762, reinterpret_tensor(buf271, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf250, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf251, (24, 128, 128), (16384, 1, 128), 0), buf763, reinterpret_tensor(buf245, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf224, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf225, (24, 128, 128), (16384, 1, 128), 0), buf764, reinterpret_tensor(buf219, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf198, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf199, (24, 128, 128), (16384, 1, 128), 0), buf765, reinterpret_tensor(buf193, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf172, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf173, (24, 128, 128), (16384, 1, 128), 0), buf766, reinterpret_tensor(buf167, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf146, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf147, (24, 128, 128), (16384, 1, 128), 0), buf767, reinterpret_tensor(buf141, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf120, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf121, (24, 128, 128), (16384, 1, 128), 0), buf768, reinterpret_tensor(buf115, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf94, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf95, (24, 128, 128), (16384, 1, 128), 0), buf769, reinterpret_tensor(buf89, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf68, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf69, (24, 128, 128), (16384, 1, 128), 0), buf770, reinterpret_tensor(buf63, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf42, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf43, (24, 128, 128), (16384, 1, 128), 0), buf771, reinterpret_tensor(buf37, (24, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf17, (24, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf18, (24, 128, 128), (16384, 1, 128), 0), buf772, reinterpret_tensor(buf12, (24, 128, 128), (16384, 128, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((1, 128), (128, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((128256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_15 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_16 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_17 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_18 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_19 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_20 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_21 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_22 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_23 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_24 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_25 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_26 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_27 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_28 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_29 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_30 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_31 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_32 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_33 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_34 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_35 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_36 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_37 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_38 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_39 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_40 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_41 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_42 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_43 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_44 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_45 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_46 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_47 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_48 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_49 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_50 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_51 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_52 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_53 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_54 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_55 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_56 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_57 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_58 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_59 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_60 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_61 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_62 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_63 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_64 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_65 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_66 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_67 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_68 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_69 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_70 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_71 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_72 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_73 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_74 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_75 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_76 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_77 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_78 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_79 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_80 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_81 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_82 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_83 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_84 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_85 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_86 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_87 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_88 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_89 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_90 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_91 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_92 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_93 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_94 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_95 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_96 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_97 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_98 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_99 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_100 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_101 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_102 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_103 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_104 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_105 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_106 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_107 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_108 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_109 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_110 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_111 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_112 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_113 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_114 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_115 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_116 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_117 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_118 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_119 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_120 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_121 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_122 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_123 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_124 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_125 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_126 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_127 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_128 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_129 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_130 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_131 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_132 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_133 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_134 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_135 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_136 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_137 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_138 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_139 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_140 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_141 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_142 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_143 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_144 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_145 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_146 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_147 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_148 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_149 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_150 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_151 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_152 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_153 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_154 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_155 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_156 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_157 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_158 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_159 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_160 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_161 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_162 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_163 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_164 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_165 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_166 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_167 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_168 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_169 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_170 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_171 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_172 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_173 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_174 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_175 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_176 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_177 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_178 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_179 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_180 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_181 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_182 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_183 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_184 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_185 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_186 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_187 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_188 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_189 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_190 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_191 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_192 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_193 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_194 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_195 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_196 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_197 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_198 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_199 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_200 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_201 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_202 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_203 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_204 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_205 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_206 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_207 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_208 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_209 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_210 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_211 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_212 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_213 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_214 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_215 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_216 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_217 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_218 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_219 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_220 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_221 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_222 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_223 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_224 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_225 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_226 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_227 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_228 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_229 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_230 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_231 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_232 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_233 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_234 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_235 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_236 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_237 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_238 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_239 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_240 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_241 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_242 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_243 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_244 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_245 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_246 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_247 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_248 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_249 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_250 = rand_strided((1024, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_251 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_252 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_253 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_254 = rand_strided((8192, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_255 = rand_strided((3072, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_256 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
