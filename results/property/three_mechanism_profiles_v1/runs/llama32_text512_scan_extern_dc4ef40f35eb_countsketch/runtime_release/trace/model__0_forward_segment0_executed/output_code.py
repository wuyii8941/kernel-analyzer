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


# kernel path: /tmp/torchinductor_tzh/2z/c2z3rtvqw6xrh4wj6nwxkr5vurptpl5ughqukj256gq6dof44fn2.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub]
# Source node to ATen node mapping:
#   cache_position => iota
#   first_dummy_value => sub
#   getitem => slice_1
#   position_ids => unsqueeze
# Graph fragment:
#   %iota : Tensor "i64[512][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 1][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%unsqueeze, 1, 0, 1), kwargs = {})
#   %sub : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, 1), kwargs = {})
#   return %sub
triton_poi_fused_arange_slice_sub_unsqueeze_0 = async_compile.triton('triton_poi_fused_arange_slice_sub_unsqueeze_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_slice_sub_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_slice_sub_unsqueeze_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.full([1], -1, tl.int64)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/ka/ckaq4k2jbafxbmnztxnkpgdwzx6ibsnbz45e3wmqwq2ml4xgklz7.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat]
# Source node to ATen node mapping:
#   cache_position => iota
#   first_dummy_value => sub
#   getitem => slice_1
#   position_diff => cat
#   position_ids => unsqueeze
# Graph fragment:
#   %iota : Tensor "i64[512][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %slice_1 : Tensor "i64[1, 1][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%unsqueeze, 1, 0, 1), kwargs = {})
#   %sub : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, 1), kwargs = {})
#   %cat : Tensor "i64[1, 513][513, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sub, %unsqueeze], -1), kwargs = {})
#   return %buf1
triton_poi_fused_arange_cat_slice_sub_unsqueeze_1 = async_compile.triton('triton_poi_fused_arange_cat_slice_sub_unsqueeze_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(1,), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_arange_cat_slice_sub_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 8192}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_cat_slice_sub_unsqueeze_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/bo/cbo5o4z44lhuzmyyqk3x2hxpnxaabg4aw7q6umajxlr6no6w7q4d.py
# Topologically Sorted Source Nodes: [position_diff, ne, packed_sequence_mask], Original ATen: [aten.slice, aten.sub, aten.ne, aten.cumsum]
# Source node to ATen node mapping:
#   ne => ne
#   packed_sequence_mask => cumsum
#   position_diff => slice_2, slice_3, sub_1
# Graph fragment:
#   %cat : Tensor "i64[1, 513][513, 1]cuda:0" = PlaceHolder[target=cat]
#   %slice_2 : Tensor "i64[1, 512][513, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 0, 512), kwargs = {})
#   %slice_3 : Tensor "i64[1, 512][513, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cat, -1, 1, 513), kwargs = {})
#   %sub_1 : Tensor "i64[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_3, %slice_2), kwargs = {})
#   %ne : Tensor "b8[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ne.Scalar](args = (%sub_1, 1), kwargs = {})
#   %cumsum : Tensor "i64[1, 512][512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%ne, -1), kwargs = {})
#   return %cumsum
triton_per_fused_cumsum_ne_slice_sub_2 = async_compile.triton('triton_per_fused_cumsum_ne_slice_sub_2', '''
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
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=(2,))]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_cumsum_ne_slice_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'r0_': 16384}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused_cumsum_ne_slice_sub_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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


# kernel path: /tmp/torchinductor_tzh/tr/ctr5r2umwngoqfvtpzrwrucyqeh3w2hhbt6icjrit53uovvyf6hm.py
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
#   %primals_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "bf16[128256, 3072][3072, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_4 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_4]
#   %buf4 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf4]
#   %embedding : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_3, 2), kwargs = {})
#   %mean : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul_3 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %rsqrt), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, %convert_element_type_4), kwargs = {})
#   return %buf4,%mul_4
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 4096, 'r0_': 6297600}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
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
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 128256")
        tmp10 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 3072.0, tl.float32)
        tmp13 = (tmp6 / tmp12)
        tmp14 = tl.full([1, 1], 1e-05, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp8 * tmp18
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp19, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/s2/cs25gquaclncymmqvayrak2yoe7xx4qyuxwi2lvn5vagtastltru.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear, view, query_states, linear_1, view_1, key_states, cos_3, sin_3, mul_4, x1, x2, neg, cat_1, mul_5, q_embed, mul_6, x1_1, x2_1, neg_1, cat_2, mul_7, k_embed, getitem_7, hidden_states_3, key_states_1], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
# Source node to ATen node mapping:
#   cache_position => iota
#   cat_1 => cat_1
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
#   linear => view_12
#   linear_1 => view_15
#   matmul => mul
#   mul_4 => mul_5
#   mul_5 => mul_6
#   mul_6 => mul_7
#   mul_7 => mul_8
#   neg => neg
#   neg_1 => neg_1
#   position_ids => unsqueeze
#   position_ids_expanded => convert_element_type
#   q_embed => add_2
#   query_states => permute_2
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_6
#   view => view_13
#   view_1 => view_16
#   x1 => slice_4
#   x1_1 => slice_6
#   x2 => slice_5
#   x2_1 => slice_7
# Graph fragment:
#   %mm : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %mm_1 : Tensor "bf16[512, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %iota : Tensor "i64[512][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 512][512, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, 0), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_2, [1, -1, 1]), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type : Tensor "f32[1, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_3, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 512][32768, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 512, 64][32768, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 512, 1, 64][32768, 1, 32768, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 512, 2, 64][32768, 1, 0, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 512, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 512, 2, 64][65536, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_10 : Tensor "f32[1, 512, 128][65536, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 512, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 512, 128][65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_10,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 512, 128][65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %sin : Tensor "f32[1, 512, 128][65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_10,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 512, 128][65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 512, 128][65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 512, 128][65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %view_12 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 512, 3072]), kwargs = {})
#   %view_13 : Tensor "bf16[1, 512, 24, 128][1572864, 3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_12, [1, 512, -1, 128]), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 24, 512, 128][1572864, 128, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 2, 1, 3]), kwargs = {})
#   %view_15 : Tensor "bf16[1, 512, 1024][524288, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 512, 1024]), kwargs = {})
#   %view_16 : Tensor "bf16[1, 512, 8, 128][524288, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_15, [1, 512, -1, 128]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 8, 512, 128][524288, 128, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_16, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 512, 128][65536, 65536, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 512, 128][65536, 65536, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_5 : Tensor "bf16[1, 24, 512, 128][1572864, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_5), kwargs = {})
#   %slice_4 : Tensor "bf16[1, 24, 512, 64][1572864, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 64), kwargs = {})
#   %slice_5 : Tensor "bf16[1, 24, 512, 64][1572864, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 64, 9223372036854775807), kwargs = {})
#   %neg : Tensor "bf16[1, 24, 512, 64][786432, 64, 1536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %cat_1 : Tensor "bf16[1, 24, 512, 128][1572864, 65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_4], -1), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 24, 512, 128][1572864, 65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_6), kwargs = {})
#   %add_2 : Tensor "bf16[1, 24, 512, 128][1572864, 128, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %mul_7 : Tensor "bf16[1, 8, 512, 128][524288, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_5), kwargs = {})
#   %slice_6 : Tensor "bf16[1, 8, 512, 64][524288, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 0, 64), kwargs = {})
#   %slice_7 : Tensor "bf16[1, 8, 512, 64][524288, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 64, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "bf16[1, 8, 512, 64][262144, 64, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_7,), kwargs = {})
#   %cat_2 : Tensor "bf16[1, 8, 512, 128][524288, 65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_6], -1), kwargs = {})
#   %mul_8 : Tensor "bf16[1, 8, 512, 128][524288, 65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_2, %unsqueeze_6), kwargs = {})
#   %add_3 : Tensor "bf16[1, 8, 512, 128][524288, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %mul_8), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 8, 1, 512, 128][524288, 128, 524288, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_3, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 3, 512, 128][524288, 128, 0, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_7, [1, 8, 3, 512, 128]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 3, 512, 128][1572864, 196608, 65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   return %expand_7,%clone_2
triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 25166080}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x6 = xindex
    x2 = xindex // 3072
    x0 = (xindex % 128)
    x7 = xindex // 128
    x1 = ((xindex // 128) % 24)
    x3 = ((xindex // 128) % 512)
    x5 = xindex // 196608
    tmp0 = tl.load(in_ptr0 + (x6), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + ((x6 % 64)), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr2 + (x0 + 128*x5 + 1024*x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = x2
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
    tmp15 = tl.load(in_ptr0 + (64 + 128*x7 + (x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = -tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp14, tmp16, tmp17)
    tmp19 = tmp10 >= tmp13
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp10 < tmp20
    tmp22 = tl.load(in_ptr0 + (128*x7 + ((-64) + x0)), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.where(tmp14, tmp18, tmp22)
    tmp24 = tl_math.sin(tmp4)
    tmp25 = tmp24 * tmp6
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp23 * tmp26
    tmp28 = tmp9 + tmp27
    tmp30 = x3
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp1 * tmp31
    tmp33 = tl_math.cos(tmp32)
    tmp34 = tmp33 * tmp6
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp29 * tmp35
    tmp37 = tl.load(in_ptr2 + (64 + 128*x5 + 1024*x3 + (x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp38 = -tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp14, tmp38, tmp39)
    tmp41 = tl.load(in_ptr2 + (128*x5 + 1024*x3 + ((-64) + x0)), tmp19, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp42 = tl.where(tmp14, tmp40, tmp41)
    tmp43 = tl_math.sin(tmp32)
    tmp44 = tmp43 * tmp6
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tmp42 * tmp45
    tmp47 = tmp36 + tmp46
    tl.store(out_ptr0 + (x0 + 128*x2 + 65536*x1), tmp28, None)
    tl.store(out_ptr1 + (x6), tmp47, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/gp/cgpvfbo75c6kovvpc2uiawxiy272nouw4lk5hiwrhlqydvss5vfr.py
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
#   %bmm : Tensor "bf16[24, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=bmm]
#   %cumsum : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %getitem_56 : Tensor "f32[1, 24, 512, 1][12288, 512, 1, 12288]cuda:0" = PlaceHolder[target=getitem_56]
#   %getitem_57 : Tensor "f32[1, 24, 512, 1][12288, 512, 1, 12288]cuda:0" = PlaceHolder[target=getitem_57]
#   %iota : Tensor "i64[512][1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.iota.default](args = (512,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %iota_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [512, 1]), kwargs = {})
#   %le : Tensor "b8[512, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add, %view), kwargs = {})
#   %full_default : Tensor "b8[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([512, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and : Tensor "b8[512, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %view_2 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index : Tensor "i64[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_2, %iota]), kwargs = {})
#   %index_1 : Tensor "i64[1, 512][512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%cumsum, [%view_2, %add]), kwargs = {})
#   %view_4 : Tensor "i64[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [1, 512, 1]), kwargs = {})
#   %view_5 : Tensor "i64[1, 1, 512][512, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index_1, [1, 1, 512]), kwargs = {})
#   %eq : Tensor "b8[1, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%view_4, %view_5), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 512, 512][262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %eq), kwargs = {})
#   %view_6 : Tensor "b8[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and_1, [1, 1, 512, 512]), kwargs = {})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.3895313892515355e+38), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_1, %full_default_2), kwargs = {})
#   %view_24 : Tensor "bf16[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [1, 24, 512, 512]), kwargs = {})
#   %mul_9 : Tensor "bf16[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_24, 0.08838834764831845), kwargs = {})
#   %add_4 : Tensor "bf16[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %where), kwargs = {})
#   %convert_element_type_13 : Tensor "f32[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_28 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_13, -1), kwargs = {})
#   %sub_tensor_28 : Tensor "f32[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_13, %getitem_56), kwargs = {})
#   %exp_default_28 : Tensor "f32[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_28,), kwargs = {})
#   %div : Tensor "f32[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_28, %getitem_57), kwargs = {})
#   %convert_element_type_14 : Tensor "bf16[1, 24, 512, 512][6291456, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   return %getitem_56,%getitem_57,%expand_9
triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5 = async_compile.triton('triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 4096, 'r0_': 37752832}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
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
    tmp21 = triton_helpers.max2(tmp19, 1)[:, None].to(tl.float32)
    tmp22 = tmp17 - tmp21
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
    tmp26 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp27 = tmp16 - tmp21
    tmp28 = libdevice.exp(tmp27)
    tmp29 = (tmp28 / tmp26)
    tmp30 = tmp29.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/t3/ct3thansnqvexjex4q2itanyzfue3ncg2oocqllbay4evw7zj5hz.py
# Topologically Sorted Source Nodes: [linear_2, view_2, value_states, getitem_8, hidden_states_4, value_states_1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   getitem_8 => unsqueeze_8
#   hidden_states_4 => expand_6
#   linear_2 => view_18
#   value_states => permute_6
#   value_states_1 => clone_3
#   view_2 => view_19
# Graph fragment:
#   %mm_2 : Tensor "bf16[512, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_18 : Tensor "bf16[1, 512, 1024][524288, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 512, 1024]), kwargs = {})
#   %view_19 : Tensor "bf16[1, 512, 8, 128][524288, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_18, [1, 512, -1, 128]), kwargs = {})
#   %permute_6 : Tensor "bf16[1, 8, 512, 128][524288, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_19, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_8 : Tensor "bf16[1, 8, 1, 512, 128][524288, 128, 524288, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_6, 2), kwargs = {})
#   %expand_6 : Tensor "bf16[1, 8, 3, 512, 128][524288, 128, 0, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, 8, 3, 512, 128]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 3, 512, 128][1572864, 196608, 65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 7340032}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 512)
    x3 = xindex // 196608
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x3 + 1024*x1), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/mm/cmmyuivjm4m3rwm5cqxfmxepvvea3umtsbnn4vkz3usz3ek3i53n.py
# Topologically Sorted Source Nodes: [attn_output, transpose_5, attn_output_1], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   attn_output => view_27
#   attn_output_1 => clone_5
#   transpose_5 => permute_8
# Graph fragment:
#   %bmm_1 : Tensor "bf16[24, 512, 128][65536, 128, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_27 : Tensor "bf16[1, 24, 512, 128][1572864, 65536, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [1, 24, 512, 128]), kwargs = {})
#   %permute_8 : Tensor "bf16[1, 512, 24, 128][1572864, 128, 65536, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_27, [0, 2, 1, 3]), kwargs = {})
#   %clone_5 : Tensor "bf16[1, 512, 24, 128][1572864, 3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_clone_transpose_view_7 = async_compile.triton('triton_poi_fused_clone_transpose_view_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 9437184}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 24)
    x2 = xindex // 3072
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 65536*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/6v/c6vnvhhzopmv4jt7lpmrpk65nemzse3akyndg7pf4tlyy7drtano.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_5, rsqrt_1, hidden_states_7, to_9, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_5 => add_6
#   attn_output_3 => view_30
#   hidden_states_5 => add_5
#   hidden_states_6 => convert_element_type_19
#   hidden_states_7 => mul_10
#   hidden_states_8 => mul_11
#   inputs_embeds => embedding
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to_9 => convert_element_type_20
#   variance_1 => mean_1
# Graph fragment:
#   %primals_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "bf16[128256, 3072][3072, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %mm_3 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %primals_9 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_9]
#   %buf19 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf19]
#   %embedding : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1), kwargs = {})
#   %view_30 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 512, 3072]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_30), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_19, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_6 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %mul_10 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %rsqrt_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   %mul_11 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, %convert_element_type_20), kwargs = {})
#   return %buf19,%mul_11
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 4096, 'r0_': 9443328}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
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
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 128256")
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 128256")
        tmp12 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tl.full([1, 1], 3072.0, tl.float32)
        tmp17 = (tmp8 / tmp16)
        tmp18 = tl.full([1, 1], 1e-05, tl.float32)
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp10 * tmp22
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp23, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/xj/cxjiclbuzodfmdu2daxqqomu733fft6aip57tsjurzbcli2asnhi.py
# Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   linear_4 => view_32
#   linear_5 => view_34
#   mul_11 => mul_12
#   silu => add_7, convert_element_type_23, convert_element_type_24, div_1, exp_1, neg_2
# Graph fragment:
#   %mm_4 : Tensor "bf16[512, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %mm_5 : Tensor "bf16[512, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %view_32 : Tensor "bf16[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [1, 512, 8192]), kwargs = {})
#   %convert_element_type_23 : Tensor "f32[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_32, torch.float32), kwargs = {})
#   %neg_2 : Tensor "f32[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_23,), kwargs = {})
#   %exp_1 : Tensor "f32[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_7 : Tensor "f32[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_1, 1), kwargs = {})
#   %div_1 : Tensor "f32[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_23, %add_7), kwargs = {})
#   %convert_element_type_24 : Tensor "bf16[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_1, torch.bfloat16), kwargs = {})
#   %view_34 : Tensor "bf16[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [1, 512, 8192]), kwargs = {})
#   %mul_12 : Tensor "bf16[1, 512, 8192][4194304, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %view_34), kwargs = {})
#   return %mul_12
triton_poi_fused__unsafe_view_mul_silu_9 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 33554432}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/f3/cf3j4o3zkeysoj6qdpjioov6esjdte7y6rne7tdielxsel6i7l2a.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, to_11, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_7 => add_9
#   attn_output_3 => view_30
#   down_proj => view_36
#   hidden_states_10 => convert_element_type_29
#   hidden_states_11 => mul_13
#   hidden_states_12 => mul_14
#   hidden_states_5 => add_5
#   hidden_states_9 => add_8
#   inputs_embeds => embedding
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   to_11 => convert_element_type_30
#   variance_2 => mean_2
# Graph fragment:
#   %primals_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "bf16[128256, 3072][3072, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %mm_3 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_6 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %primals_13 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_13]
#   %buf25 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf25]
#   %embedding : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1), kwargs = {})
#   %view_30 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 512, 3072]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_30), kwargs = {})
#   %view_36 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 512, 3072]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_36), kwargs = {})
#   %convert_element_type_29 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_29, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_9 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-05), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_13 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_29, %rsqrt_2), kwargs = {})
#   %convert_element_type_30 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   %mul_14 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_13, %convert_element_type_30), kwargs = {})
#   return %buf25,%mul_14
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 4096, 'r0_': 12589056}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
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
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 128256")
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 128256")
        tmp14 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full([1, 1], 3072.0, tl.float32)
        tmp21 = (tmp10 / tmp20)
        tmp22 = tl.full([1, 1], 1e-05, tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp12 * tmp26
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp27, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/dt/cdtjqxwfgmpv2mfh22ur3rgeyi6rq66go7elkyrtoytarsz6ztvs.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, attn_output_7, hidden_states_15, hidden_states_16, pow_4, variance_3, add_12, rsqrt_3, hidden_states_17, to_14, hidden_states_18], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_12 => add_14
#   attn_output_3 => view_30
#   attn_output_7 => view_56
#   down_proj => view_36
#   hidden_states_15 => add_13
#   hidden_states_16 => convert_element_type_45
#   hidden_states_17 => mul_20
#   hidden_states_18 => mul_21
#   hidden_states_5 => add_5
#   hidden_states_9 => add_8
#   inputs_embeds => embedding
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   to_14 => convert_element_type_46
#   variance_3 => mean_3
# Graph fragment:
#   %primals_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %primals_2 : Tensor "bf16[128256, 3072][3072, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %mm_3 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_6 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %mm_10 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %add_13 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=add_13]
#   %primals_18 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_18]
#   %buf41 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf41]
#   %embedding : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_2, %primals_1), kwargs = {})
#   %view_30 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 512, 3072]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_30), kwargs = {})
#   %view_36 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 512, 3072]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_36), kwargs = {})
#   %view_56 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [1, 512, 3072]), kwargs = {})
#   %add_13 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_56), kwargs = {})
#   %convert_element_type_45 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_13, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_45, 2), kwargs = {})
#   %mean_3 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_14 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-05), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_20 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_45, %rsqrt_3), kwargs = {})
#   %convert_element_type_46 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_20, torch.bfloat16), kwargs = {})
#   %mul_21 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, %convert_element_type_46), kwargs = {})
#   return %add_13,%buf41,%mul_21
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 4096, 'r0_': 22026240}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
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
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp3 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tl.device_assert(((0 <= tmp0) & (tmp0 < 128256)) | ~(xmask), "index out of bounds: 0 <= tmp0 < 128256")
        tmp2 = tl.load(in_ptr1 + (r0_1 + 3072*tmp0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp8, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full([1, 1], 3072.0, tl.float32)
        tmp18 = (tmp12 / tmp17)
        tmp19 = tl.full([1, 1], 1e-05, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp14 * tmp23
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/6n/c6n7kwq5svwnfwmbf4wnfxdv5ishpoci2qrlal4aovn2hqswiahk.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, hidden_states_20, pow_5, variance_4, add_14, rsqrt_4, hidden_states_21, to_16, hidden_states_22], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_14 => add_17
#   down_proj_1 => view_62
#   hidden_states_19 => add_16
#   hidden_states_20 => convert_element_type_55
#   hidden_states_21 => mul_23
#   hidden_states_22 => mul_24
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
#   to_16 => convert_element_type_56
#   variance_4 => mean_4
# Graph fragment:
#   %add_13 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=add_13]
#   %mm_13 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %primals_22 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_22]
#   %buf47 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf47]
#   %view_62 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 512, 3072]), kwargs = {})
#   %add_16 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_62), kwargs = {})
#   %convert_element_type_55 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_16, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_55, 2), kwargs = {})
#   %mean_4 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_17 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-05), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %mul_23 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_55, %rsqrt_4), kwargs = {})
#   %convert_element_type_56 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.bfloat16), kwargs = {})
#   %mul_24 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_22, %convert_element_type_56), kwargs = {})
#   return %buf47,%mul_24
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 0, 'r0_': 12589056}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
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
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.full([1, 1], 3072.0, tl.float32)
        tmp14 = (tmp6 / tmp13)
        tmp15 = tl.full([1, 1], 1e-05, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp8 * tmp19
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/ei/cei3uzhiixhbt7fkpar2lkab3hc4lwm4expna2sy5xho65ylf2we.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, hidden_states_26, pow_6, variance_5, add_19, rsqrt_5, hidden_states_27, to_19, hidden_states_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_19 => add_22
#   attn_output_11 => view_82
#   down_proj_1 => view_62
#   hidden_states_19 => add_16
#   hidden_states_25 => add_21
#   hidden_states_26 => convert_element_type_71
#   hidden_states_27 => mul_30
#   hidden_states_28 => mul_31
#   pow_6 => pow_6
#   rsqrt_5 => rsqrt_5
#   to_19 => convert_element_type_72
#   variance_5 => mean_5
# Graph fragment:
#   %add_13 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=add_13]
#   %mm_13 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %mm_17 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %primals_27 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_27]
#   %buf62 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf62]
#   %view_62 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 512, 3072]), kwargs = {})
#   %add_16 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_62), kwargs = {})
#   %view_82 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [1, 512, 3072]), kwargs = {})
#   %add_21 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_82), kwargs = {})
#   %convert_element_type_71 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_21, torch.float32), kwargs = {})
#   %pow_6 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_71, 2), kwargs = {})
#   %mean_5 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-05), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_30 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_71, %rsqrt_5), kwargs = {})
#   %convert_element_type_72 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_30, torch.bfloat16), kwargs = {})
#   %mul_31 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_27, %convert_element_type_72), kwargs = {})
#   return %buf62,%mul_31
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 0, 'r0_': 15734784}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full([1, 1], 3072.0, tl.float32)
        tmp18 = (tmp8 / tmp17)
        tmp19 = tl.full([1, 1], 1e-05, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp10 * tmp23
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/54/c543642vf7n5bj7tb6nxlkru6sw5czrpknwc2nzwhezqzrzfdt6r.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, hidden_states_30, pow_7, variance_6, add_21, rsqrt_6, hidden_states_31, to_21, hidden_states_32], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_21 => add_25
#   attn_output_11 => view_82
#   down_proj_1 => view_62
#   down_proj_2 => view_88
#   hidden_states_19 => add_16
#   hidden_states_25 => add_21
#   hidden_states_29 => add_24
#   hidden_states_30 => convert_element_type_81
#   hidden_states_31 => mul_33
#   hidden_states_32 => mul_34
#   pow_7 => pow_7
#   rsqrt_6 => rsqrt_6
#   to_21 => convert_element_type_82
#   variance_6 => mean_6
# Graph fragment:
#   %add_13 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=add_13]
#   %mm_13 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %mm_17 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %mm_20 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %primals_31 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_31]
#   %buf68 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf68]
#   %view_62 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 512, 3072]), kwargs = {})
#   %add_16 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_62), kwargs = {})
#   %view_82 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [1, 512, 3072]), kwargs = {})
#   %add_21 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_82), kwargs = {})
#   %view_88 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_20, [1, 512, 3072]), kwargs = {})
#   %add_24 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_88), kwargs = {})
#   %convert_element_type_81 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_24, torch.float32), kwargs = {})
#   %pow_7 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_81, 2), kwargs = {})
#   %mean_6 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [-1], True), kwargs = {})
#   %add_25 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-05), kwargs = {})
#   %rsqrt_6 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
#   %mul_33 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_81, %rsqrt_6), kwargs = {})
#   %convert_element_type_82 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.bfloat16), kwargs = {})
#   %mul_34 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_31, %convert_element_type_82), kwargs = {})
#   return %buf68,%mul_34
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 0, 'r0_': 18880512}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.full([1, 1], 3072.0, tl.float32)
        tmp22 = (tmp10 / tmp21)
        tmp23 = tl.full([1, 1], 1e-05, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp12 * tmp27
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp28, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/ht/chtw6urtoaetpy2iojufgq6mlvwjibz7s6vtpdxoa2fkqb3pfwpo.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, attn_output_15, hidden_states_35, hidden_states_36, pow_8, variance_7, add_26, rsqrt_7, hidden_states_37, to_24, hidden_states_38], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_26 => add_30
#   attn_output_11 => view_82
#   attn_output_15 => view_108
#   down_proj_1 => view_62
#   down_proj_2 => view_88
#   hidden_states_19 => add_16
#   hidden_states_25 => add_21
#   hidden_states_29 => add_24
#   hidden_states_35 => add_29
#   hidden_states_36 => convert_element_type_97
#   hidden_states_37 => mul_40
#   hidden_states_38 => mul_41
#   pow_8 => pow_8
#   rsqrt_7 => rsqrt_7
#   to_24 => convert_element_type_98
#   variance_7 => mean_7
# Graph fragment:
#   %add_13 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=add_13]
#   %mm_13 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %mm_17 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %mm_20 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %mm_24 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_24]
#   %add_29 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=add_29]
#   %primals_36 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_36]
#   %buf84 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf84]
#   %view_62 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 512, 3072]), kwargs = {})
#   %add_16 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_62), kwargs = {})
#   %view_82 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [1, 512, 3072]), kwargs = {})
#   %add_21 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %view_82), kwargs = {})
#   %view_88 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_20, [1, 512, 3072]), kwargs = {})
#   %add_24 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_21, %view_88), kwargs = {})
#   %view_108 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_24, [1, 512, 3072]), kwargs = {})
#   %add_29 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %view_108), kwargs = {})
#   %convert_element_type_97 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_29, torch.float32), kwargs = {})
#   %pow_8 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_97, 2), kwargs = {})
#   %mean_7 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_30 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-05), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_30,), kwargs = {})
#   %mul_40 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_97, %rsqrt_7), kwargs = {})
#   %convert_element_type_98 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_40, torch.bfloat16), kwargs = {})
#   %mul_41 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_36, %convert_element_type_98), kwargs = {})
#   return %add_29,%buf84,%mul_41
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 0, 'r0_': 28317696}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 3072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp8, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full([1, 1], 3072.0, tl.float32)
        tmp18 = (tmp12 / tmp17)
        tmp19 = tl.full([1, 1], 1e-05, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp14 * tmp23
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/y7/cy7hbvofcpvmghsjyempbt7cyxj7gn6ma3fdop7cdpnbwwzwp2lm.py
# Topologically Sorted Source Nodes: [down_proj_27, hidden_states_279, hidden_states_280, pow_57, variance_56, add_196, rsqrt_56, hidden_states_281, to_146, hidden_states_282], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_196 => add_225
#   down_proj_27 => view_738
#   hidden_states_279 => add_224
#   hidden_states_280 => convert_element_type_731
#   hidden_states_281 => mul_283
#   hidden_states_282 => mul_284
#   pow_57 => pow_57
#   rsqrt_56 => rsqrt_56
#   to_146 => convert_element_type_732
#   variance_56 => mean_56
# Graph fragment:
#   %add_221 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=add_221]
#   %mm_195 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_195]
#   %buf606 : Tensor "f32[1, 512, 1][512, 1, 512]cuda:0" = PlaceHolder[target=buf606]
#   %primals_256 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_256]
#   %convert_element_type_732 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=convert_element_type_732]
#   %view_738 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_195, [1, 512, 3072]), kwargs = {})
#   %add_224 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_221, %view_738), kwargs = {})
#   %convert_element_type_731 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_224, torch.float32), kwargs = {})
#   %pow_57 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_731, 2), kwargs = {})
#   %mean_56 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_57, [-1], True), kwargs = {})
#   %add_225 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_56, 1e-05), kwargs = {})
#   %rsqrt_56 : Tensor "f32[1, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_225,), kwargs = {})
#   %mul_283 : Tensor "f32[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_731, %rsqrt_56), kwargs = {})
#   %convert_element_type_732 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_283, torch.bfloat16), kwargs = {})
#   %mul_284 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_256, %convert_element_type_732), kwargs = {})
#   return %buf606,%convert_element_type_732,%mul_284
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_16 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 0, 'r0_': 18880512}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_16(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.full([1, 1], 3072.0, tl.float32)
        tmp13 = (tmp6 / tmp12)
        tmp14 = tl.full([1, 1], 1e-05, tl.float32)
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19 * tmp18
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp18, r0_mask & xmask)
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/uh/cuhetunqkcjbt4xveor4afglycwbyemaad4gzfg45zhv4rcdxfzu.py
# Topologically Sorted Source Nodes: [labels], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   labels => constant_pad_nd
# Graph fragment:
#   %primals_1 : Tensor "i64[1, 512][512, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %constant_pad_nd : Tensor "i64[1, 513][513, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [0, 1], -100.0), kwargs = {})
#   return %constant_pad_nd
triton_poi_fused_constant_pad_nd_17 = async_compile.triton('triton_poi_fused_constant_pad_nd_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 12304}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 513
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 513, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.full([1], -100, tl.int64)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp6, tmp9, tmp10)
    tmp12 = tl.where(tmp4, tmp5, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/6i/c6izikhhg3o6zz34cn7kxyl6terg7sqgenf6sbcfzcq3lrg6bnfm.py
# Topologically Sorted Source Nodes: [logits, logits_1, logits_2, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online, aten._log_softmax]
# Source node to ATen node mapping:
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
#   loss => log
# Graph fragment:
#   %mm_196 : Tensor "bf16[512, 128256][128256, 1]cuda:0" = PlaceHolder[target=mm_196]
#   %getitem_1 : Tensor "f32[512, 1][1, 512]cuda:0" = PlaceHolder[target=getitem_1]
#   %view_740 : Tensor "bf16[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 512, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %view_741 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%view_741, 1), kwargs = {})
#   %log : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   return %getitem,%getitem_1,%log
triton_red_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18 = async_compile.triton('triton_red_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 8192, 'r0_': 131334144}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 128256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp3_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 128256*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])

        _tmp3_max_next, _tmp3_sum_next = triton_helpers.online_softmax_combine(
            _tmp3_max, _tmp3_sum, tmp2, False
        )

        _tmp3_max = tl.where(r0_mask & xmask, _tmp3_max_next, _tmp3_max)
        _tmp3_sum = tl.where(r0_mask & xmask, _tmp3_sum_next, _tmp3_sum)

    tmp3, tmp4 = triton_helpers.online_softmax_reduce(
        _tmp3_max, _tmp3_sum, 1, False)
    tmp3 = tmp3[:, None]
    tmp4 = tmp4[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tmp5 = tl_math.log(tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/5m/c5mlekokzf23dwwp72b2lzsmx54sns4tnjzolp2jrz7bl54guu5p.py
# Topologically Sorted Source Nodes: [logits, logits_1, getitem_200, logits_2, shift_labels_1, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.slice, aten.view, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   getitem_200 => slice_116
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
#   loss => convert_element_type_736, div_56, full_default_3, full_default_4, gather, ne_1, neg_84, squeeze, sub_31, sub_tensor, sum_30, sum_31, unsqueeze_117, where_1, where_2
#   shift_labels_1 => view_742
# Graph fragment:
#   %constant_pad_nd : Tensor "i64[1, 513][513, 1]cuda:0" = PlaceHolder[target=constant_pad_nd]
#   %mm_196 : Tensor "bf16[512, 128256][128256, 1]cuda:0" = PlaceHolder[target=mm_196]
#   %getitem : Tensor "f32[512, 1][1, 1]cuda:0" = PlaceHolder[target=getitem]
#   %log : Tensor "f32[512, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_30 : Tensor "i64[][]cuda:0" = PlaceHolder[target=sum_30]
#   %sum_31 : Tensor "f32[][]cuda:0" = PlaceHolder[target=sum_31]
#   %convert_element_type_736 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_736]
#   %view_740 : Tensor "bf16[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 512, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %slice_116 : Tensor "i64[1, 512][513, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, 1, 1, 9223372036854775807), kwargs = {})
#   %view_741 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %view_742 : Tensor "i64[512][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_116, [-1]), kwargs = {})
#   %sub_tensor : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_741, %getitem), kwargs = {})
#   %sub_31 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne_1 : Tensor "b8[512][1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.ne.Scalar](args = (%view_742, -100), kwargs = {})
#   %full_default_3 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %view_742, %full_default_3), kwargs = {})
#   %unsqueeze_117 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where_1, 1), kwargs = {})
#   %gather : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_31, 1, %unsqueeze_117), kwargs = {})
#   %squeeze : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg_84 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_1, %neg_84, %full_default_4), kwargs = {})
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
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(6,))]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'r0_': 8192}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf2 = empty_strided_cuda((1, 513), (513, 1), torch.int64)
            buf0 = reinterpret_tensor(buf2, (1, 1), (513, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_slice_sub_unsqueeze_0.run(buf0, 1, stream=raw_stream0)
            buf1 = reinterpret_tensor(buf2, (1, 512), (513, 1), 1)  # alias
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem, first_dummy_value, position_diff], Original ATen: [aten.arange, aten.unsqueeze, aten.slice, aten.sub, aten.cat]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_arange_cat_slice_sub_unsqueeze_1.run(buf1, 512, stream=raw_stream0)
            buf3 = empty_strided_cuda((1, 512), (512, 1), torch.int64)
            # Topologically Sorted Source Nodes: [position_diff, ne, packed_sequence_mask], Original ATen: [aten.slice, aten.sub, aten.ne, aten.cumsum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_cumsum_ne_slice_sub_2.run(buf2, buf3, 1, 512, stream=raw_stream0)
            del buf0
            del buf1
            del buf2
            assert_size_stride(primals_1, (1, 512), (512, 1), 'input')
            assert_size_stride(primals_2, (128256, 3072), (3072, 1), 'input')
            assert_size_stride(primals_4, (3072, ), (1, ), 'input')
            primals_1 = copy_if_misaligned(primals_1)
            buf5 = empty_strided_cuda((1, 512, 3072), (1572864, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_6, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_3.run(primals_1, primals_2, primals_4, buf5, 512, 3072, stream=raw_stream0)
            del primals_4
            assert_size_stride(primals_5, (3072, 3072), (3072, 1), 'input')
            buf6 = empty_strided_cuda((512, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_6, hidden_states_2, linear], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_5, (3072, 3072), (1, 3072), 0), out=buf6)
            del primals_5
            assert_size_stride(primals_6, (1024, 3072), (3072, 1), 'input')
            buf7 = empty_strided_cuda((512, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_6, hidden_states_2, linear, linear_1], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_6, (3072, 1024), (1, 3072), 0), out=buf7)
            del primals_6
            assert_size_stride(primals_7, (1024, 3072), (3072, 1), 'input')
            buf8 = empty_strided_cuda((512, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add, rsqrt, hidden_states_1, to_6, hidden_states_2, linear, linear_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_7, (3072, 1024), (1, 3072), 0), out=buf8)
            del primals_7
            assert_size_stride(primals_3, (64, ), (1, ), 'input')
            buf9 = reinterpret_tensor(buf5, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf5  # reuse
            buf10 = empty_strided_cuda((1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear, view, query_states, linear_1, view_1, key_states, cos_3, sin_3, mul_4, x1, x2, neg, cat_1, mul_5, q_embed, mul_6, x1_1, x2_1, neg_1, cat_2, mul_7, k_embed, getitem_7, hidden_states_3, key_states_1], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf6, primals_3, buf7, buf9, buf10, 1572864, stream=raw_stream0)
            buf11 = empty_strided_cuda((24, 512, 512), (262144, 512, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, linear, view, query_states, linear_1, view_1, key_states, cos_3, sin_3, mul_4, x1, x2, neg, cat_1, mul_5, q_embed, mul_6, x1_1, x2_1, neg_1, cat_2, mul_7, k_embed, getitem_7, hidden_states_3, key_states_1, transpose_4, matmul_1], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf9, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf10, (24, 128, 512), (65536, 1, 128), 0), out=buf11)
            buf14 = reinterpret_tensor(buf11, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_1, attn_weights, attn_weights_1, softmax, attn_weights_2], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf14, buf3, 12288, 512, stream=raw_stream0)
            buf15 = reinterpret_tensor(buf9, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [linear_2, view_2, value_states, getitem_8, hidden_states_4, value_states_1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf8, buf15, 1572864, stream=raw_stream0)
            buf16 = reinterpret_tensor(buf10, (24, 512, 128), (65536, 128, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_2, view_2, value_states, getitem_8, hidden_states_4, value_states_1, matmul_1, attn_weights, attn_weights_1, softmax, attn_weights_2, attn_output], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf14, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf15, (24, 512, 128), (65536, 128, 1), 0), out=buf16)
            buf17 = reinterpret_tensor(buf15, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [attn_output, transpose_5, attn_output_1], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf16, buf17, 1572864, stream=raw_stream0)
            assert_size_stride(primals_8, (3072, 3072), (3072, 1), 'input')
            buf18 = reinterpret_tensor(buf16, (512, 3072), (3072, 1), 0); del buf16  # reuse
            # Topologically Sorted Source Nodes: [attn_output, transpose_5, attn_output_1, reshape_2, attn_output_3], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf17, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_8, (3072, 3072), (1, 3072), 0), out=buf18)
            del primals_8
            assert_size_stride(primals_9, (3072, ), (1, ), 'input')
            buf20 = reinterpret_tensor(buf17, (1, 512, 3072), (1572864, 3072, 1), 0); del buf17  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_5, rsqrt_1, hidden_states_7, to_9, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8.run(primals_1, primals_2, buf18, primals_9, buf20, 512, 3072, stream=raw_stream0)
            del primals_9
            assert_size_stride(primals_10, (8192, 3072), (3072, 1), 'input')
            buf21 = empty_strided_cuda((512, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_5, rsqrt_1, hidden_states_7, to_9, hidden_states_8, linear_4], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_10, (3072, 8192), (1, 3072), 0), out=buf21)
            del primals_10
            assert_size_stride(primals_11, (8192, 3072), (3072, 1), 'input')
            buf22 = empty_strided_cuda((512, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, hidden_states_6, pow_2, variance_1, add_5, rsqrt_1, hidden_states_7, to_9, hidden_states_8, linear_4, linear_5], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_11, (3072, 8192), (1, 3072), 0), out=buf22)
            del primals_11
            buf23 = reinterpret_tensor(buf21, (1, 512, 8192), (4194304, 8192, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf23, buf22, 4194304, stream=raw_stream0)
            assert_size_stride(primals_12, (3072, 8192), (8192, 1), 'input')
            buf24 = reinterpret_tensor(buf20, (512, 3072), (3072, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_11, down_proj], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf23, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_12, (8192, 3072), (1, 8192), 0), out=buf24)
            del primals_12
            assert_size_stride(primals_13, (3072, ), (1, ), 'input')
            buf26 = reinterpret_tensor(buf6, (1, 512, 3072), (1572864, 3072, 1), 0); del buf6  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, to_11, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10.run(primals_1, primals_2, buf18, buf24, primals_13, buf26, 512, 3072, stream=raw_stream0)
            del primals_13
            assert_size_stride(primals_14, (3072, 3072), (3072, 1), 'input')
            buf27 = empty_strided_cuda((512, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, to_11, hidden_states_12, linear_7], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_14, (3072, 3072), (1, 3072), 0), out=buf27)
            del primals_14
            assert_size_stride(primals_15, (1024, 3072), (3072, 1), 'input')
            buf28 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, to_11, hidden_states_12, linear_7, linear_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_15, (3072, 1024), (1, 3072), 0), out=buf28)
            del primals_15
            assert_size_stride(primals_16, (1024, 3072), (3072, 1), 'input')
            buf29 = buf7; del buf7  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, hidden_states_10, pow_3, variance_2, add_7, rsqrt_2, hidden_states_11, to_11, hidden_states_12, linear_7, linear_9], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf26, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_16, (3072, 1024), (1, 3072), 0), out=buf29)
            del primals_16
            buf30 = reinterpret_tensor(buf26, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf26  # reuse
            buf31 = empty_strided_cuda((1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_7, view_3, query_states_1, linear_8, view_4, key_states_2, mul_14, x1_2, x2_2, neg_2, cat_3, mul_15, q_embed_1, mul_16, x1_3, x2_3, neg_3, cat_4, mul_17, k_embed_1, getitem_14, hidden_states_13, key_states_3], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf27, primals_3, buf28, buf30, buf31, 1572864, stream=raw_stream0)
            buf32 = reinterpret_tensor(buf14, (24, 512, 512), (262144, 512, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_7, view_3, query_states_1, linear_8, view_4, key_states_2, mul_14, x1_2, x2_2, neg_2, cat_3, mul_15, q_embed_1, mul_16, x1_3, x2_3, neg_3, cat_4, mul_17, k_embed_1, getitem_14, hidden_states_13, key_states_3, transpose_9, matmul_3], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf30, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf31, (24, 128, 512), (65536, 1, 128), 0), out=buf32)
            buf35 = reinterpret_tensor(buf32, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_3, attn_weights_4, attn_weights_5, softmax_1, attn_weights_6], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf35, buf3, 12288, 512, stream=raw_stream0)
            buf36 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [linear_9, view_5, value_states_2, getitem_15, hidden_states_14, value_states_3], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf29, buf36, 1572864, stream=raw_stream0)
            buf37 = reinterpret_tensor(buf30, (24, 512, 128), (65536, 128, 1), 0); del buf30  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_9, view_5, value_states_2, getitem_15, hidden_states_14, value_states_3, matmul_3, attn_weights_4, attn_weights_5, softmax_1, attn_weights_6, attn_output_4], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf35, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf36, (24, 512, 128), (65536, 128, 1), 0), out=buf37)
            buf38 = reinterpret_tensor(buf36, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf36  # reuse
            # Topologically Sorted Source Nodes: [attn_output_4, transpose_10, attn_output_5], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf37, buf38, 1572864, stream=raw_stream0)
            assert_size_stride(primals_17, (3072, 3072), (3072, 1), 'input')
            buf39 = reinterpret_tensor(buf37, (512, 3072), (3072, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [attn_output_4, transpose_10, attn_output_5, reshape_5, attn_output_7], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf38, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_17, (3072, 3072), (1, 3072), 0), out=buf39)
            del primals_17
            assert_size_stride(primals_18, (3072, ), (1, ), 'input')
            buf40 = reinterpret_tensor(buf18, (1, 512, 3072), (1572864, 3072, 1), 0); del buf18  # reuse
            buf42 = reinterpret_tensor(buf38, (1, 512, 3072), (1572864, 3072, 1), 0); del buf38  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_5, down_proj, hidden_states_9, attn_output_7, hidden_states_15, hidden_states_16, pow_4, variance_3, add_12, rsqrt_3, hidden_states_17, to_14, hidden_states_18], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_11.run(buf40, primals_1, primals_2, buf24, buf39, primals_18, buf42, 512, 3072, stream=raw_stream0)
            del primals_18
            assert_size_stride(primals_19, (8192, 3072), (3072, 1), 'input')
            buf43 = reinterpret_tensor(buf23, (512, 8192), (8192, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_16, pow_4, variance_3, add_12, rsqrt_3, hidden_states_17, to_14, hidden_states_18, linear_11], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_19, (3072, 8192), (1, 3072), 0), out=buf43)
            del primals_19
            assert_size_stride(primals_20, (8192, 3072), (3072, 1), 'input')
            buf44 = buf22; del buf22  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_16, pow_4, variance_3, add_12, rsqrt_3, hidden_states_17, to_14, hidden_states_18, linear_11, linear_12], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf42, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_20, (3072, 8192), (1, 3072), 0), out=buf44)
            del primals_20
            buf45 = reinterpret_tensor(buf43, (1, 512, 8192), (4194304, 8192, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_21], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf45, buf44, 4194304, stream=raw_stream0)
            assert_size_stride(primals_21, (3072, 8192), (8192, 1), 'input')
            buf46 = reinterpret_tensor(buf42, (512, 3072), (3072, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_21, down_proj_1], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf45, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_21, (8192, 3072), (1, 8192), 0), out=buf46)
            del primals_21
            assert_size_stride(primals_22, (3072, ), (1, ), 'input')
            buf48 = reinterpret_tensor(buf39, (1, 512, 3072), (1572864, 3072, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, hidden_states_20, pow_5, variance_4, add_14, rsqrt_4, hidden_states_21, to_16, hidden_states_22], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf40, buf46, primals_22, buf48, 512, 3072, stream=raw_stream0)
            del primals_22
            assert_size_stride(primals_23, (3072, 3072), (3072, 1), 'input')
            buf49 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, hidden_states_20, pow_5, variance_4, add_14, rsqrt_4, hidden_states_21, to_16, hidden_states_22, linear_14], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_23, (3072, 3072), (1, 3072), 0), out=buf49)
            del primals_23
            assert_size_stride(primals_24, (1024, 3072), (3072, 1), 'input')
            buf50 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, hidden_states_20, pow_5, variance_4, add_14, rsqrt_4, hidden_states_21, to_16, hidden_states_22, linear_14, linear_15], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_24, (3072, 1024), (1, 3072), 0), out=buf50)
            del primals_24
            assert_size_stride(primals_25, (1024, 3072), (3072, 1), 'input')
            buf51 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, hidden_states_20, pow_5, variance_4, add_14, rsqrt_4, hidden_states_21, to_16, hidden_states_22, linear_14, linear_16], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_25, (3072, 1024), (1, 3072), 0), out=buf51)
            del primals_25
            buf52 = reinterpret_tensor(buf48, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf48  # reuse
            buf53 = reinterpret_tensor(buf27, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_14, view_6, query_states_2, linear_15, view_7, key_states_4, mul_24, x1_4, x2_4, neg_4, cat_5, mul_25, q_embed_2, mul_26, x1_5, x2_5, neg_5, cat_6, mul_27, k_embed_2, getitem_21, hidden_states_23, key_states_5], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf49, primals_3, buf50, buf52, buf53, 1572864, stream=raw_stream0)
            buf54 = reinterpret_tensor(buf35, (24, 512, 512), (262144, 512, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_14, view_6, query_states_2, linear_15, view_7, key_states_4, mul_24, x1_4, x2_4, neg_4, cat_5, mul_25, q_embed_2, mul_26, x1_5, x2_5, neg_5, cat_6, mul_27, k_embed_2, getitem_21, hidden_states_23, key_states_5, transpose_14, matmul_5], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf52, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf53, (24, 128, 512), (65536, 1, 128), 0), out=buf54)
            buf57 = reinterpret_tensor(buf54, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_5, attn_weights_8, attn_weights_9, softmax_2, attn_weights_10], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf57, buf3, 12288, 512, stream=raw_stream0)
            buf58 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [linear_16, view_8, value_states_4, getitem_22, hidden_states_24, value_states_5], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf51, buf58, 1572864, stream=raw_stream0)
            buf59 = reinterpret_tensor(buf52, (24, 512, 128), (65536, 128, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_16, view_8, value_states_4, getitem_22, hidden_states_24, value_states_5, matmul_5, attn_weights_8, attn_weights_9, softmax_2, attn_weights_10, attn_output_8], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf57, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf58, (24, 512, 128), (65536, 128, 1), 0), out=buf59)
            buf60 = reinterpret_tensor(buf58, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [attn_output_8, transpose_15, attn_output_9], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf59, buf60, 1572864, stream=raw_stream0)
            assert_size_stride(primals_26, (3072, 3072), (3072, 1), 'input')
            buf61 = reinterpret_tensor(buf59, (512, 3072), (3072, 1), 0); del buf59  # reuse
            # Topologically Sorted Source Nodes: [attn_output_8, transpose_15, attn_output_9, reshape_8, attn_output_11], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf60, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_26, (3072, 3072), (1, 3072), 0), out=buf61)
            del primals_26
            assert_size_stride(primals_27, (3072, ), (1, ), 'input')
            buf63 = reinterpret_tensor(buf60, (1, 512, 3072), (1572864, 3072, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, hidden_states_26, pow_6, variance_5, add_19, rsqrt_5, hidden_states_27, to_19, hidden_states_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf40, buf46, buf61, primals_27, buf63, 512, 3072, stream=raw_stream0)
            del primals_27
            assert_size_stride(primals_28, (8192, 3072), (3072, 1), 'input')
            buf64 = reinterpret_tensor(buf45, (512, 8192), (8192, 1), 0); del buf45  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, hidden_states_26, pow_6, variance_5, add_19, rsqrt_5, hidden_states_27, to_19, hidden_states_28, linear_18], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf63, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_28, (3072, 8192), (1, 3072), 0), out=buf64)
            del primals_28
            assert_size_stride(primals_29, (8192, 3072), (3072, 1), 'input')
            buf65 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, hidden_states_26, pow_6, variance_5, add_19, rsqrt_5, hidden_states_27, to_19, hidden_states_28, linear_18, linear_19], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf63, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_29, (3072, 8192), (1, 3072), 0), out=buf65)
            del primals_29
            buf66 = reinterpret_tensor(buf64, (1, 512, 8192), (4194304, 8192, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_31], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf66, buf65, 4194304, stream=raw_stream0)
            assert_size_stride(primals_30, (3072, 8192), (8192, 1), 'input')
            buf67 = reinterpret_tensor(buf63, (512, 3072), (3072, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_31, down_proj_2], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf66, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_30, (8192, 3072), (1, 8192), 0), out=buf67)
            del primals_30
            assert_size_stride(primals_31, (3072, ), (1, ), 'input')
            buf69 = reinterpret_tensor(buf49, (1, 512, 3072), (1572864, 3072, 1), 0); del buf49  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, hidden_states_30, pow_7, variance_6, add_21, rsqrt_6, hidden_states_31, to_21, hidden_states_32], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf40, buf46, buf61, buf67, primals_31, buf69, 512, 3072, stream=raw_stream0)
            del primals_31
            assert_size_stride(primals_32, (3072, 3072), (3072, 1), 'input')
            buf70 = empty_strided_cuda((512, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, hidden_states_30, pow_7, variance_6, add_21, rsqrt_6, hidden_states_31, to_21, hidden_states_32, linear_21], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_32, (3072, 3072), (1, 3072), 0), out=buf70)
            del primals_32
            assert_size_stride(primals_33, (1024, 3072), (3072, 1), 'input')
            buf71 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, hidden_states_30, pow_7, variance_6, add_21, rsqrt_6, hidden_states_31, to_21, hidden_states_32, linear_21, linear_22], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_33, (3072, 1024), (1, 3072), 0), out=buf71)
            del primals_33
            assert_size_stride(primals_34, (1024, 3072), (3072, 1), 'input')
            buf72 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, hidden_states_30, pow_7, variance_6, add_21, rsqrt_6, hidden_states_31, to_21, hidden_states_32, linear_21, linear_23], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf69, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_34, (3072, 1024), (1, 3072), 0), out=buf72)
            del primals_34
            buf73 = reinterpret_tensor(buf69, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf69  # reuse
            buf74 = empty_strided_cuda((1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_21, view_9, query_states_3, linear_22, view_10, key_states_6, mul_34, x1_6, x2_6, neg_6, cat_7, mul_35, q_embed_3, mul_36, x1_7, x2_7, neg_7, cat_8, mul_37, k_embed_3, getitem_28, hidden_states_33, key_states_7], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf70, primals_3, buf71, buf73, buf74, 1572864, stream=raw_stream0)
            buf75 = reinterpret_tensor(buf57, (24, 512, 512), (262144, 512, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_21, view_9, query_states_3, linear_22, view_10, key_states_6, mul_34, x1_6, x2_6, neg_6, cat_7, mul_35, q_embed_3, mul_36, x1_7, x2_7, neg_7, cat_8, mul_37, k_embed_3, getitem_28, hidden_states_33, key_states_7, transpose_19, matmul_7], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf73, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf74, (24, 128, 512), (65536, 1, 128), 0), out=buf75)
            buf78 = reinterpret_tensor(buf75, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_7, attn_weights_12, attn_weights_13, softmax_3, attn_weights_14], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf78, buf3, 12288, 512, stream=raw_stream0)
            buf79 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [linear_23, view_11, value_states_6, getitem_29, hidden_states_34, value_states_7], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf72, buf79, 1572864, stream=raw_stream0)
            buf80 = reinterpret_tensor(buf73, (24, 512, 128), (65536, 128, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_23, view_11, value_states_6, getitem_29, hidden_states_34, value_states_7, matmul_7, attn_weights_12, attn_weights_13, softmax_3, attn_weights_14, attn_output_12], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf78, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf79, (24, 512, 128), (65536, 128, 1), 0), out=buf80)
            buf81 = reinterpret_tensor(buf79, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, transpose_20, attn_output_13], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf80, buf81, 1572864, stream=raw_stream0)
            assert_size_stride(primals_35, (3072, 3072), (3072, 1), 'input')
            buf82 = reinterpret_tensor(buf80, (512, 3072), (3072, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [attn_output_12, transpose_20, attn_output_13, reshape_11, attn_output_15], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf81, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_35, (3072, 3072), (1, 3072), 0), out=buf82)
            del primals_35
            assert_size_stride(primals_36, (3072, ), (1, ), 'input')
            buf83 = buf40; del buf40  # reuse
            buf85 = reinterpret_tensor(buf81, (1, 512, 3072), (1572864, 3072, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_19, attn_output_11, hidden_states_25, down_proj_2, hidden_states_29, attn_output_15, hidden_states_35, hidden_states_36, pow_8, variance_7, add_26, rsqrt_7, hidden_states_37, to_24, hidden_states_38], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf83, buf46, buf61, buf67, buf82, primals_36, buf85, 512, 3072, stream=raw_stream0)
            del primals_36
            assert_size_stride(primals_37, (8192, 3072), (3072, 1), 'input')
            buf86 = reinterpret_tensor(buf66, (512, 8192), (8192, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_36, pow_8, variance_7, add_26, rsqrt_7, hidden_states_37, to_24, hidden_states_38, linear_25], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf85, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_37, (3072, 8192), (1, 3072), 0), out=buf86)
            del primals_37
            assert_size_stride(primals_38, (8192, 3072), (3072, 1), 'input')
            buf87 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_36, pow_8, variance_7, add_26, rsqrt_7, hidden_states_37, to_24, hidden_states_38, linear_25, linear_26], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf85, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_38, (3072, 8192), (1, 3072), 0), out=buf87)
            del primals_38
            buf88 = reinterpret_tensor(buf86, (1, 512, 8192), (4194304, 8192, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_41], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf88, buf87, 4194304, stream=raw_stream0)
            assert_size_stride(primals_39, (3072, 8192), (8192, 1), 'input')
            buf89 = reinterpret_tensor(buf85, (512, 3072), (3072, 1), 0); del buf85  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_41, down_proj_3], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf88, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_39, (8192, 3072), (1, 8192), 0), out=buf89)
            del primals_39
            assert_size_stride(primals_40, (3072, ), (1, ), 'input')
            buf91 = reinterpret_tensor(buf82, (1, 512, 3072), (1572864, 3072, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, hidden_states_40, pow_9, variance_8, add_28, rsqrt_8, hidden_states_41, to_26, hidden_states_42], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf83, buf89, primals_40, buf91, 512, 3072, stream=raw_stream0)
            del primals_40
            assert_size_stride(primals_41, (3072, 3072), (3072, 1), 'input')
            buf92 = buf67; del buf67  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, hidden_states_40, pow_9, variance_8, add_28, rsqrt_8, hidden_states_41, to_26, hidden_states_42, linear_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf91, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_41, (3072, 3072), (1, 3072), 0), out=buf92)
            del primals_41
            assert_size_stride(primals_42, (1024, 3072), (3072, 1), 'input')
            buf93 = buf72; del buf72  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, hidden_states_40, pow_9, variance_8, add_28, rsqrt_8, hidden_states_41, to_26, hidden_states_42, linear_28, linear_29], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf91, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_42, (3072, 1024), (1, 3072), 0), out=buf93)
            del primals_42
            assert_size_stride(primals_43, (1024, 3072), (3072, 1), 'input')
            buf94 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, hidden_states_40, pow_9, variance_8, add_28, rsqrt_8, hidden_states_41, to_26, hidden_states_42, linear_28, linear_30], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf91, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_43, (3072, 1024), (1, 3072), 0), out=buf94)
            del primals_43
            buf95 = reinterpret_tensor(buf91, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf91  # reuse
            buf96 = reinterpret_tensor(buf61, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_28, view_12, query_states_4, linear_29, view_13, key_states_8, mul_44, x1_8, x2_8, neg_8, cat_9, mul_45, q_embed_4, mul_46, x1_9, x2_9, neg_9, cat_10, mul_47, k_embed_4, getitem_35, hidden_states_43, key_states_9], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf92, primals_3, buf93, buf95, buf96, 1572864, stream=raw_stream0)
            buf97 = reinterpret_tensor(buf78, (24, 512, 512), (262144, 512, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_28, view_12, query_states_4, linear_29, view_13, key_states_8, mul_44, x1_8, x2_8, neg_8, cat_9, mul_45, q_embed_4, mul_46, x1_9, x2_9, neg_9, cat_10, mul_47, k_embed_4, getitem_35, hidden_states_43, key_states_9, transpose_24, matmul_9], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf95, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf96, (24, 128, 512), (65536, 1, 128), 0), out=buf97)
            buf100 = reinterpret_tensor(buf97, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf97  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_9, attn_weights_16, attn_weights_17, softmax_4, attn_weights_18], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf100, buf3, 12288, 512, stream=raw_stream0)
            buf101 = buf96; del buf96  # reuse
            # Topologically Sorted Source Nodes: [linear_30, view_14, value_states_8, getitem_36, hidden_states_44, value_states_9], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf94, buf101, 1572864, stream=raw_stream0)
            buf102 = reinterpret_tensor(buf95, (24, 512, 128), (65536, 128, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_30, view_14, value_states_8, getitem_36, hidden_states_44, value_states_9, matmul_9, attn_weights_16, attn_weights_17, softmax_4, attn_weights_18, attn_output_16], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf100, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf101, (24, 512, 128), (65536, 128, 1), 0), out=buf102)
            buf103 = reinterpret_tensor(buf101, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [attn_output_16, transpose_25, attn_output_17], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf102, buf103, 1572864, stream=raw_stream0)
            assert_size_stride(primals_44, (3072, 3072), (3072, 1), 'input')
            buf104 = reinterpret_tensor(buf102, (512, 3072), (3072, 1), 0); del buf102  # reuse
            # Topologically Sorted Source Nodes: [attn_output_16, transpose_25, attn_output_17, reshape_14, attn_output_19], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf103, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_44, (3072, 3072), (1, 3072), 0), out=buf104)
            del primals_44
            assert_size_stride(primals_45, (3072, ), (1, ), 'input')
            buf106 = reinterpret_tensor(buf103, (1, 512, 3072), (1572864, 3072, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, hidden_states_46, pow_10, variance_9, add_33, rsqrt_9, hidden_states_47, to_29, hidden_states_48], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf83, buf89, buf104, primals_45, buf106, 512, 3072, stream=raw_stream0)
            del primals_45
            assert_size_stride(primals_46, (8192, 3072), (3072, 1), 'input')
            buf107 = reinterpret_tensor(buf88, (512, 8192), (8192, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, hidden_states_46, pow_10, variance_9, add_33, rsqrt_9, hidden_states_47, to_29, hidden_states_48, linear_32], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf106, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_46, (3072, 8192), (1, 3072), 0), out=buf107)
            del primals_46
            assert_size_stride(primals_47, (8192, 3072), (3072, 1), 'input')
            buf108 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, hidden_states_46, pow_10, variance_9, add_33, rsqrt_9, hidden_states_47, to_29, hidden_states_48, linear_32, linear_33], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf106, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_47, (3072, 8192), (1, 3072), 0), out=buf108)
            del primals_47
            buf109 = reinterpret_tensor(buf107, (1, 512, 8192), (4194304, 8192, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_51], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf109, buf108, 4194304, stream=raw_stream0)
            assert_size_stride(primals_48, (3072, 8192), (8192, 1), 'input')
            buf110 = reinterpret_tensor(buf106, (512, 3072), (3072, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_51, down_proj_4], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf109, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_48, (8192, 3072), (1, 8192), 0), out=buf110)
            del primals_48
            assert_size_stride(primals_49, (3072, ), (1, ), 'input')
            buf112 = reinterpret_tensor(buf92, (1, 512, 3072), (1572864, 3072, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, hidden_states_50, pow_11, variance_10, add_35, rsqrt_10, hidden_states_51, to_31, hidden_states_52], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf83, buf89, buf104, buf110, primals_49, buf112, 512, 3072, stream=raw_stream0)
            del primals_49
            assert_size_stride(primals_50, (3072, 3072), (3072, 1), 'input')
            buf113 = buf46; del buf46  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, hidden_states_50, pow_11, variance_10, add_35, rsqrt_10, hidden_states_51, to_31, hidden_states_52, linear_35], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf112, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_50, (3072, 3072), (1, 3072), 0), out=buf113)
            del primals_50
            assert_size_stride(primals_51, (1024, 3072), (3072, 1), 'input')
            buf114 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, hidden_states_50, pow_11, variance_10, add_35, rsqrt_10, hidden_states_51, to_31, hidden_states_52, linear_35, linear_36], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf112, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_51, (3072, 1024), (1, 3072), 0), out=buf114)
            del primals_51
            assert_size_stride(primals_52, (1024, 3072), (3072, 1), 'input')
            buf115 = buf93; del buf93  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, hidden_states_50, pow_11, variance_10, add_35, rsqrt_10, hidden_states_51, to_31, hidden_states_52, linear_35, linear_37], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf112, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_52, (3072, 1024), (1, 3072), 0), out=buf115)
            del primals_52
            buf116 = reinterpret_tensor(buf112, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf112  # reuse
            buf117 = reinterpret_tensor(buf70, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_35, view_15, query_states_5, linear_36, view_16, key_states_10, mul_54, x1_10, x2_10, neg_10, cat_11, mul_55, q_embed_5, mul_56, x1_11, x2_11, neg_11, cat_12, mul_57, k_embed_5, getitem_42, hidden_states_53, key_states_11], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf113, primals_3, buf114, buf116, buf117, 1572864, stream=raw_stream0)
            buf118 = reinterpret_tensor(buf100, (24, 512, 512), (262144, 512, 1), 0); del buf100  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_35, view_15, query_states_5, linear_36, view_16, key_states_10, mul_54, x1_10, x2_10, neg_10, cat_11, mul_55, q_embed_5, mul_56, x1_11, x2_11, neg_11, cat_12, mul_57, k_embed_5, getitem_42, hidden_states_53, key_states_11, transpose_29, matmul_11], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf116, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf117, (24, 128, 512), (65536, 1, 128), 0), out=buf118)
            buf121 = reinterpret_tensor(buf118, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_11, attn_weights_20, attn_weights_21, softmax_5, attn_weights_22], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf121, buf3, 12288, 512, stream=raw_stream0)
            buf122 = buf117; del buf117  # reuse
            # Topologically Sorted Source Nodes: [linear_37, view_17, value_states_10, getitem_43, hidden_states_54, value_states_11], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf115, buf122, 1572864, stream=raw_stream0)
            buf123 = reinterpret_tensor(buf116, (24, 512, 128), (65536, 128, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_37, view_17, value_states_10, getitem_43, hidden_states_54, value_states_11, matmul_11, attn_weights_20, attn_weights_21, softmax_5, attn_weights_22, attn_output_20], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf121, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf122, (24, 512, 128), (65536, 128, 1), 0), out=buf123)
            buf124 = reinterpret_tensor(buf122, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [attn_output_20, transpose_30, attn_output_21], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf123, buf124, 1572864, stream=raw_stream0)
            assert_size_stride(primals_53, (3072, 3072), (3072, 1), 'input')
            buf125 = reinterpret_tensor(buf123, (512, 3072), (3072, 1), 0); del buf123  # reuse
            # Topologically Sorted Source Nodes: [attn_output_20, transpose_30, attn_output_21, reshape_17, attn_output_23], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf124, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_53, (3072, 3072), (1, 3072), 0), out=buf125)
            del primals_53
            assert_size_stride(primals_54, (3072, ), (1, ), 'input')
            buf126 = buf83; del buf83  # reuse
            buf128 = reinterpret_tensor(buf124, (1, 512, 3072), (1572864, 3072, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_39, attn_output_19, hidden_states_45, down_proj_4, hidden_states_49, attn_output_23, hidden_states_55, hidden_states_56, pow_12, variance_11, add_40, rsqrt_11, hidden_states_57, to_34, hidden_states_58], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf126, buf89, buf104, buf110, buf125, primals_54, buf128, 512, 3072, stream=raw_stream0)
            del primals_54
            assert_size_stride(primals_55, (8192, 3072), (3072, 1), 'input')
            buf129 = reinterpret_tensor(buf109, (512, 8192), (8192, 1), 0); del buf109  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_56, pow_12, variance_11, add_40, rsqrt_11, hidden_states_57, to_34, hidden_states_58, linear_39], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf128, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_55, (3072, 8192), (1, 3072), 0), out=buf129)
            del primals_55
            assert_size_stride(primals_56, (8192, 3072), (3072, 1), 'input')
            buf130 = buf108; del buf108  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_56, pow_12, variance_11, add_40, rsqrt_11, hidden_states_57, to_34, hidden_states_58, linear_39, linear_40], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf128, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_56, (3072, 8192), (1, 3072), 0), out=buf130)
            del primals_56
            buf131 = reinterpret_tensor(buf129, (1, 512, 8192), (4194304, 8192, 1), 0); del buf129  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_61], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf131, buf130, 4194304, stream=raw_stream0)
            assert_size_stride(primals_57, (3072, 8192), (8192, 1), 'input')
            buf132 = reinterpret_tensor(buf128, (512, 3072), (3072, 1), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_61, down_proj_5], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf131, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_57, (8192, 3072), (1, 8192), 0), out=buf132)
            del primals_57
            assert_size_stride(primals_58, (3072, ), (1, ), 'input')
            buf134 = reinterpret_tensor(buf89, (1, 512, 3072), (1572864, 3072, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, hidden_states_60, pow_13, variance_12, add_42, rsqrt_12, hidden_states_61, to_36, hidden_states_62], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf126, buf132, primals_58, buf134, 512, 3072, stream=raw_stream0)
            del primals_58
            assert_size_stride(primals_59, (3072, 3072), (3072, 1), 'input')
            buf135 = buf125; del buf125  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, hidden_states_60, pow_13, variance_12, add_42, rsqrt_12, hidden_states_61, to_36, hidden_states_62, linear_42], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf134, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 3072), (1, 3072), 0), out=buf135)
            del primals_59
            assert_size_stride(primals_60, (1024, 3072), (3072, 1), 'input')
            buf136 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, hidden_states_60, pow_13, variance_12, add_42, rsqrt_12, hidden_states_61, to_36, hidden_states_62, linear_42, linear_43], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf134, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_60, (3072, 1024), (1, 3072), 0), out=buf136)
            del primals_60
            assert_size_stride(primals_61, (1024, 3072), (3072, 1), 'input')
            buf137 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, hidden_states_60, pow_13, variance_12, add_42, rsqrt_12, hidden_states_61, to_36, hidden_states_62, linear_42, linear_44], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf134, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_61, (3072, 1024), (1, 3072), 0), out=buf137)
            del primals_61
            buf138 = reinterpret_tensor(buf134, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf134  # reuse
            buf139 = reinterpret_tensor(buf110, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf110  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_42, view_18, query_states_6, linear_43, view_19, key_states_12, mul_64, x1_12, x2_12, neg_12, cat_13, mul_65, q_embed_6, mul_66, x1_13, x2_13, neg_13, cat_14, mul_67, k_embed_6, getitem_49, hidden_states_63, key_states_13], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf135, primals_3, buf136, buf138, buf139, 1572864, stream=raw_stream0)
            buf140 = reinterpret_tensor(buf121, (24, 512, 512), (262144, 512, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_42, view_18, query_states_6, linear_43, view_19, key_states_12, mul_64, x1_12, x2_12, neg_12, cat_13, mul_65, q_embed_6, mul_66, x1_13, x2_13, neg_13, cat_14, mul_67, k_embed_6, getitem_49, hidden_states_63, key_states_13, transpose_34, matmul_13], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf138, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf139, (24, 128, 512), (65536, 1, 128), 0), out=buf140)
            buf143 = reinterpret_tensor(buf140, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_13, attn_weights_24, attn_weights_25, softmax_6, attn_weights_26], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf143, buf3, 12288, 512, stream=raw_stream0)
            buf144 = buf139; del buf139  # reuse
            # Topologically Sorted Source Nodes: [linear_44, view_20, value_states_12, getitem_50, hidden_states_64, value_states_13], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf137, buf144, 1572864, stream=raw_stream0)
            buf145 = reinterpret_tensor(buf138, (24, 512, 128), (65536, 128, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_44, view_20, value_states_12, getitem_50, hidden_states_64, value_states_13, matmul_13, attn_weights_24, attn_weights_25, softmax_6, attn_weights_26, attn_output_24], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf143, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf144, (24, 512, 128), (65536, 128, 1), 0), out=buf145)
            buf146 = reinterpret_tensor(buf144, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf144  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, transpose_35, attn_output_25], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf145, buf146, 1572864, stream=raw_stream0)
            assert_size_stride(primals_62, (3072, 3072), (3072, 1), 'input')
            buf147 = reinterpret_tensor(buf145, (512, 3072), (3072, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [attn_output_24, transpose_35, attn_output_25, reshape_20, attn_output_27], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf146, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_62, (3072, 3072), (1, 3072), 0), out=buf147)
            del primals_62
            assert_size_stride(primals_63, (3072, ), (1, ), 'input')
            buf149 = reinterpret_tensor(buf146, (1, 512, 3072), (1572864, 3072, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, hidden_states_66, pow_14, variance_13, add_47, rsqrt_13, hidden_states_67, to_39, hidden_states_68], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf126, buf132, buf147, primals_63, buf149, 512, 3072, stream=raw_stream0)
            del primals_63
            assert_size_stride(primals_64, (8192, 3072), (3072, 1), 'input')
            buf150 = reinterpret_tensor(buf131, (512, 8192), (8192, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, hidden_states_66, pow_14, variance_13, add_47, rsqrt_13, hidden_states_67, to_39, hidden_states_68, linear_46], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_64, (3072, 8192), (1, 3072), 0), out=buf150)
            del primals_64
            assert_size_stride(primals_65, (8192, 3072), (3072, 1), 'input')
            buf151 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, hidden_states_66, pow_14, variance_13, add_47, rsqrt_13, hidden_states_67, to_39, hidden_states_68, linear_46, linear_47], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_65, (3072, 8192), (1, 3072), 0), out=buf151)
            del primals_65
            buf152 = reinterpret_tensor(buf150, (1, 512, 8192), (4194304, 8192, 1), 0); del buf150  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_71], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf152, buf151, 4194304, stream=raw_stream0)
            assert_size_stride(primals_66, (3072, 8192), (8192, 1), 'input')
            buf153 = reinterpret_tensor(buf149, (512, 3072), (3072, 1), 0); del buf149  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_71, down_proj_6], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf152, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_66, (8192, 3072), (1, 8192), 0), out=buf153)
            del primals_66
            assert_size_stride(primals_67, (3072, ), (1, ), 'input')
            buf155 = reinterpret_tensor(buf135, (1, 512, 3072), (1572864, 3072, 1), 0); del buf135  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, hidden_states_70, pow_15, variance_14, add_49, rsqrt_14, hidden_states_71, to_41, hidden_states_72], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf126, buf132, buf147, buf153, primals_67, buf155, 512, 3072, stream=raw_stream0)
            del primals_67
            assert_size_stride(primals_68, (3072, 3072), (3072, 1), 'input')
            buf156 = buf104; del buf104  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, hidden_states_70, pow_15, variance_14, add_49, rsqrt_14, hidden_states_71, to_41, hidden_states_72, linear_49], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf155, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_68, (3072, 3072), (1, 3072), 0), out=buf156)
            del primals_68
            assert_size_stride(primals_69, (1024, 3072), (3072, 1), 'input')
            buf157 = buf137; del buf137  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, hidden_states_70, pow_15, variance_14, add_49, rsqrt_14, hidden_states_71, to_41, hidden_states_72, linear_49, linear_50], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf155, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_69, (3072, 1024), (1, 3072), 0), out=buf157)
            del primals_69
            assert_size_stride(primals_70, (1024, 3072), (3072, 1), 'input')
            buf158 = buf136; del buf136  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, hidden_states_70, pow_15, variance_14, add_49, rsqrt_14, hidden_states_71, to_41, hidden_states_72, linear_49, linear_51], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf155, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_70, (3072, 1024), (1, 3072), 0), out=buf158)
            del primals_70
            buf159 = reinterpret_tensor(buf155, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf155  # reuse
            buf160 = reinterpret_tensor(buf113, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_49, view_21, query_states_7, linear_50, view_22, key_states_14, mul_74, x1_14, x2_14, neg_14, cat_15, mul_75, q_embed_7, mul_76, x1_15, x2_15, neg_15, cat_16, mul_77, k_embed_7, getitem_56, hidden_states_73, key_states_15], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf156, primals_3, buf157, buf159, buf160, 1572864, stream=raw_stream0)
            buf161 = reinterpret_tensor(buf143, (24, 512, 512), (262144, 512, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_49, view_21, query_states_7, linear_50, view_22, key_states_14, mul_74, x1_14, x2_14, neg_14, cat_15, mul_75, q_embed_7, mul_76, x1_15, x2_15, neg_15, cat_16, mul_77, k_embed_7, getitem_56, hidden_states_73, key_states_15, transpose_39, matmul_15], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf159, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf160, (24, 128, 512), (65536, 1, 128), 0), out=buf161)
            buf164 = reinterpret_tensor(buf161, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf161  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_15, attn_weights_28, attn_weights_29, softmax_7, attn_weights_30], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf164, buf3, 12288, 512, stream=raw_stream0)
            buf165 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [linear_51, view_23, value_states_14, getitem_57, hidden_states_74, value_states_15], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf158, buf165, 1572864, stream=raw_stream0)
            buf166 = reinterpret_tensor(buf159, (24, 512, 128), (65536, 128, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_51, view_23, value_states_14, getitem_57, hidden_states_74, value_states_15, matmul_15, attn_weights_28, attn_weights_29, softmax_7, attn_weights_30, attn_output_28], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf164, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf165, (24, 512, 128), (65536, 128, 1), 0), out=buf166)
            buf167 = reinterpret_tensor(buf165, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [attn_output_28, transpose_40, attn_output_29], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf166, buf167, 1572864, stream=raw_stream0)
            assert_size_stride(primals_71, (3072, 3072), (3072, 1), 'input')
            buf168 = reinterpret_tensor(buf166, (512, 3072), (3072, 1), 0); del buf166  # reuse
            # Topologically Sorted Source Nodes: [attn_output_28, transpose_40, attn_output_29, reshape_23, attn_output_31], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf167, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 3072), (1, 3072), 0), out=buf168)
            del primals_71
            assert_size_stride(primals_72, (3072, ), (1, ), 'input')
            buf169 = buf126; del buf126  # reuse
            buf171 = reinterpret_tensor(buf167, (1, 512, 3072), (1572864, 3072, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_59, attn_output_27, hidden_states_65, down_proj_6, hidden_states_69, attn_output_31, hidden_states_75, hidden_states_76, pow_16, variance_15, add_54, rsqrt_15, hidden_states_77, to_44, hidden_states_78], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf169, buf132, buf147, buf153, buf168, primals_72, buf171, 512, 3072, stream=raw_stream0)
            del primals_72
            assert_size_stride(primals_73, (8192, 3072), (3072, 1), 'input')
            buf172 = reinterpret_tensor(buf152, (512, 8192), (8192, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_76, pow_16, variance_15, add_54, rsqrt_15, hidden_states_77, to_44, hidden_states_78, linear_53], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf171, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_73, (3072, 8192), (1, 3072), 0), out=buf172)
            del primals_73
            assert_size_stride(primals_74, (8192, 3072), (3072, 1), 'input')
            buf173 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_76, pow_16, variance_15, add_54, rsqrt_15, hidden_states_77, to_44, hidden_states_78, linear_53, linear_54], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf171, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_74, (3072, 8192), (1, 3072), 0), out=buf173)
            del primals_74
            buf174 = reinterpret_tensor(buf172, (1, 512, 8192), (4194304, 8192, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_81], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf174, buf173, 4194304, stream=raw_stream0)
            assert_size_stride(primals_75, (3072, 8192), (8192, 1), 'input')
            buf175 = reinterpret_tensor(buf171, (512, 3072), (3072, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_81, down_proj_7], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf174, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_75, (8192, 3072), (1, 8192), 0), out=buf175)
            del primals_75
            assert_size_stride(primals_76, (3072, ), (1, ), 'input')
            buf177 = reinterpret_tensor(buf168, (1, 512, 3072), (1572864, 3072, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, hidden_states_80, pow_17, variance_16, add_56, rsqrt_16, hidden_states_81, to_46, hidden_states_82], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf169, buf175, primals_76, buf177, 512, 3072, stream=raw_stream0)
            del primals_76
            assert_size_stride(primals_77, (3072, 3072), (3072, 1), 'input')
            buf178 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, hidden_states_80, pow_17, variance_16, add_56, rsqrt_16, hidden_states_81, to_46, hidden_states_82, linear_56], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf177, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_77, (3072, 3072), (1, 3072), 0), out=buf178)
            del primals_77
            assert_size_stride(primals_78, (1024, 3072), (3072, 1), 'input')
            buf179 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, hidden_states_80, pow_17, variance_16, add_56, rsqrt_16, hidden_states_81, to_46, hidden_states_82, linear_56, linear_57], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf177, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_78, (3072, 1024), (1, 3072), 0), out=buf179)
            del primals_78
            assert_size_stride(primals_79, (1024, 3072), (3072, 1), 'input')
            buf180 = buf157; del buf157  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, hidden_states_80, pow_17, variance_16, add_56, rsqrt_16, hidden_states_81, to_46, hidden_states_82, linear_56, linear_58], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf177, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_79, (3072, 1024), (1, 3072), 0), out=buf180)
            del primals_79
            buf181 = reinterpret_tensor(buf177, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf177  # reuse
            buf182 = reinterpret_tensor(buf147, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf147  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_56, view_24, query_states_8, linear_57, view_25, key_states_16, mul_84, x1_16, x2_16, neg_16, cat_17, mul_85, q_embed_8, mul_86, x1_17, x2_17, neg_17, cat_18, mul_87, k_embed_8, getitem_63, hidden_states_83, key_states_17], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf178, primals_3, buf179, buf181, buf182, 1572864, stream=raw_stream0)
            buf183 = reinterpret_tensor(buf164, (24, 512, 512), (262144, 512, 1), 0); del buf164  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_56, view_24, query_states_8, linear_57, view_25, key_states_16, mul_84, x1_16, x2_16, neg_16, cat_17, mul_85, q_embed_8, mul_86, x1_17, x2_17, neg_17, cat_18, mul_87, k_embed_8, getitem_63, hidden_states_83, key_states_17, transpose_44, matmul_17], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf181, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf182, (24, 128, 512), (65536, 1, 128), 0), out=buf183)
            buf186 = reinterpret_tensor(buf183, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_17, attn_weights_32, attn_weights_33, softmax_8, attn_weights_34], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf186, buf3, 12288, 512, stream=raw_stream0)
            buf187 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [linear_58, view_26, value_states_16, getitem_64, hidden_states_84, value_states_17], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf180, buf187, 1572864, stream=raw_stream0)
            buf188 = reinterpret_tensor(buf181, (24, 512, 128), (65536, 128, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_58, view_26, value_states_16, getitem_64, hidden_states_84, value_states_17, matmul_17, attn_weights_32, attn_weights_33, softmax_8, attn_weights_34, attn_output_32], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf186, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf187, (24, 512, 128), (65536, 128, 1), 0), out=buf188)
            buf189 = reinterpret_tensor(buf187, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf187  # reuse
            # Topologically Sorted Source Nodes: [attn_output_32, transpose_45, attn_output_33], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf188, buf189, 1572864, stream=raw_stream0)
            assert_size_stride(primals_80, (3072, 3072), (3072, 1), 'input')
            buf190 = reinterpret_tensor(buf188, (512, 3072), (3072, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [attn_output_32, transpose_45, attn_output_33, reshape_26, attn_output_35], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf189, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_80, (3072, 3072), (1, 3072), 0), out=buf190)
            del primals_80
            assert_size_stride(primals_81, (3072, ), (1, ), 'input')
            buf192 = reinterpret_tensor(buf189, (1, 512, 3072), (1572864, 3072, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, hidden_states_86, pow_18, variance_17, add_61, rsqrt_17, hidden_states_87, to_49, hidden_states_88], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf169, buf175, buf190, primals_81, buf192, 512, 3072, stream=raw_stream0)
            del primals_81
            assert_size_stride(primals_82, (8192, 3072), (3072, 1), 'input')
            buf193 = reinterpret_tensor(buf174, (512, 8192), (8192, 1), 0); del buf174  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, hidden_states_86, pow_18, variance_17, add_61, rsqrt_17, hidden_states_87, to_49, hidden_states_88, linear_60], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf192, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_82, (3072, 8192), (1, 3072), 0), out=buf193)
            del primals_82
            assert_size_stride(primals_83, (8192, 3072), (3072, 1), 'input')
            buf194 = buf173; del buf173  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, hidden_states_86, pow_18, variance_17, add_61, rsqrt_17, hidden_states_87, to_49, hidden_states_88, linear_60, linear_61], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf192, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 8192), (1, 3072), 0), out=buf194)
            del primals_83
            buf195 = reinterpret_tensor(buf193, (1, 512, 8192), (4194304, 8192, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_91], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf195, buf194, 4194304, stream=raw_stream0)
            assert_size_stride(primals_84, (3072, 8192), (8192, 1), 'input')
            buf196 = reinterpret_tensor(buf192, (512, 3072), (3072, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_91, down_proj_8], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf195, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_84, (8192, 3072), (1, 8192), 0), out=buf196)
            del primals_84
            assert_size_stride(primals_85, (3072, ), (1, ), 'input')
            buf198 = reinterpret_tensor(buf178, (1, 512, 3072), (1572864, 3072, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, hidden_states_90, pow_19, variance_18, add_63, rsqrt_18, hidden_states_91, to_51, hidden_states_92], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf169, buf175, buf190, buf196, primals_85, buf198, 512, 3072, stream=raw_stream0)
            del primals_85
            assert_size_stride(primals_86, (3072, 3072), (3072, 1), 'input')
            buf199 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, hidden_states_90, pow_19, variance_18, add_63, rsqrt_18, hidden_states_91, to_51, hidden_states_92, linear_63], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf198, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_86, (3072, 3072), (1, 3072), 0), out=buf199)
            del primals_86
            assert_size_stride(primals_87, (1024, 3072), (3072, 1), 'input')
            buf200 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, hidden_states_90, pow_19, variance_18, add_63, rsqrt_18, hidden_states_91, to_51, hidden_states_92, linear_63, linear_64], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf198, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_87, (3072, 1024), (1, 3072), 0), out=buf200)
            del primals_87
            assert_size_stride(primals_88, (1024, 3072), (3072, 1), 'input')
            buf201 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, hidden_states_90, pow_19, variance_18, add_63, rsqrt_18, hidden_states_91, to_51, hidden_states_92, linear_63, linear_65], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf198, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_88, (3072, 1024), (1, 3072), 0), out=buf201)
            del primals_88
            buf202 = reinterpret_tensor(buf198, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf198  # reuse
            buf203 = reinterpret_tensor(buf156, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_63, view_27, query_states_9, linear_64, view_28, key_states_18, mul_94, x1_18, x2_18, neg_18, cat_19, mul_95, q_embed_9, mul_96, x1_19, x2_19, neg_19, cat_20, mul_97, k_embed_9, getitem_70, hidden_states_93, key_states_19], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf199, primals_3, buf200, buf202, buf203, 1572864, stream=raw_stream0)
            buf204 = reinterpret_tensor(buf186, (24, 512, 512), (262144, 512, 1), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_63, view_27, query_states_9, linear_64, view_28, key_states_18, mul_94, x1_18, x2_18, neg_18, cat_19, mul_95, q_embed_9, mul_96, x1_19, x2_19, neg_19, cat_20, mul_97, k_embed_9, getitem_70, hidden_states_93, key_states_19, transpose_49, matmul_19], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf202, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf203, (24, 128, 512), (65536, 1, 128), 0), out=buf204)
            buf207 = reinterpret_tensor(buf204, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_19, attn_weights_36, attn_weights_37, softmax_9, attn_weights_38], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf207, buf3, 12288, 512, stream=raw_stream0)
            buf208 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [linear_65, view_29, value_states_18, getitem_71, hidden_states_94, value_states_19], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf201, buf208, 1572864, stream=raw_stream0)
            buf209 = reinterpret_tensor(buf202, (24, 512, 128), (65536, 128, 1), 0); del buf202  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_65, view_29, value_states_18, getitem_71, hidden_states_94, value_states_19, matmul_19, attn_weights_36, attn_weights_37, softmax_9, attn_weights_38, attn_output_36], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf207, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf208, (24, 512, 128), (65536, 128, 1), 0), out=buf209)
            buf210 = reinterpret_tensor(buf208, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf208  # reuse
            # Topologically Sorted Source Nodes: [attn_output_36, transpose_50, attn_output_37], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf209, buf210, 1572864, stream=raw_stream0)
            assert_size_stride(primals_89, (3072, 3072), (3072, 1), 'input')
            buf211 = reinterpret_tensor(buf209, (512, 3072), (3072, 1), 0); del buf209  # reuse
            # Topologically Sorted Source Nodes: [attn_output_36, transpose_50, attn_output_37, reshape_29, attn_output_39], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf210, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_89, (3072, 3072), (1, 3072), 0), out=buf211)
            del primals_89
            assert_size_stride(primals_90, (3072, ), (1, ), 'input')
            buf212 = buf169; del buf169  # reuse
            buf214 = reinterpret_tensor(buf210, (1, 512, 3072), (1572864, 3072, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_79, attn_output_35, hidden_states_85, down_proj_8, hidden_states_89, attn_output_39, hidden_states_95, hidden_states_96, pow_20, variance_19, add_68, rsqrt_19, hidden_states_97, to_54, hidden_states_98], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf212, buf175, buf190, buf196, buf211, primals_90, buf214, 512, 3072, stream=raw_stream0)
            del primals_90
            assert_size_stride(primals_91, (8192, 3072), (3072, 1), 'input')
            buf215 = reinterpret_tensor(buf195, (512, 8192), (8192, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_96, pow_20, variance_19, add_68, rsqrt_19, hidden_states_97, to_54, hidden_states_98, linear_67], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 8192), (1, 3072), 0), out=buf215)
            del primals_91
            assert_size_stride(primals_92, (8192, 3072), (3072, 1), 'input')
            buf216 = buf194; del buf194  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_96, pow_20, variance_19, add_68, rsqrt_19, hidden_states_97, to_54, hidden_states_98, linear_67, linear_68], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_92, (3072, 8192), (1, 3072), 0), out=buf216)
            del primals_92
            buf217 = reinterpret_tensor(buf215, (1, 512, 8192), (4194304, 8192, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_101], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf217, buf216, 4194304, stream=raw_stream0)
            assert_size_stride(primals_93, (3072, 8192), (8192, 1), 'input')
            buf218 = reinterpret_tensor(buf214, (512, 3072), (3072, 1), 0); del buf214  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_101, down_proj_9], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf217, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_93, (8192, 3072), (1, 8192), 0), out=buf218)
            del primals_93
            assert_size_stride(primals_94, (3072, ), (1, ), 'input')
            buf220 = reinterpret_tensor(buf211, (1, 512, 3072), (1572864, 3072, 1), 0); del buf211  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, hidden_states_100, pow_21, variance_20, add_70, rsqrt_20, hidden_states_101, to_56, hidden_states_102], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf212, buf218, primals_94, buf220, 512, 3072, stream=raw_stream0)
            del primals_94
            assert_size_stride(primals_95, (3072, 3072), (3072, 1), 'input')
            buf221 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, hidden_states_100, pow_21, variance_20, add_70, rsqrt_20, hidden_states_101, to_56, hidden_states_102, linear_70], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf220, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 3072), (1, 3072), 0), out=buf221)
            del primals_95
            assert_size_stride(primals_96, (1024, 3072), (3072, 1), 'input')
            buf222 = buf201; del buf201  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, hidden_states_100, pow_21, variance_20, add_70, rsqrt_20, hidden_states_101, to_56, hidden_states_102, linear_70, linear_71], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf220, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_96, (3072, 1024), (1, 3072), 0), out=buf222)
            del primals_96
            assert_size_stride(primals_97, (1024, 3072), (3072, 1), 'input')
            buf223 = buf200; del buf200  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, hidden_states_100, pow_21, variance_20, add_70, rsqrt_20, hidden_states_101, to_56, hidden_states_102, linear_70, linear_72], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf220, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_97, (3072, 1024), (1, 3072), 0), out=buf223)
            del primals_97
            buf224 = reinterpret_tensor(buf220, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf220  # reuse
            buf225 = reinterpret_tensor(buf190, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_70, view_30, query_states_10, linear_71, view_31, key_states_20, mul_104, x1_20, x2_20, neg_20, cat_21, mul_105, q_embed_10, mul_106, x1_21, x2_21, neg_21, cat_22, mul_107, k_embed_10, getitem_77, hidden_states_103, key_states_21], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf221, primals_3, buf222, buf224, buf225, 1572864, stream=raw_stream0)
            buf226 = reinterpret_tensor(buf207, (24, 512, 512), (262144, 512, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_70, view_30, query_states_10, linear_71, view_31, key_states_20, mul_104, x1_20, x2_20, neg_20, cat_21, mul_105, q_embed_10, mul_106, x1_21, x2_21, neg_21, cat_22, mul_107, k_embed_10, getitem_77, hidden_states_103, key_states_21, transpose_54, matmul_21], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf224, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf225, (24, 128, 512), (65536, 1, 128), 0), out=buf226)
            buf229 = reinterpret_tensor(buf226, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf226  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_21, attn_weights_40, attn_weights_41, softmax_10, attn_weights_42], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf229, buf3, 12288, 512, stream=raw_stream0)
            buf230 = buf225; del buf225  # reuse
            # Topologically Sorted Source Nodes: [linear_72, view_32, value_states_20, getitem_78, hidden_states_104, value_states_21], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf223, buf230, 1572864, stream=raw_stream0)
            buf231 = reinterpret_tensor(buf224, (24, 512, 128), (65536, 128, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_72, view_32, value_states_20, getitem_78, hidden_states_104, value_states_21, matmul_21, attn_weights_40, attn_weights_41, softmax_10, attn_weights_42, attn_output_40], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf229, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf230, (24, 512, 128), (65536, 128, 1), 0), out=buf231)
            buf232 = reinterpret_tensor(buf230, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf230  # reuse
            # Topologically Sorted Source Nodes: [attn_output_40, transpose_55, attn_output_41], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf231, buf232, 1572864, stream=raw_stream0)
            assert_size_stride(primals_98, (3072, 3072), (3072, 1), 'input')
            buf233 = reinterpret_tensor(buf231, (512, 3072), (3072, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [attn_output_40, transpose_55, attn_output_41, reshape_32, attn_output_43], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_98, (3072, 3072), (1, 3072), 0), out=buf233)
            del primals_98
            assert_size_stride(primals_99, (3072, ), (1, ), 'input')
            buf235 = reinterpret_tensor(buf232, (1, 512, 3072), (1572864, 3072, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, hidden_states_106, pow_22, variance_21, add_75, rsqrt_21, hidden_states_107, to_59, hidden_states_108], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf212, buf218, buf233, primals_99, buf235, 512, 3072, stream=raw_stream0)
            del primals_99
            assert_size_stride(primals_100, (8192, 3072), (3072, 1), 'input')
            buf236 = reinterpret_tensor(buf217, (512, 8192), (8192, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, hidden_states_106, pow_22, variance_21, add_75, rsqrt_21, hidden_states_107, to_59, hidden_states_108, linear_74], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf235, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_100, (3072, 8192), (1, 3072), 0), out=buf236)
            del primals_100
            assert_size_stride(primals_101, (8192, 3072), (3072, 1), 'input')
            buf237 = buf216; del buf216  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, hidden_states_106, pow_22, variance_21, add_75, rsqrt_21, hidden_states_107, to_59, hidden_states_108, linear_74, linear_75], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf235, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_101, (3072, 8192), (1, 3072), 0), out=buf237)
            del primals_101
            buf238 = reinterpret_tensor(buf236, (1, 512, 8192), (4194304, 8192, 1), 0); del buf236  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_111], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf238, buf237, 4194304, stream=raw_stream0)
            assert_size_stride(primals_102, (3072, 8192), (8192, 1), 'input')
            buf239 = reinterpret_tensor(buf235, (512, 3072), (3072, 1), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_111, down_proj_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf238, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_102, (8192, 3072), (1, 8192), 0), out=buf239)
            del primals_102
            assert_size_stride(primals_103, (3072, ), (1, ), 'input')
            buf241 = reinterpret_tensor(buf221, (1, 512, 3072), (1572864, 3072, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, hidden_states_110, pow_23, variance_22, add_77, rsqrt_22, hidden_states_111, to_61, hidden_states_112], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf212, buf218, buf233, buf239, primals_103, buf241, 512, 3072, stream=raw_stream0)
            del primals_103
            assert_size_stride(primals_104, (3072, 3072), (3072, 1), 'input')
            buf242 = buf175; del buf175  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, hidden_states_110, pow_23, variance_22, add_77, rsqrt_22, hidden_states_111, to_61, hidden_states_112, linear_77], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf241, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_104, (3072, 3072), (1, 3072), 0), out=buf242)
            del primals_104
            assert_size_stride(primals_105, (1024, 3072), (3072, 1), 'input')
            buf243 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, hidden_states_110, pow_23, variance_22, add_77, rsqrt_22, hidden_states_111, to_61, hidden_states_112, linear_77, linear_78], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf241, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_105, (3072, 1024), (1, 3072), 0), out=buf243)
            del primals_105
            assert_size_stride(primals_106, (1024, 3072), (3072, 1), 'input')
            buf244 = buf222; del buf222  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, hidden_states_110, pow_23, variance_22, add_77, rsqrt_22, hidden_states_111, to_61, hidden_states_112, linear_77, linear_79], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf241, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_106, (3072, 1024), (1, 3072), 0), out=buf244)
            del primals_106
            buf245 = reinterpret_tensor(buf241, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf241  # reuse
            buf246 = reinterpret_tensor(buf199, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf199  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_77, view_33, query_states_11, linear_78, view_34, key_states_22, mul_114, x1_22, x2_22, neg_22, cat_23, mul_115, q_embed_11, mul_116, x1_23, x2_23, neg_23, cat_24, mul_117, k_embed_11, getitem_84, hidden_states_113, key_states_23], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf242, primals_3, buf243, buf245, buf246, 1572864, stream=raw_stream0)
            buf247 = reinterpret_tensor(buf229, (24, 512, 512), (262144, 512, 1), 0); del buf229  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_77, view_33, query_states_11, linear_78, view_34, key_states_22, mul_114, x1_22, x2_22, neg_22, cat_23, mul_115, q_embed_11, mul_116, x1_23, x2_23, neg_23, cat_24, mul_117, k_embed_11, getitem_84, hidden_states_113, key_states_23, transpose_59, matmul_23], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf245, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf246, (24, 128, 512), (65536, 1, 128), 0), out=buf247)
            buf250 = reinterpret_tensor(buf247, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_23, attn_weights_44, attn_weights_45, softmax_11, attn_weights_46], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf250, buf3, 12288, 512, stream=raw_stream0)
            buf251 = buf246; del buf246  # reuse
            # Topologically Sorted Source Nodes: [linear_79, view_35, value_states_22, getitem_85, hidden_states_114, value_states_23], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf244, buf251, 1572864, stream=raw_stream0)
            buf252 = reinterpret_tensor(buf245, (24, 512, 128), (65536, 128, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_79, view_35, value_states_22, getitem_85, hidden_states_114, value_states_23, matmul_23, attn_weights_44, attn_weights_45, softmax_11, attn_weights_46, attn_output_44], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf250, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf251, (24, 512, 128), (65536, 128, 1), 0), out=buf252)
            buf253 = reinterpret_tensor(buf251, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf251  # reuse
            # Topologically Sorted Source Nodes: [attn_output_44, transpose_60, attn_output_45], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf252, buf253, 1572864, stream=raw_stream0)
            assert_size_stride(primals_107, (3072, 3072), (3072, 1), 'input')
            buf254 = reinterpret_tensor(buf252, (512, 3072), (3072, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [attn_output_44, transpose_60, attn_output_45, reshape_35, attn_output_47], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf253, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 3072), (1, 3072), 0), out=buf254)
            del primals_107
            assert_size_stride(primals_108, (3072, ), (1, ), 'input')
            buf255 = buf212; del buf212  # reuse
            buf257 = reinterpret_tensor(buf253, (1, 512, 3072), (1572864, 3072, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_99, attn_output_43, hidden_states_105, down_proj_10, hidden_states_109, attn_output_47, hidden_states_115, hidden_states_116, pow_24, variance_23, add_82, rsqrt_23, hidden_states_117, to_64, hidden_states_118], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf255, buf218, buf233, buf239, buf254, primals_108, buf257, 512, 3072, stream=raw_stream0)
            del primals_108
            assert_size_stride(primals_109, (8192, 3072), (3072, 1), 'input')
            buf258 = reinterpret_tensor(buf238, (512, 8192), (8192, 1), 0); del buf238  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_116, pow_24, variance_23, add_82, rsqrt_23, hidden_states_117, to_64, hidden_states_118, linear_81], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf257, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_109, (3072, 8192), (1, 3072), 0), out=buf258)
            del primals_109
            assert_size_stride(primals_110, (8192, 3072), (3072, 1), 'input')
            buf259 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_116, pow_24, variance_23, add_82, rsqrt_23, hidden_states_117, to_64, hidden_states_118, linear_81, linear_82], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf257, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_110, (3072, 8192), (1, 3072), 0), out=buf259)
            del primals_110
            buf260 = reinterpret_tensor(buf258, (1, 512, 8192), (4194304, 8192, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_121], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf260, buf259, 4194304, stream=raw_stream0)
            assert_size_stride(primals_111, (3072, 8192), (8192, 1), 'input')
            buf261 = reinterpret_tensor(buf257, (512, 3072), (3072, 1), 0); del buf257  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_121, down_proj_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf260, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_111, (8192, 3072), (1, 8192), 0), out=buf261)
            del primals_111
            assert_size_stride(primals_112, (3072, ), (1, ), 'input')
            buf263 = reinterpret_tensor(buf254, (1, 512, 3072), (1572864, 3072, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, hidden_states_120, pow_25, variance_24, add_84, rsqrt_24, hidden_states_121, to_66, hidden_states_122], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf255, buf261, primals_112, buf263, 512, 3072, stream=raw_stream0)
            del primals_112
            assert_size_stride(primals_113, (3072, 3072), (3072, 1), 'input')
            buf264 = buf239; del buf239  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, hidden_states_120, pow_25, variance_24, add_84, rsqrt_24, hidden_states_121, to_66, hidden_states_122, linear_84], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf263, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_113, (3072, 3072), (1, 3072), 0), out=buf264)
            del primals_113
            assert_size_stride(primals_114, (1024, 3072), (3072, 1), 'input')
            buf265 = buf244; del buf244  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, hidden_states_120, pow_25, variance_24, add_84, rsqrt_24, hidden_states_121, to_66, hidden_states_122, linear_84, linear_85], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf263, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_114, (3072, 1024), (1, 3072), 0), out=buf265)
            del primals_114
            assert_size_stride(primals_115, (1024, 3072), (3072, 1), 'input')
            buf266 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, hidden_states_120, pow_25, variance_24, add_84, rsqrt_24, hidden_states_121, to_66, hidden_states_122, linear_84, linear_86], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf263, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_115, (3072, 1024), (1, 3072), 0), out=buf266)
            del primals_115
            buf267 = reinterpret_tensor(buf263, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf263  # reuse
            buf268 = reinterpret_tensor(buf233, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf233  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_84, view_36, query_states_12, linear_85, view_37, key_states_24, mul_124, x1_24, x2_24, neg_24, cat_25, mul_125, q_embed_12, mul_126, x1_25, x2_25, neg_25, cat_26, mul_127, k_embed_12, getitem_91, hidden_states_123, key_states_25], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf264, primals_3, buf265, buf267, buf268, 1572864, stream=raw_stream0)
            buf269 = reinterpret_tensor(buf250, (24, 512, 512), (262144, 512, 1), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_84, view_36, query_states_12, linear_85, view_37, key_states_24, mul_124, x1_24, x2_24, neg_24, cat_25, mul_125, q_embed_12, mul_126, x1_25, x2_25, neg_25, cat_26, mul_127, k_embed_12, getitem_91, hidden_states_123, key_states_25, transpose_64, matmul_25], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf267, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf268, (24, 128, 512), (65536, 1, 128), 0), out=buf269)
            buf272 = reinterpret_tensor(buf269, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf269  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_25, attn_weights_48, attn_weights_49, softmax_12, attn_weights_50], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf272, buf3, 12288, 512, stream=raw_stream0)
            buf273 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [linear_86, view_38, value_states_24, getitem_92, hidden_states_124, value_states_25], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf266, buf273, 1572864, stream=raw_stream0)
            buf274 = reinterpret_tensor(buf267, (24, 512, 128), (65536, 128, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_86, view_38, value_states_24, getitem_92, hidden_states_124, value_states_25, matmul_25, attn_weights_48, attn_weights_49, softmax_12, attn_weights_50, attn_output_48], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf272, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf273, (24, 512, 128), (65536, 128, 1), 0), out=buf274)
            buf275 = reinterpret_tensor(buf273, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf273  # reuse
            # Topologically Sorted Source Nodes: [attn_output_48, transpose_65, attn_output_49], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf274, buf275, 1572864, stream=raw_stream0)
            assert_size_stride(primals_116, (3072, 3072), (3072, 1), 'input')
            buf276 = reinterpret_tensor(buf274, (512, 3072), (3072, 1), 0); del buf274  # reuse
            # Topologically Sorted Source Nodes: [attn_output_48, transpose_65, attn_output_49, reshape_38, attn_output_51], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf275, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_116, (3072, 3072), (1, 3072), 0), out=buf276)
            del primals_116
            assert_size_stride(primals_117, (3072, ), (1, ), 'input')
            buf278 = reinterpret_tensor(buf275, (1, 512, 3072), (1572864, 3072, 1), 0); del buf275  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, hidden_states_126, pow_26, variance_25, add_89, rsqrt_25, hidden_states_127, to_69, hidden_states_128], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf255, buf261, buf276, primals_117, buf278, 512, 3072, stream=raw_stream0)
            del primals_117
            assert_size_stride(primals_118, (8192, 3072), (3072, 1), 'input')
            buf279 = reinterpret_tensor(buf260, (512, 8192), (8192, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, hidden_states_126, pow_26, variance_25, add_89, rsqrt_25, hidden_states_127, to_69, hidden_states_128, linear_88], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf278, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_118, (3072, 8192), (1, 3072), 0), out=buf279)
            del primals_118
            assert_size_stride(primals_119, (8192, 3072), (3072, 1), 'input')
            buf280 = buf259; del buf259  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, hidden_states_126, pow_26, variance_25, add_89, rsqrt_25, hidden_states_127, to_69, hidden_states_128, linear_88, linear_89], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf278, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_119, (3072, 8192), (1, 3072), 0), out=buf280)
            del primals_119
            buf281 = reinterpret_tensor(buf279, (1, 512, 8192), (4194304, 8192, 1), 0); del buf279  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_131], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf281, buf280, 4194304, stream=raw_stream0)
            assert_size_stride(primals_120, (3072, 8192), (8192, 1), 'input')
            buf282 = reinterpret_tensor(buf278, (512, 3072), (3072, 1), 0); del buf278  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_131, down_proj_12], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf281, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_120, (8192, 3072), (1, 8192), 0), out=buf282)
            del primals_120
            assert_size_stride(primals_121, (3072, ), (1, ), 'input')
            buf284 = reinterpret_tensor(buf264, (1, 512, 3072), (1572864, 3072, 1), 0); del buf264  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, hidden_states_130, pow_27, variance_26, add_91, rsqrt_26, hidden_states_131, to_71, hidden_states_132], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf255, buf261, buf276, buf282, primals_121, buf284, 512, 3072, stream=raw_stream0)
            del primals_121
            assert_size_stride(primals_122, (3072, 3072), (3072, 1), 'input')
            buf285 = buf218; del buf218  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, hidden_states_130, pow_27, variance_26, add_91, rsqrt_26, hidden_states_131, to_71, hidden_states_132, linear_91], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf284, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 3072), (1, 3072), 0), out=buf285)
            del primals_122
            assert_size_stride(primals_123, (1024, 3072), (3072, 1), 'input')
            buf286 = buf266; del buf266  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, hidden_states_130, pow_27, variance_26, add_91, rsqrt_26, hidden_states_131, to_71, hidden_states_132, linear_91, linear_92], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf284, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_123, (3072, 1024), (1, 3072), 0), out=buf286)
            del primals_123
            assert_size_stride(primals_124, (1024, 3072), (3072, 1), 'input')
            buf287 = buf265; del buf265  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, hidden_states_130, pow_27, variance_26, add_91, rsqrt_26, hidden_states_131, to_71, hidden_states_132, linear_91, linear_93], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf284, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_124, (3072, 1024), (1, 3072), 0), out=buf287)
            del primals_124
            buf288 = reinterpret_tensor(buf284, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf284  # reuse
            buf289 = reinterpret_tensor(buf242, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_91, view_39, query_states_13, linear_92, view_40, key_states_26, mul_134, x1_26, x2_26, neg_26, cat_27, mul_135, q_embed_13, mul_136, x1_27, x2_27, neg_27, cat_28, mul_137, k_embed_13, getitem_98, hidden_states_133, key_states_27], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf285, primals_3, buf286, buf288, buf289, 1572864, stream=raw_stream0)
            buf290 = reinterpret_tensor(buf272, (24, 512, 512), (262144, 512, 1), 0); del buf272  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_91, view_39, query_states_13, linear_92, view_40, key_states_26, mul_134, x1_26, x2_26, neg_26, cat_27, mul_135, q_embed_13, mul_136, x1_27, x2_27, neg_27, cat_28, mul_137, k_embed_13, getitem_98, hidden_states_133, key_states_27, transpose_69, matmul_27], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf288, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf289, (24, 128, 512), (65536, 1, 128), 0), out=buf290)
            buf293 = reinterpret_tensor(buf290, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf290  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_27, attn_weights_52, attn_weights_53, softmax_13, attn_weights_54], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf293, buf3, 12288, 512, stream=raw_stream0)
            buf294 = buf289; del buf289  # reuse
            # Topologically Sorted Source Nodes: [linear_93, view_41, value_states_26, getitem_99, hidden_states_134, value_states_27], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf287, buf294, 1572864, stream=raw_stream0)
            buf295 = reinterpret_tensor(buf288, (24, 512, 128), (65536, 128, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_93, view_41, value_states_26, getitem_99, hidden_states_134, value_states_27, matmul_27, attn_weights_52, attn_weights_53, softmax_13, attn_weights_54, attn_output_52], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf293, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf294, (24, 512, 128), (65536, 128, 1), 0), out=buf295)
            buf296 = reinterpret_tensor(buf294, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf294  # reuse
            # Topologically Sorted Source Nodes: [attn_output_52, transpose_70, attn_output_53], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf295, buf296, 1572864, stream=raw_stream0)
            assert_size_stride(primals_125, (3072, 3072), (3072, 1), 'input')
            buf297 = reinterpret_tensor(buf295, (512, 3072), (3072, 1), 0); del buf295  # reuse
            # Topologically Sorted Source Nodes: [attn_output_52, transpose_70, attn_output_53, reshape_41, attn_output_55], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf296, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_125, (3072, 3072), (1, 3072), 0), out=buf297)
            del primals_125
            assert_size_stride(primals_126, (3072, ), (1, ), 'input')
            buf298 = buf255; del buf255  # reuse
            buf300 = reinterpret_tensor(buf296, (1, 512, 3072), (1572864, 3072, 1), 0); del buf296  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_119, attn_output_51, hidden_states_125, down_proj_12, hidden_states_129, attn_output_55, hidden_states_135, hidden_states_136, pow_28, variance_27, add_96, rsqrt_27, hidden_states_137, to_74, hidden_states_138], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf298, buf261, buf276, buf282, buf297, primals_126, buf300, 512, 3072, stream=raw_stream0)
            del primals_126
            assert_size_stride(primals_127, (8192, 3072), (3072, 1), 'input')
            buf301 = reinterpret_tensor(buf281, (512, 8192), (8192, 1), 0); del buf281  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_136, pow_28, variance_27, add_96, rsqrt_27, hidden_states_137, to_74, hidden_states_138, linear_95], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf300, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_127, (3072, 8192), (1, 3072), 0), out=buf301)
            del primals_127
            assert_size_stride(primals_128, (8192, 3072), (3072, 1), 'input')
            buf302 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_136, pow_28, variance_27, add_96, rsqrt_27, hidden_states_137, to_74, hidden_states_138, linear_95, linear_96], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf300, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_128, (3072, 8192), (1, 3072), 0), out=buf302)
            del primals_128
            buf303 = reinterpret_tensor(buf301, (1, 512, 8192), (4194304, 8192, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_141], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf303, buf302, 4194304, stream=raw_stream0)
            assert_size_stride(primals_129, (3072, 8192), (8192, 1), 'input')
            buf304 = reinterpret_tensor(buf300, (512, 3072), (3072, 1), 0); del buf300  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_141, down_proj_13], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf303, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_129, (8192, 3072), (1, 8192), 0), out=buf304)
            del primals_129
            assert_size_stride(primals_130, (3072, ), (1, ), 'input')
            buf306 = reinterpret_tensor(buf297, (1, 512, 3072), (1572864, 3072, 1), 0); del buf297  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, hidden_states_140, pow_29, variance_28, add_98, rsqrt_28, hidden_states_141, to_76, hidden_states_142], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf298, buf304, primals_130, buf306, 512, 3072, stream=raw_stream0)
            del primals_130
            assert_size_stride(primals_131, (3072, 3072), (3072, 1), 'input')
            buf307 = buf282; del buf282  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, hidden_states_140, pow_29, variance_28, add_98, rsqrt_28, hidden_states_141, to_76, hidden_states_142, linear_98], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf306, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 3072), (1, 3072), 0), out=buf307)
            del primals_131
            assert_size_stride(primals_132, (1024, 3072), (3072, 1), 'input')
            buf308 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, hidden_states_140, pow_29, variance_28, add_98, rsqrt_28, hidden_states_141, to_76, hidden_states_142, linear_98, linear_99], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf306, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_132, (3072, 1024), (1, 3072), 0), out=buf308)
            del primals_132
            assert_size_stride(primals_133, (1024, 3072), (3072, 1), 'input')
            buf309 = buf286; del buf286  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, hidden_states_140, pow_29, variance_28, add_98, rsqrt_28, hidden_states_141, to_76, hidden_states_142, linear_98, linear_100], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf306, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_133, (3072, 1024), (1, 3072), 0), out=buf309)
            del primals_133
            buf310 = reinterpret_tensor(buf306, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf306  # reuse
            buf311 = reinterpret_tensor(buf276, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_98, view_42, query_states_14, linear_99, view_43, key_states_28, mul_144, x1_28, x2_28, neg_28, cat_29, mul_145, q_embed_14, mul_146, x1_29, x2_29, neg_29, cat_30, mul_147, k_embed_14, getitem_105, hidden_states_143, key_states_29], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf307, primals_3, buf308, buf310, buf311, 1572864, stream=raw_stream0)
            buf312 = reinterpret_tensor(buf293, (24, 512, 512), (262144, 512, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_98, view_42, query_states_14, linear_99, view_43, key_states_28, mul_144, x1_28, x2_28, neg_28, cat_29, mul_145, q_embed_14, mul_146, x1_29, x2_29, neg_29, cat_30, mul_147, k_embed_14, getitem_105, hidden_states_143, key_states_29, transpose_74, matmul_29], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf310, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf311, (24, 128, 512), (65536, 1, 128), 0), out=buf312)
            buf315 = reinterpret_tensor(buf312, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf312  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_29, attn_weights_56, attn_weights_57, softmax_14, attn_weights_58], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf315, buf3, 12288, 512, stream=raw_stream0)
            buf316 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [linear_100, view_44, value_states_28, getitem_106, hidden_states_144, value_states_29], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf309, buf316, 1572864, stream=raw_stream0)
            buf317 = reinterpret_tensor(buf310, (24, 512, 128), (65536, 128, 1), 0); del buf310  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_100, view_44, value_states_28, getitem_106, hidden_states_144, value_states_29, matmul_29, attn_weights_56, attn_weights_57, softmax_14, attn_weights_58, attn_output_56], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf315, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf316, (24, 512, 128), (65536, 128, 1), 0), out=buf317)
            buf318 = reinterpret_tensor(buf316, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf316  # reuse
            # Topologically Sorted Source Nodes: [attn_output_56, transpose_75, attn_output_57], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf317, buf318, 1572864, stream=raw_stream0)
            assert_size_stride(primals_134, (3072, 3072), (3072, 1), 'input')
            buf319 = reinterpret_tensor(buf317, (512, 3072), (3072, 1), 0); del buf317  # reuse
            # Topologically Sorted Source Nodes: [attn_output_56, transpose_75, attn_output_57, reshape_44, attn_output_59], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf318, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_134, (3072, 3072), (1, 3072), 0), out=buf319)
            del primals_134
            assert_size_stride(primals_135, (3072, ), (1, ), 'input')
            buf321 = reinterpret_tensor(buf318, (1, 512, 3072), (1572864, 3072, 1), 0); del buf318  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, hidden_states_146, pow_30, variance_29, add_103, rsqrt_29, hidden_states_147, to_79, hidden_states_148], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf298, buf304, buf319, primals_135, buf321, 512, 3072, stream=raw_stream0)
            del primals_135
            assert_size_stride(primals_136, (8192, 3072), (3072, 1), 'input')
            buf322 = reinterpret_tensor(buf303, (512, 8192), (8192, 1), 0); del buf303  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, hidden_states_146, pow_30, variance_29, add_103, rsqrt_29, hidden_states_147, to_79, hidden_states_148, linear_102], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf321, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_136, (3072, 8192), (1, 3072), 0), out=buf322)
            del primals_136
            assert_size_stride(primals_137, (8192, 3072), (3072, 1), 'input')
            buf323 = buf302; del buf302  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, hidden_states_146, pow_30, variance_29, add_103, rsqrt_29, hidden_states_147, to_79, hidden_states_148, linear_102, linear_103], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf321, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_137, (3072, 8192), (1, 3072), 0), out=buf323)
            del primals_137
            buf324 = reinterpret_tensor(buf322, (1, 512, 8192), (4194304, 8192, 1), 0); del buf322  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_151], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf324, buf323, 4194304, stream=raw_stream0)
            assert_size_stride(primals_138, (3072, 8192), (8192, 1), 'input')
            buf325 = reinterpret_tensor(buf321, (512, 3072), (3072, 1), 0); del buf321  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_151, down_proj_14], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf324, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_138, (8192, 3072), (1, 8192), 0), out=buf325)
            del primals_138
            assert_size_stride(primals_139, (3072, ), (1, ), 'input')
            buf327 = reinterpret_tensor(buf307, (1, 512, 3072), (1572864, 3072, 1), 0); del buf307  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, hidden_states_150, pow_31, variance_30, add_105, rsqrt_30, hidden_states_151, to_81, hidden_states_152], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf298, buf304, buf319, buf325, primals_139, buf327, 512, 3072, stream=raw_stream0)
            del primals_139
            assert_size_stride(primals_140, (3072, 3072), (3072, 1), 'input')
            buf328 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, hidden_states_150, pow_31, variance_30, add_105, rsqrt_30, hidden_states_151, to_81, hidden_states_152, linear_105], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf327, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_140, (3072, 3072), (1, 3072), 0), out=buf328)
            del primals_140
            assert_size_stride(primals_141, (1024, 3072), (3072, 1), 'input')
            buf329 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, hidden_states_150, pow_31, variance_30, add_105, rsqrt_30, hidden_states_151, to_81, hidden_states_152, linear_105, linear_106], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf327, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_141, (3072, 1024), (1, 3072), 0), out=buf329)
            del primals_141
            assert_size_stride(primals_142, (1024, 3072), (3072, 1), 'input')
            buf330 = buf308; del buf308  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, hidden_states_150, pow_31, variance_30, add_105, rsqrt_30, hidden_states_151, to_81, hidden_states_152, linear_105, linear_107], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf327, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_142, (3072, 1024), (1, 3072), 0), out=buf330)
            del primals_142
            buf331 = reinterpret_tensor(buf327, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf327  # reuse
            buf332 = reinterpret_tensor(buf285, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_105, view_45, query_states_15, linear_106, view_46, key_states_30, mul_154, x1_30, x2_30, neg_30, cat_31, mul_155, q_embed_15, mul_156, x1_31, x2_31, neg_31, cat_32, mul_157, k_embed_15, getitem_112, hidden_states_153, key_states_31], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf328, primals_3, buf329, buf331, buf332, 1572864, stream=raw_stream0)
            buf333 = reinterpret_tensor(buf315, (24, 512, 512), (262144, 512, 1), 0); del buf315  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_105, view_45, query_states_15, linear_106, view_46, key_states_30, mul_154, x1_30, x2_30, neg_30, cat_31, mul_155, q_embed_15, mul_156, x1_31, x2_31, neg_31, cat_32, mul_157, k_embed_15, getitem_112, hidden_states_153, key_states_31, transpose_79, matmul_31], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf331, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf332, (24, 128, 512), (65536, 1, 128), 0), out=buf333)
            buf336 = reinterpret_tensor(buf333, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf333  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_31, attn_weights_60, attn_weights_61, softmax_15, attn_weights_62], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf336, buf3, 12288, 512, stream=raw_stream0)
            buf337 = buf332; del buf332  # reuse
            # Topologically Sorted Source Nodes: [linear_107, view_47, value_states_30, getitem_113, hidden_states_154, value_states_31], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf330, buf337, 1572864, stream=raw_stream0)
            buf338 = reinterpret_tensor(buf331, (24, 512, 128), (65536, 128, 1), 0); del buf331  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_107, view_47, value_states_30, getitem_113, hidden_states_154, value_states_31, matmul_31, attn_weights_60, attn_weights_61, softmax_15, attn_weights_62, attn_output_60], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf336, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf337, (24, 512, 128), (65536, 128, 1), 0), out=buf338)
            buf339 = reinterpret_tensor(buf337, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf337  # reuse
            # Topologically Sorted Source Nodes: [attn_output_60, transpose_80, attn_output_61], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf338, buf339, 1572864, stream=raw_stream0)
            assert_size_stride(primals_143, (3072, 3072), (3072, 1), 'input')
            buf340 = reinterpret_tensor(buf338, (512, 3072), (3072, 1), 0); del buf338  # reuse
            # Topologically Sorted Source Nodes: [attn_output_60, transpose_80, attn_output_61, reshape_47, attn_output_63], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf339, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 3072), (1, 3072), 0), out=buf340)
            del primals_143
            assert_size_stride(primals_144, (3072, ), (1, ), 'input')
            buf341 = buf298; del buf298  # reuse
            buf343 = reinterpret_tensor(buf339, (1, 512, 3072), (1572864, 3072, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_139, attn_output_59, hidden_states_145, down_proj_14, hidden_states_149, attn_output_63, hidden_states_155, hidden_states_156, pow_32, variance_31, add_110, rsqrt_31, hidden_states_157, to_84, hidden_states_158], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf341, buf304, buf319, buf325, buf340, primals_144, buf343, 512, 3072, stream=raw_stream0)
            del primals_144
            assert_size_stride(primals_145, (8192, 3072), (3072, 1), 'input')
            buf344 = reinterpret_tensor(buf324, (512, 8192), (8192, 1), 0); del buf324  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_156, pow_32, variance_31, add_110, rsqrt_31, hidden_states_157, to_84, hidden_states_158, linear_109], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf343, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_145, (3072, 8192), (1, 3072), 0), out=buf344)
            del primals_145
            assert_size_stride(primals_146, (8192, 3072), (3072, 1), 'input')
            buf345 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_156, pow_32, variance_31, add_110, rsqrt_31, hidden_states_157, to_84, hidden_states_158, linear_109, linear_110], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf343, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_146, (3072, 8192), (1, 3072), 0), out=buf345)
            del primals_146
            buf346 = reinterpret_tensor(buf344, (1, 512, 8192), (4194304, 8192, 1), 0); del buf344  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_161], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf346, buf345, 4194304, stream=raw_stream0)
            assert_size_stride(primals_147, (3072, 8192), (8192, 1), 'input')
            buf347 = reinterpret_tensor(buf343, (512, 3072), (3072, 1), 0); del buf343  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_161, down_proj_15], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf346, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_147, (8192, 3072), (1, 8192), 0), out=buf347)
            del primals_147
            assert_size_stride(primals_148, (3072, ), (1, ), 'input')
            buf349 = reinterpret_tensor(buf340, (1, 512, 3072), (1572864, 3072, 1), 0); del buf340  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, hidden_states_160, pow_33, variance_32, add_112, rsqrt_32, hidden_states_161, to_86, hidden_states_162], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf341, buf347, primals_148, buf349, 512, 3072, stream=raw_stream0)
            del primals_148
            assert_size_stride(primals_149, (3072, 3072), (3072, 1), 'input')
            buf350 = buf325; del buf325  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, hidden_states_160, pow_33, variance_32, add_112, rsqrt_32, hidden_states_161, to_86, hidden_states_162, linear_112], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf349, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_149, (3072, 3072), (1, 3072), 0), out=buf350)
            del primals_149
            assert_size_stride(primals_150, (1024, 3072), (3072, 1), 'input')
            buf351 = buf330; del buf330  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, hidden_states_160, pow_33, variance_32, add_112, rsqrt_32, hidden_states_161, to_86, hidden_states_162, linear_112, linear_113], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf349, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_150, (3072, 1024), (1, 3072), 0), out=buf351)
            del primals_150
            assert_size_stride(primals_151, (1024, 3072), (3072, 1), 'input')
            buf352 = buf329; del buf329  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, hidden_states_160, pow_33, variance_32, add_112, rsqrt_32, hidden_states_161, to_86, hidden_states_162, linear_112, linear_114], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf349, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_151, (3072, 1024), (1, 3072), 0), out=buf352)
            del primals_151
            buf353 = reinterpret_tensor(buf349, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf349  # reuse
            buf354 = reinterpret_tensor(buf319, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_112, view_48, query_states_16, linear_113, view_49, key_states_32, mul_164, x1_32, x2_32, neg_32, cat_33, mul_165, q_embed_16, mul_166, x1_33, x2_33, neg_33, cat_34, mul_167, k_embed_16, getitem_119, hidden_states_163, key_states_33], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf350, primals_3, buf351, buf353, buf354, 1572864, stream=raw_stream0)
            buf355 = reinterpret_tensor(buf336, (24, 512, 512), (262144, 512, 1), 0); del buf336  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_112, view_48, query_states_16, linear_113, view_49, key_states_32, mul_164, x1_32, x2_32, neg_32, cat_33, mul_165, q_embed_16, mul_166, x1_33, x2_33, neg_33, cat_34, mul_167, k_embed_16, getitem_119, hidden_states_163, key_states_33, transpose_84, matmul_33], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf353, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf354, (24, 128, 512), (65536, 1, 128), 0), out=buf355)
            buf358 = reinterpret_tensor(buf355, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf355  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_33, attn_weights_64, attn_weights_65, softmax_16, attn_weights_66], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf358, buf3, 12288, 512, stream=raw_stream0)
            buf359 = buf354; del buf354  # reuse
            # Topologically Sorted Source Nodes: [linear_114, view_50, value_states_32, getitem_120, hidden_states_164, value_states_33], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf352, buf359, 1572864, stream=raw_stream0)
            buf360 = reinterpret_tensor(buf353, (24, 512, 128), (65536, 128, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_114, view_50, value_states_32, getitem_120, hidden_states_164, value_states_33, matmul_33, attn_weights_64, attn_weights_65, softmax_16, attn_weights_66, attn_output_64], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf358, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf359, (24, 512, 128), (65536, 128, 1), 0), out=buf360)
            buf361 = reinterpret_tensor(buf359, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf359  # reuse
            # Topologically Sorted Source Nodes: [attn_output_64, transpose_85, attn_output_65], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf360, buf361, 1572864, stream=raw_stream0)
            assert_size_stride(primals_152, (3072, 3072), (3072, 1), 'input')
            buf362 = reinterpret_tensor(buf360, (512, 3072), (3072, 1), 0); del buf360  # reuse
            # Topologically Sorted Source Nodes: [attn_output_64, transpose_85, attn_output_65, reshape_50, attn_output_67], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf361, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_152, (3072, 3072), (1, 3072), 0), out=buf362)
            del primals_152
            assert_size_stride(primals_153, (3072, ), (1, ), 'input')
            buf364 = reinterpret_tensor(buf361, (1, 512, 3072), (1572864, 3072, 1), 0); del buf361  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, hidden_states_166, pow_34, variance_33, add_117, rsqrt_33, hidden_states_167, to_89, hidden_states_168], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf341, buf347, buf362, primals_153, buf364, 512, 3072, stream=raw_stream0)
            del primals_153
            assert_size_stride(primals_154, (8192, 3072), (3072, 1), 'input')
            buf365 = reinterpret_tensor(buf346, (512, 8192), (8192, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, hidden_states_166, pow_34, variance_33, add_117, rsqrt_33, hidden_states_167, to_89, hidden_states_168, linear_116], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf364, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_154, (3072, 8192), (1, 3072), 0), out=buf365)
            del primals_154
            assert_size_stride(primals_155, (8192, 3072), (3072, 1), 'input')
            buf366 = buf345; del buf345  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, hidden_states_166, pow_34, variance_33, add_117, rsqrt_33, hidden_states_167, to_89, hidden_states_168, linear_116, linear_117], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf364, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_155, (3072, 8192), (1, 3072), 0), out=buf366)
            del primals_155
            buf367 = reinterpret_tensor(buf365, (1, 512, 8192), (4194304, 8192, 1), 0); del buf365  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_171], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf367, buf366, 4194304, stream=raw_stream0)
            assert_size_stride(primals_156, (3072, 8192), (8192, 1), 'input')
            buf368 = reinterpret_tensor(buf364, (512, 3072), (3072, 1), 0); del buf364  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_171, down_proj_16], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf367, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_156, (8192, 3072), (1, 8192), 0), out=buf368)
            del primals_156
            assert_size_stride(primals_157, (3072, ), (1, ), 'input')
            buf370 = reinterpret_tensor(buf350, (1, 512, 3072), (1572864, 3072, 1), 0); del buf350  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, hidden_states_170, pow_35, variance_34, add_119, rsqrt_34, hidden_states_171, to_91, hidden_states_172], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf341, buf347, buf362, buf368, primals_157, buf370, 512, 3072, stream=raw_stream0)
            del primals_157
            assert_size_stride(primals_158, (3072, 3072), (3072, 1), 'input')
            buf371 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, hidden_states_170, pow_35, variance_34, add_119, rsqrt_34, hidden_states_171, to_91, hidden_states_172, linear_119], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_158, (3072, 3072), (1, 3072), 0), out=buf371)
            del primals_158
            assert_size_stride(primals_159, (1024, 3072), (3072, 1), 'input')
            buf372 = buf352; del buf352  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, hidden_states_170, pow_35, variance_34, add_119, rsqrt_34, hidden_states_171, to_91, hidden_states_172, linear_119, linear_120], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_159, (3072, 1024), (1, 3072), 0), out=buf372)
            del primals_159
            assert_size_stride(primals_160, (1024, 3072), (3072, 1), 'input')
            buf373 = buf351; del buf351  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, hidden_states_170, pow_35, variance_34, add_119, rsqrt_34, hidden_states_171, to_91, hidden_states_172, linear_119, linear_121], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_160, (3072, 1024), (1, 3072), 0), out=buf373)
            del primals_160
            buf374 = reinterpret_tensor(buf370, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf370  # reuse
            buf375 = reinterpret_tensor(buf328, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf328  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_119, view_51, query_states_17, linear_120, view_52, key_states_34, mul_174, x1_34, x2_34, neg_34, cat_35, mul_175, q_embed_17, mul_176, x1_35, x2_35, neg_35, cat_36, mul_177, k_embed_17, getitem_126, hidden_states_173, key_states_35], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf371, primals_3, buf372, buf374, buf375, 1572864, stream=raw_stream0)
            buf376 = reinterpret_tensor(buf358, (24, 512, 512), (262144, 512, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_119, view_51, query_states_17, linear_120, view_52, key_states_34, mul_174, x1_34, x2_34, neg_34, cat_35, mul_175, q_embed_17, mul_176, x1_35, x2_35, neg_35, cat_36, mul_177, k_embed_17, getitem_126, hidden_states_173, key_states_35, transpose_89, matmul_35], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf374, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf375, (24, 128, 512), (65536, 1, 128), 0), out=buf376)
            buf379 = reinterpret_tensor(buf376, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf376  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_35, attn_weights_68, attn_weights_69, softmax_17, attn_weights_70], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf379, buf3, 12288, 512, stream=raw_stream0)
            buf380 = buf375; del buf375  # reuse
            # Topologically Sorted Source Nodes: [linear_121, view_53, value_states_34, getitem_127, hidden_states_174, value_states_35], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf373, buf380, 1572864, stream=raw_stream0)
            buf381 = reinterpret_tensor(buf374, (24, 512, 128), (65536, 128, 1), 0); del buf374  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_121, view_53, value_states_34, getitem_127, hidden_states_174, value_states_35, matmul_35, attn_weights_68, attn_weights_69, softmax_17, attn_weights_70, attn_output_68], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf379, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf380, (24, 512, 128), (65536, 128, 1), 0), out=buf381)
            buf382 = reinterpret_tensor(buf380, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf380  # reuse
            # Topologically Sorted Source Nodes: [attn_output_68, transpose_90, attn_output_69], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf381, buf382, 1572864, stream=raw_stream0)
            assert_size_stride(primals_161, (3072, 3072), (3072, 1), 'input')
            buf383 = reinterpret_tensor(buf381, (512, 3072), (3072, 1), 0); del buf381  # reuse
            # Topologically Sorted Source Nodes: [attn_output_68, transpose_90, attn_output_69, reshape_53, attn_output_71], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf382, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_161, (3072, 3072), (1, 3072), 0), out=buf383)
            del primals_161
            assert_size_stride(primals_162, (3072, ), (1, ), 'input')
            buf384 = buf341; del buf341  # reuse
            buf386 = reinterpret_tensor(buf382, (1, 512, 3072), (1572864, 3072, 1), 0); del buf382  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_159, attn_output_67, hidden_states_165, down_proj_16, hidden_states_169, attn_output_71, hidden_states_175, hidden_states_176, pow_36, variance_35, add_124, rsqrt_35, hidden_states_177, to_94, hidden_states_178], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf384, buf347, buf362, buf368, buf383, primals_162, buf386, 512, 3072, stream=raw_stream0)
            del primals_162
            assert_size_stride(primals_163, (8192, 3072), (3072, 1), 'input')
            buf387 = reinterpret_tensor(buf367, (512, 8192), (8192, 1), 0); del buf367  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_176, pow_36, variance_35, add_124, rsqrt_35, hidden_states_177, to_94, hidden_states_178, linear_123], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf386, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_163, (3072, 8192), (1, 3072), 0), out=buf387)
            del primals_163
            assert_size_stride(primals_164, (8192, 3072), (3072, 1), 'input')
            buf388 = buf366; del buf366  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_176, pow_36, variance_35, add_124, rsqrt_35, hidden_states_177, to_94, hidden_states_178, linear_123, linear_124], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf386, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_164, (3072, 8192), (1, 3072), 0), out=buf388)
            del primals_164
            buf389 = reinterpret_tensor(buf387, (1, 512, 8192), (4194304, 8192, 1), 0); del buf387  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_181], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf389, buf388, 4194304, stream=raw_stream0)
            assert_size_stride(primals_165, (3072, 8192), (8192, 1), 'input')
            buf390 = reinterpret_tensor(buf386, (512, 3072), (3072, 1), 0); del buf386  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_181, down_proj_17], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf389, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_165, (8192, 3072), (1, 8192), 0), out=buf390)
            del primals_165
            assert_size_stride(primals_166, (3072, ), (1, ), 'input')
            buf392 = reinterpret_tensor(buf383, (1, 512, 3072), (1572864, 3072, 1), 0); del buf383  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, hidden_states_180, pow_37, variance_36, add_126, rsqrt_36, hidden_states_181, to_96, hidden_states_182], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf384, buf390, primals_166, buf392, 512, 3072, stream=raw_stream0)
            del primals_166
            assert_size_stride(primals_167, (3072, 3072), (3072, 1), 'input')
            buf393 = buf368; del buf368  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, hidden_states_180, pow_37, variance_36, add_126, rsqrt_36, hidden_states_181, to_96, hidden_states_182, linear_126], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf392, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_167, (3072, 3072), (1, 3072), 0), out=buf393)
            del primals_167
            assert_size_stride(primals_168, (1024, 3072), (3072, 1), 'input')
            buf394 = buf373; del buf373  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, hidden_states_180, pow_37, variance_36, add_126, rsqrt_36, hidden_states_181, to_96, hidden_states_182, linear_126, linear_127], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf392, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_168, (3072, 1024), (1, 3072), 0), out=buf394)
            del primals_168
            assert_size_stride(primals_169, (1024, 3072), (3072, 1), 'input')
            buf395 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, hidden_states_180, pow_37, variance_36, add_126, rsqrt_36, hidden_states_181, to_96, hidden_states_182, linear_126, linear_128], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf392, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_169, (3072, 1024), (1, 3072), 0), out=buf395)
            del primals_169
            buf396 = reinterpret_tensor(buf392, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf392  # reuse
            buf397 = reinterpret_tensor(buf362, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_126, view_54, query_states_18, linear_127, view_55, key_states_36, mul_184, x1_36, x2_36, neg_36, cat_37, mul_185, q_embed_18, mul_186, x1_37, x2_37, neg_37, cat_38, mul_187, k_embed_18, getitem_133, hidden_states_183, key_states_37], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf393, primals_3, buf394, buf396, buf397, 1572864, stream=raw_stream0)
            buf398 = reinterpret_tensor(buf379, (24, 512, 512), (262144, 512, 1), 0); del buf379  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_126, view_54, query_states_18, linear_127, view_55, key_states_36, mul_184, x1_36, x2_36, neg_36, cat_37, mul_185, q_embed_18, mul_186, x1_37, x2_37, neg_37, cat_38, mul_187, k_embed_18, getitem_133, hidden_states_183, key_states_37, transpose_94, matmul_37], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf396, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf397, (24, 128, 512), (65536, 1, 128), 0), out=buf398)
            buf401 = reinterpret_tensor(buf398, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf398  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_37, attn_weights_72, attn_weights_73, softmax_18, attn_weights_74], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf401, buf3, 12288, 512, stream=raw_stream0)
            buf402 = buf397; del buf397  # reuse
            # Topologically Sorted Source Nodes: [linear_128, view_56, value_states_36, getitem_134, hidden_states_184, value_states_37], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf395, buf402, 1572864, stream=raw_stream0)
            buf403 = reinterpret_tensor(buf396, (24, 512, 128), (65536, 128, 1), 0); del buf396  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_128, view_56, value_states_36, getitem_134, hidden_states_184, value_states_37, matmul_37, attn_weights_72, attn_weights_73, softmax_18, attn_weights_74, attn_output_72], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf401, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf402, (24, 512, 128), (65536, 128, 1), 0), out=buf403)
            buf404 = reinterpret_tensor(buf402, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf402  # reuse
            # Topologically Sorted Source Nodes: [attn_output_72, transpose_95, attn_output_73], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf403, buf404, 1572864, stream=raw_stream0)
            assert_size_stride(primals_170, (3072, 3072), (3072, 1), 'input')
            buf405 = reinterpret_tensor(buf403, (512, 3072), (3072, 1), 0); del buf403  # reuse
            # Topologically Sorted Source Nodes: [attn_output_72, transpose_95, attn_output_73, reshape_56, attn_output_75], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf404, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_170, (3072, 3072), (1, 3072), 0), out=buf405)
            del primals_170
            assert_size_stride(primals_171, (3072, ), (1, ), 'input')
            buf407 = reinterpret_tensor(buf404, (1, 512, 3072), (1572864, 3072, 1), 0); del buf404  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, hidden_states_186, pow_38, variance_37, add_131, rsqrt_37, hidden_states_187, to_99, hidden_states_188], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf384, buf390, buf405, primals_171, buf407, 512, 3072, stream=raw_stream0)
            del primals_171
            assert_size_stride(primals_172, (8192, 3072), (3072, 1), 'input')
            buf408 = reinterpret_tensor(buf389, (512, 8192), (8192, 1), 0); del buf389  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, hidden_states_186, pow_38, variance_37, add_131, rsqrt_37, hidden_states_187, to_99, hidden_states_188, linear_130], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf407, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_172, (3072, 8192), (1, 3072), 0), out=buf408)
            del primals_172
            assert_size_stride(primals_173, (8192, 3072), (3072, 1), 'input')
            buf409 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, hidden_states_186, pow_38, variance_37, add_131, rsqrt_37, hidden_states_187, to_99, hidden_states_188, linear_130, linear_131], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf407, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_173, (3072, 8192), (1, 3072), 0), out=buf409)
            del primals_173
            buf410 = reinterpret_tensor(buf408, (1, 512, 8192), (4194304, 8192, 1), 0); del buf408  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_191], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf410, buf409, 4194304, stream=raw_stream0)
            assert_size_stride(primals_174, (3072, 8192), (8192, 1), 'input')
            buf411 = reinterpret_tensor(buf407, (512, 3072), (3072, 1), 0); del buf407  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_191, down_proj_18], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf410, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_174, (8192, 3072), (1, 8192), 0), out=buf411)
            del primals_174
            assert_size_stride(primals_175, (3072, ), (1, ), 'input')
            buf413 = reinterpret_tensor(buf393, (1, 512, 3072), (1572864, 3072, 1), 0); del buf393  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, hidden_states_190, pow_39, variance_38, add_133, rsqrt_38, hidden_states_191, to_101, hidden_states_192], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf384, buf390, buf405, buf411, primals_175, buf413, 512, 3072, stream=raw_stream0)
            del primals_175
            assert_size_stride(primals_176, (3072, 3072), (3072, 1), 'input')
            buf414 = buf347; del buf347  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, hidden_states_190, pow_39, variance_38, add_133, rsqrt_38, hidden_states_191, to_101, hidden_states_192, linear_133], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf413, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_176, (3072, 3072), (1, 3072), 0), out=buf414)
            del primals_176
            assert_size_stride(primals_177, (1024, 3072), (3072, 1), 'input')
            buf415 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, hidden_states_190, pow_39, variance_38, add_133, rsqrt_38, hidden_states_191, to_101, hidden_states_192, linear_133, linear_134], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf413, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_177, (3072, 1024), (1, 3072), 0), out=buf415)
            del primals_177
            assert_size_stride(primals_178, (1024, 3072), (3072, 1), 'input')
            buf416 = buf394; del buf394  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, hidden_states_190, pow_39, variance_38, add_133, rsqrt_38, hidden_states_191, to_101, hidden_states_192, linear_133, linear_135], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf413, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_178, (3072, 1024), (1, 3072), 0), out=buf416)
            del primals_178
            buf417 = reinterpret_tensor(buf413, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf413  # reuse
            buf418 = reinterpret_tensor(buf371, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_133, view_57, query_states_19, linear_134, view_58, key_states_38, mul_194, x1_38, x2_38, neg_38, cat_39, mul_195, q_embed_19, mul_196, x1_39, x2_39, neg_39, cat_40, mul_197, k_embed_19, getitem_140, hidden_states_193, key_states_39], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf414, primals_3, buf415, buf417, buf418, 1572864, stream=raw_stream0)
            buf419 = reinterpret_tensor(buf401, (24, 512, 512), (262144, 512, 1), 0); del buf401  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_133, view_57, query_states_19, linear_134, view_58, key_states_38, mul_194, x1_38, x2_38, neg_38, cat_39, mul_195, q_embed_19, mul_196, x1_39, x2_39, neg_39, cat_40, mul_197, k_embed_19, getitem_140, hidden_states_193, key_states_39, transpose_99, matmul_39], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf417, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf418, (24, 128, 512), (65536, 1, 128), 0), out=buf419)
            buf422 = reinterpret_tensor(buf419, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf419  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_39, attn_weights_76, attn_weights_77, softmax_19, attn_weights_78], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf422, buf3, 12288, 512, stream=raw_stream0)
            buf423 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [linear_135, view_59, value_states_38, getitem_141, hidden_states_194, value_states_39], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf416, buf423, 1572864, stream=raw_stream0)
            buf424 = reinterpret_tensor(buf417, (24, 512, 128), (65536, 128, 1), 0); del buf417  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_135, view_59, value_states_38, getitem_141, hidden_states_194, value_states_39, matmul_39, attn_weights_76, attn_weights_77, softmax_19, attn_weights_78, attn_output_76], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf422, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf423, (24, 512, 128), (65536, 128, 1), 0), out=buf424)
            buf425 = reinterpret_tensor(buf423, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf423  # reuse
            # Topologically Sorted Source Nodes: [attn_output_76, transpose_100, attn_output_77], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf424, buf425, 1572864, stream=raw_stream0)
            assert_size_stride(primals_179, (3072, 3072), (3072, 1), 'input')
            buf426 = reinterpret_tensor(buf424, (512, 3072), (3072, 1), 0); del buf424  # reuse
            # Topologically Sorted Source Nodes: [attn_output_76, transpose_100, attn_output_77, reshape_59, attn_output_79], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf425, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_179, (3072, 3072), (1, 3072), 0), out=buf426)
            del primals_179
            assert_size_stride(primals_180, (3072, ), (1, ), 'input')
            buf427 = buf384; del buf384  # reuse
            buf429 = reinterpret_tensor(buf425, (1, 512, 3072), (1572864, 3072, 1), 0); del buf425  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_179, attn_output_75, hidden_states_185, down_proj_18, hidden_states_189, attn_output_79, hidden_states_195, hidden_states_196, pow_40, variance_39, add_138, rsqrt_39, hidden_states_197, to_104, hidden_states_198], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf427, buf390, buf405, buf411, buf426, primals_180, buf429, 512, 3072, stream=raw_stream0)
            del primals_180
            assert_size_stride(primals_181, (8192, 3072), (3072, 1), 'input')
            buf430 = reinterpret_tensor(buf410, (512, 8192), (8192, 1), 0); del buf410  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_196, pow_40, variance_39, add_138, rsqrt_39, hidden_states_197, to_104, hidden_states_198, linear_137], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf429, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_181, (3072, 8192), (1, 3072), 0), out=buf430)
            del primals_181
            assert_size_stride(primals_182, (8192, 3072), (3072, 1), 'input')
            buf431 = buf409; del buf409  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_196, pow_40, variance_39, add_138, rsqrt_39, hidden_states_197, to_104, hidden_states_198, linear_137, linear_138], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf429, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_182, (3072, 8192), (1, 3072), 0), out=buf431)
            del primals_182
            buf432 = reinterpret_tensor(buf430, (1, 512, 8192), (4194304, 8192, 1), 0); del buf430  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_201], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf432, buf431, 4194304, stream=raw_stream0)
            assert_size_stride(primals_183, (3072, 8192), (8192, 1), 'input')
            buf433 = reinterpret_tensor(buf429, (512, 3072), (3072, 1), 0); del buf429  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_201, down_proj_19], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf432, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_183, (8192, 3072), (1, 8192), 0), out=buf433)
            del primals_183
            assert_size_stride(primals_184, (3072, ), (1, ), 'input')
            buf435 = reinterpret_tensor(buf426, (1, 512, 3072), (1572864, 3072, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, hidden_states_200, pow_41, variance_40, add_140, rsqrt_40, hidden_states_201, to_106, hidden_states_202], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf427, buf433, primals_184, buf435, 512, 3072, stream=raw_stream0)
            del primals_184
            assert_size_stride(primals_185, (3072, 3072), (3072, 1), 'input')
            buf436 = buf411; del buf411  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, hidden_states_200, pow_41, variance_40, add_140, rsqrt_40, hidden_states_201, to_106, hidden_states_202, linear_140], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf435, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_185, (3072, 3072), (1, 3072), 0), out=buf436)
            del primals_185
            assert_size_stride(primals_186, (1024, 3072), (3072, 1), 'input')
            buf437 = buf416; del buf416  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, hidden_states_200, pow_41, variance_40, add_140, rsqrt_40, hidden_states_201, to_106, hidden_states_202, linear_140, linear_141], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf435, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_186, (3072, 1024), (1, 3072), 0), out=buf437)
            del primals_186
            assert_size_stride(primals_187, (1024, 3072), (3072, 1), 'input')
            buf438 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, hidden_states_200, pow_41, variance_40, add_140, rsqrt_40, hidden_states_201, to_106, hidden_states_202, linear_140, linear_142], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf435, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_187, (3072, 1024), (1, 3072), 0), out=buf438)
            del primals_187
            buf439 = reinterpret_tensor(buf435, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf435  # reuse
            buf440 = reinterpret_tensor(buf405, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf405  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_140, view_60, query_states_20, linear_141, view_61, key_states_40, mul_204, x1_40, x2_40, neg_40, cat_41, mul_205, q_embed_20, mul_206, x1_41, x2_41, neg_41, cat_42, mul_207, k_embed_20, getitem_147, hidden_states_203, key_states_41], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf436, primals_3, buf437, buf439, buf440, 1572864, stream=raw_stream0)
            buf441 = reinterpret_tensor(buf422, (24, 512, 512), (262144, 512, 1), 0); del buf422  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_140, view_60, query_states_20, linear_141, view_61, key_states_40, mul_204, x1_40, x2_40, neg_40, cat_41, mul_205, q_embed_20, mul_206, x1_41, x2_41, neg_41, cat_42, mul_207, k_embed_20, getitem_147, hidden_states_203, key_states_41, transpose_104, matmul_41], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf439, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf440, (24, 128, 512), (65536, 1, 128), 0), out=buf441)
            buf444 = reinterpret_tensor(buf441, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf441  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_41, attn_weights_80, attn_weights_81, softmax_20, attn_weights_82], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf444, buf3, 12288, 512, stream=raw_stream0)
            buf445 = buf440; del buf440  # reuse
            # Topologically Sorted Source Nodes: [linear_142, view_62, value_states_40, getitem_148, hidden_states_204, value_states_41], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf438, buf445, 1572864, stream=raw_stream0)
            buf446 = reinterpret_tensor(buf439, (24, 512, 128), (65536, 128, 1), 0); del buf439  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_142, view_62, value_states_40, getitem_148, hidden_states_204, value_states_41, matmul_41, attn_weights_80, attn_weights_81, softmax_20, attn_weights_82, attn_output_80], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf444, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf445, (24, 512, 128), (65536, 128, 1), 0), out=buf446)
            buf447 = reinterpret_tensor(buf445, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf445  # reuse
            # Topologically Sorted Source Nodes: [attn_output_80, transpose_105, attn_output_81], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf446, buf447, 1572864, stream=raw_stream0)
            assert_size_stride(primals_188, (3072, 3072), (3072, 1), 'input')
            buf448 = reinterpret_tensor(buf446, (512, 3072), (3072, 1), 0); del buf446  # reuse
            # Topologically Sorted Source Nodes: [attn_output_80, transpose_105, attn_output_81, reshape_62, attn_output_83], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf447, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_188, (3072, 3072), (1, 3072), 0), out=buf448)
            del primals_188
            assert_size_stride(primals_189, (3072, ), (1, ), 'input')
            buf450 = reinterpret_tensor(buf447, (1, 512, 3072), (1572864, 3072, 1), 0); del buf447  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, hidden_states_206, pow_42, variance_41, add_145, rsqrt_41, hidden_states_207, to_109, hidden_states_208], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf427, buf433, buf448, primals_189, buf450, 512, 3072, stream=raw_stream0)
            del primals_189
            assert_size_stride(primals_190, (8192, 3072), (3072, 1), 'input')
            buf451 = reinterpret_tensor(buf432, (512, 8192), (8192, 1), 0); del buf432  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, hidden_states_206, pow_42, variance_41, add_145, rsqrt_41, hidden_states_207, to_109, hidden_states_208, linear_144], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf450, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_190, (3072, 8192), (1, 3072), 0), out=buf451)
            del primals_190
            assert_size_stride(primals_191, (8192, 3072), (3072, 1), 'input')
            buf452 = buf431; del buf431  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, hidden_states_206, pow_42, variance_41, add_145, rsqrt_41, hidden_states_207, to_109, hidden_states_208, linear_144, linear_145], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf450, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_191, (3072, 8192), (1, 3072), 0), out=buf452)
            del primals_191
            buf453 = reinterpret_tensor(buf451, (1, 512, 8192), (4194304, 8192, 1), 0); del buf451  # reuse
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_211], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf453, buf452, 4194304, stream=raw_stream0)
            assert_size_stride(primals_192, (3072, 8192), (8192, 1), 'input')
            buf454 = reinterpret_tensor(buf450, (512, 3072), (3072, 1), 0); del buf450  # reuse
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_211, down_proj_20], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf453, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_192, (8192, 3072), (1, 8192), 0), out=buf454)
            del primals_192
            assert_size_stride(primals_193, (3072, ), (1, ), 'input')
            buf456 = reinterpret_tensor(buf436, (1, 512, 3072), (1572864, 3072, 1), 0); del buf436  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, hidden_states_210, pow_43, variance_42, add_147, rsqrt_42, hidden_states_211, to_111, hidden_states_212], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf427, buf433, buf448, buf454, primals_193, buf456, 512, 3072, stream=raw_stream0)
            del primals_193
            assert_size_stride(primals_194, (3072, 3072), (3072, 1), 'input')
            buf457 = buf390; del buf390  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, hidden_states_210, pow_43, variance_42, add_147, rsqrt_42, hidden_states_211, to_111, hidden_states_212, linear_147], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf456, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_194, (3072, 3072), (1, 3072), 0), out=buf457)
            del primals_194
            assert_size_stride(primals_195, (1024, 3072), (3072, 1), 'input')
            buf458 = buf438; del buf438  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, hidden_states_210, pow_43, variance_42, add_147, rsqrt_42, hidden_states_211, to_111, hidden_states_212, linear_147, linear_148], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf456, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_195, (3072, 1024), (1, 3072), 0), out=buf458)
            del primals_195
            assert_size_stride(primals_196, (1024, 3072), (3072, 1), 'input')
            buf459 = buf437; del buf437  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, hidden_states_210, pow_43, variance_42, add_147, rsqrt_42, hidden_states_211, to_111, hidden_states_212, linear_147, linear_149], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf456, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_196, (3072, 1024), (1, 3072), 0), out=buf459)
            del primals_196
            buf460 = reinterpret_tensor(buf456, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf456  # reuse
            buf461 = reinterpret_tensor(buf414, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf414  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_147, view_63, query_states_21, linear_148, view_64, key_states_42, mul_214, x1_42, x2_42, neg_42, cat_43, mul_215, q_embed_21, mul_216, x1_43, x2_43, neg_43, cat_44, mul_217, k_embed_21, getitem_154, hidden_states_213, key_states_43], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf457, primals_3, buf458, buf460, buf461, 1572864, stream=raw_stream0)
            buf462 = reinterpret_tensor(buf444, (24, 512, 512), (262144, 512, 1), 0); del buf444  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_147, view_63, query_states_21, linear_148, view_64, key_states_42, mul_214, x1_42, x2_42, neg_42, cat_43, mul_215, q_embed_21, mul_216, x1_43, x2_43, neg_43, cat_44, mul_217, k_embed_21, getitem_154, hidden_states_213, key_states_43, transpose_109, matmul_43], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf460, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf461, (24, 128, 512), (65536, 1, 128), 0), out=buf462)
            buf465 = reinterpret_tensor(buf462, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf462  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_43, attn_weights_84, attn_weights_85, softmax_21, attn_weights_86], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf465, buf3, 12288, 512, stream=raw_stream0)
            buf466 = buf461; del buf461  # reuse
            # Topologically Sorted Source Nodes: [linear_149, view_65, value_states_42, getitem_155, hidden_states_214, value_states_43], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf459, buf466, 1572864, stream=raw_stream0)
            buf467 = reinterpret_tensor(buf460, (24, 512, 128), (65536, 128, 1), 0); del buf460  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_149, view_65, value_states_42, getitem_155, hidden_states_214, value_states_43, matmul_43, attn_weights_84, attn_weights_85, softmax_21, attn_weights_86, attn_output_84], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf465, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf466, (24, 512, 128), (65536, 128, 1), 0), out=buf467)
            buf468 = reinterpret_tensor(buf466, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf466  # reuse
            # Topologically Sorted Source Nodes: [attn_output_84, transpose_110, attn_output_85], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf467, buf468, 1572864, stream=raw_stream0)
            assert_size_stride(primals_197, (3072, 3072), (3072, 1), 'input')
            buf469 = reinterpret_tensor(buf467, (512, 3072), (3072, 1), 0); del buf467  # reuse
            # Topologically Sorted Source Nodes: [attn_output_84, transpose_110, attn_output_85, reshape_65, attn_output_87], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf468, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_197, (3072, 3072), (1, 3072), 0), out=buf469)
            del primals_197
            assert_size_stride(primals_198, (3072, ), (1, ), 'input')
            buf470 = buf427; del buf427  # reuse
            buf472 = reinterpret_tensor(buf468, (1, 512, 3072), (1572864, 3072, 1), 0); del buf468  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_199, attn_output_83, hidden_states_205, down_proj_20, hidden_states_209, attn_output_87, hidden_states_215, hidden_states_216, pow_44, variance_43, add_152, rsqrt_43, hidden_states_217, to_114, hidden_states_218], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf470, buf433, buf448, buf454, buf469, primals_198, buf472, 512, 3072, stream=raw_stream0)
            del primals_198
            assert_size_stride(primals_199, (8192, 3072), (3072, 1), 'input')
            buf473 = reinterpret_tensor(buf453, (512, 8192), (8192, 1), 0); del buf453  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_216, pow_44, variance_43, add_152, rsqrt_43, hidden_states_217, to_114, hidden_states_218, linear_151], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf472, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_199, (3072, 8192), (1, 3072), 0), out=buf473)
            del primals_199
            assert_size_stride(primals_200, (8192, 3072), (3072, 1), 'input')
            buf474 = buf452; del buf452  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_216, pow_44, variance_43, add_152, rsqrt_43, hidden_states_217, to_114, hidden_states_218, linear_151, linear_152], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf472, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_200, (3072, 8192), (1, 3072), 0), out=buf474)
            del primals_200
            buf475 = reinterpret_tensor(buf473, (1, 512, 8192), (4194304, 8192, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_221], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf475, buf474, 4194304, stream=raw_stream0)
            assert_size_stride(primals_201, (3072, 8192), (8192, 1), 'input')
            buf476 = reinterpret_tensor(buf472, (512, 3072), (3072, 1), 0); del buf472  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_221, down_proj_21], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf475, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_201, (8192, 3072), (1, 8192), 0), out=buf476)
            del primals_201
            assert_size_stride(primals_202, (3072, ), (1, ), 'input')
            buf478 = reinterpret_tensor(buf469, (1, 512, 3072), (1572864, 3072, 1), 0); del buf469  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, hidden_states_220, pow_45, variance_44, add_154, rsqrt_44, hidden_states_221, to_116, hidden_states_222], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf470, buf476, primals_202, buf478, 512, 3072, stream=raw_stream0)
            del primals_202
            assert_size_stride(primals_203, (3072, 3072), (3072, 1), 'input')
            buf479 = buf454; del buf454  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, hidden_states_220, pow_45, variance_44, add_154, rsqrt_44, hidden_states_221, to_116, hidden_states_222, linear_154], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf478, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_203, (3072, 3072), (1, 3072), 0), out=buf479)
            del primals_203
            assert_size_stride(primals_204, (1024, 3072), (3072, 1), 'input')
            buf480 = buf459; del buf459  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, hidden_states_220, pow_45, variance_44, add_154, rsqrt_44, hidden_states_221, to_116, hidden_states_222, linear_154, linear_155], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf478, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_204, (3072, 1024), (1, 3072), 0), out=buf480)
            del primals_204
            assert_size_stride(primals_205, (1024, 3072), (3072, 1), 'input')
            buf481 = buf458; del buf458  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, hidden_states_220, pow_45, variance_44, add_154, rsqrt_44, hidden_states_221, to_116, hidden_states_222, linear_154, linear_156], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf478, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_205, (3072, 1024), (1, 3072), 0), out=buf481)
            del primals_205
            buf482 = reinterpret_tensor(buf478, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf478  # reuse
            buf483 = reinterpret_tensor(buf448, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf448  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_154, view_66, query_states_22, linear_155, view_67, key_states_44, mul_224, x1_44, x2_44, neg_44, cat_45, mul_225, q_embed_22, mul_226, x1_45, x2_45, neg_45, cat_46, mul_227, k_embed_22, getitem_161, hidden_states_223, key_states_45], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf479, primals_3, buf480, buf482, buf483, 1572864, stream=raw_stream0)
            buf484 = reinterpret_tensor(buf465, (24, 512, 512), (262144, 512, 1), 0); del buf465  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_154, view_66, query_states_22, linear_155, view_67, key_states_44, mul_224, x1_44, x2_44, neg_44, cat_45, mul_225, q_embed_22, mul_226, x1_45, x2_45, neg_45, cat_46, mul_227, k_embed_22, getitem_161, hidden_states_223, key_states_45, transpose_114, matmul_45], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf482, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf483, (24, 128, 512), (65536, 1, 128), 0), out=buf484)
            buf487 = reinterpret_tensor(buf484, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf484  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_45, attn_weights_88, attn_weights_89, softmax_22, attn_weights_90], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf487, buf3, 12288, 512, stream=raw_stream0)
            buf488 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [linear_156, view_68, value_states_44, getitem_162, hidden_states_224, value_states_45], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf481, buf488, 1572864, stream=raw_stream0)
            buf489 = reinterpret_tensor(buf482, (24, 512, 128), (65536, 128, 1), 0); del buf482  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_156, view_68, value_states_44, getitem_162, hidden_states_224, value_states_45, matmul_45, attn_weights_88, attn_weights_89, softmax_22, attn_weights_90, attn_output_88], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf487, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf488, (24, 512, 128), (65536, 128, 1), 0), out=buf489)
            buf490 = reinterpret_tensor(buf488, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf488  # reuse
            # Topologically Sorted Source Nodes: [attn_output_88, transpose_115, attn_output_89], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf489, buf490, 1572864, stream=raw_stream0)
            assert_size_stride(primals_206, (3072, 3072), (3072, 1), 'input')
            buf491 = reinterpret_tensor(buf489, (512, 3072), (3072, 1), 0); del buf489  # reuse
            # Topologically Sorted Source Nodes: [attn_output_88, transpose_115, attn_output_89, reshape_68, attn_output_91], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf490, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_206, (3072, 3072), (1, 3072), 0), out=buf491)
            del primals_206
            assert_size_stride(primals_207, (3072, ), (1, ), 'input')
            buf493 = reinterpret_tensor(buf490, (1, 512, 3072), (1572864, 3072, 1), 0); del buf490  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, hidden_states_226, pow_46, variance_45, add_159, rsqrt_45, hidden_states_227, to_119, hidden_states_228], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf470, buf476, buf491, primals_207, buf493, 512, 3072, stream=raw_stream0)
            del primals_207
            assert_size_stride(primals_208, (8192, 3072), (3072, 1), 'input')
            buf494 = reinterpret_tensor(buf475, (512, 8192), (8192, 1), 0); del buf475  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, hidden_states_226, pow_46, variance_45, add_159, rsqrt_45, hidden_states_227, to_119, hidden_states_228, linear_158], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf493, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_208, (3072, 8192), (1, 3072), 0), out=buf494)
            del primals_208
            assert_size_stride(primals_209, (8192, 3072), (3072, 1), 'input')
            buf495 = buf474; del buf474  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, hidden_states_226, pow_46, variance_45, add_159, rsqrt_45, hidden_states_227, to_119, hidden_states_228, linear_158, linear_159], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf493, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_209, (3072, 8192), (1, 3072), 0), out=buf495)
            del primals_209
            buf496 = reinterpret_tensor(buf494, (1, 512, 8192), (4194304, 8192, 1), 0); del buf494  # reuse
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_231], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf496, buf495, 4194304, stream=raw_stream0)
            assert_size_stride(primals_210, (3072, 8192), (8192, 1), 'input')
            buf497 = reinterpret_tensor(buf493, (512, 3072), (3072, 1), 0); del buf493  # reuse
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_231, down_proj_22], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf496, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_210, (8192, 3072), (1, 8192), 0), out=buf497)
            del primals_210
            assert_size_stride(primals_211, (3072, ), (1, ), 'input')
            buf499 = reinterpret_tensor(buf479, (1, 512, 3072), (1572864, 3072, 1), 0); del buf479  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, hidden_states_230, pow_47, variance_46, add_161, rsqrt_46, hidden_states_231, to_121, hidden_states_232], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf470, buf476, buf491, buf497, primals_211, buf499, 512, 3072, stream=raw_stream0)
            del primals_211
            assert_size_stride(primals_212, (3072, 3072), (3072, 1), 'input')
            buf500 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, hidden_states_230, pow_47, variance_46, add_161, rsqrt_46, hidden_states_231, to_121, hidden_states_232, linear_161], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf499, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_212, (3072, 3072), (1, 3072), 0), out=buf500)
            del primals_212
            assert_size_stride(primals_213, (1024, 3072), (3072, 1), 'input')
            buf501 = buf481; del buf481  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, hidden_states_230, pow_47, variance_46, add_161, rsqrt_46, hidden_states_231, to_121, hidden_states_232, linear_161, linear_162], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf499, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_213, (3072, 1024), (1, 3072), 0), out=buf501)
            del primals_213
            assert_size_stride(primals_214, (1024, 3072), (3072, 1), 'input')
            buf502 = buf480; del buf480  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, hidden_states_230, pow_47, variance_46, add_161, rsqrt_46, hidden_states_231, to_121, hidden_states_232, linear_161, linear_163], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf499, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_214, (3072, 1024), (1, 3072), 0), out=buf502)
            del primals_214
            buf503 = reinterpret_tensor(buf499, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf499  # reuse
            buf504 = reinterpret_tensor(buf457, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf457  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_161, view_69, query_states_23, linear_162, view_70, key_states_46, mul_234, x1_46, x2_46, neg_46, cat_47, mul_235, q_embed_23, mul_236, x1_47, x2_47, neg_47, cat_48, mul_237, k_embed_23, getitem_168, hidden_states_233, key_states_47], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf500, primals_3, buf501, buf503, buf504, 1572864, stream=raw_stream0)
            buf505 = reinterpret_tensor(buf487, (24, 512, 512), (262144, 512, 1), 0); del buf487  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_161, view_69, query_states_23, linear_162, view_70, key_states_46, mul_234, x1_46, x2_46, neg_46, cat_47, mul_235, q_embed_23, mul_236, x1_47, x2_47, neg_47, cat_48, mul_237, k_embed_23, getitem_168, hidden_states_233, key_states_47, transpose_119, matmul_47], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf503, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf504, (24, 128, 512), (65536, 1, 128), 0), out=buf505)
            buf508 = reinterpret_tensor(buf505, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_47, attn_weights_92, attn_weights_93, softmax_23, attn_weights_94], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf508, buf3, 12288, 512, stream=raw_stream0)
            buf509 = buf504; del buf504  # reuse
            # Topologically Sorted Source Nodes: [linear_163, view_71, value_states_46, getitem_169, hidden_states_234, value_states_47], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf502, buf509, 1572864, stream=raw_stream0)
            buf510 = reinterpret_tensor(buf503, (24, 512, 128), (65536, 128, 1), 0); del buf503  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_163, view_71, value_states_46, getitem_169, hidden_states_234, value_states_47, matmul_47, attn_weights_92, attn_weights_93, softmax_23, attn_weights_94, attn_output_92], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf508, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf509, (24, 512, 128), (65536, 128, 1), 0), out=buf510)
            buf511 = reinterpret_tensor(buf509, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf509  # reuse
            # Topologically Sorted Source Nodes: [attn_output_92, transpose_120, attn_output_93], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf510, buf511, 1572864, stream=raw_stream0)
            assert_size_stride(primals_215, (3072, 3072), (3072, 1), 'input')
            buf512 = reinterpret_tensor(buf510, (512, 3072), (3072, 1), 0); del buf510  # reuse
            # Topologically Sorted Source Nodes: [attn_output_92, transpose_120, attn_output_93, reshape_71, attn_output_95], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf511, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_215, (3072, 3072), (1, 3072), 0), out=buf512)
            del primals_215
            assert_size_stride(primals_216, (3072, ), (1, ), 'input')
            buf513 = buf470; del buf470  # reuse
            buf515 = reinterpret_tensor(buf511, (1, 512, 3072), (1572864, 3072, 1), 0); del buf511  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_219, attn_output_91, hidden_states_225, down_proj_22, hidden_states_229, attn_output_95, hidden_states_235, hidden_states_236, pow_48, variance_47, add_166, rsqrt_47, hidden_states_237, to_124, hidden_states_238], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf513, buf476, buf491, buf497, buf512, primals_216, buf515, 512, 3072, stream=raw_stream0)
            del primals_216
            assert_size_stride(primals_217, (8192, 3072), (3072, 1), 'input')
            buf516 = reinterpret_tensor(buf496, (512, 8192), (8192, 1), 0); del buf496  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_236, pow_48, variance_47, add_166, rsqrt_47, hidden_states_237, to_124, hidden_states_238, linear_165], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf515, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_217, (3072, 8192), (1, 3072), 0), out=buf516)
            del primals_217
            assert_size_stride(primals_218, (8192, 3072), (3072, 1), 'input')
            buf517 = buf495; del buf495  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_236, pow_48, variance_47, add_166, rsqrt_47, hidden_states_237, to_124, hidden_states_238, linear_165, linear_166], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf515, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_218, (3072, 8192), (1, 3072), 0), out=buf517)
            del primals_218
            buf518 = reinterpret_tensor(buf516, (1, 512, 8192), (4194304, 8192, 1), 0); del buf516  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_241], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf518, buf517, 4194304, stream=raw_stream0)
            assert_size_stride(primals_219, (3072, 8192), (8192, 1), 'input')
            buf519 = reinterpret_tensor(buf515, (512, 3072), (3072, 1), 0); del buf515  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_241, down_proj_23], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf518, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_219, (8192, 3072), (1, 8192), 0), out=buf519)
            del primals_219
            assert_size_stride(primals_220, (3072, ), (1, ), 'input')
            buf521 = reinterpret_tensor(buf512, (1, 512, 3072), (1572864, 3072, 1), 0); del buf512  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, hidden_states_240, pow_49, variance_48, add_168, rsqrt_48, hidden_states_241, to_126, hidden_states_242], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf513, buf519, primals_220, buf521, 512, 3072, stream=raw_stream0)
            del primals_220
            assert_size_stride(primals_221, (3072, 3072), (3072, 1), 'input')
            buf522 = buf497; del buf497  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, hidden_states_240, pow_49, variance_48, add_168, rsqrt_48, hidden_states_241, to_126, hidden_states_242, linear_168], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf521, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_221, (3072, 3072), (1, 3072), 0), out=buf522)
            del primals_221
            assert_size_stride(primals_222, (1024, 3072), (3072, 1), 'input')
            buf523 = buf502; del buf502  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, hidden_states_240, pow_49, variance_48, add_168, rsqrt_48, hidden_states_241, to_126, hidden_states_242, linear_168, linear_169], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf521, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_222, (3072, 1024), (1, 3072), 0), out=buf523)
            del primals_222
            assert_size_stride(primals_223, (1024, 3072), (3072, 1), 'input')
            buf524 = buf501; del buf501  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, hidden_states_240, pow_49, variance_48, add_168, rsqrt_48, hidden_states_241, to_126, hidden_states_242, linear_168, linear_170], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf521, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_223, (3072, 1024), (1, 3072), 0), out=buf524)
            del primals_223
            buf525 = reinterpret_tensor(buf521, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf521  # reuse
            buf526 = reinterpret_tensor(buf491, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf491  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_168, view_72, query_states_24, linear_169, view_73, key_states_48, mul_244, x1_48, x2_48, neg_48, cat_49, mul_245, q_embed_24, mul_246, x1_49, x2_49, neg_49, cat_50, mul_247, k_embed_24, getitem_175, hidden_states_243, key_states_49], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf522, primals_3, buf523, buf525, buf526, 1572864, stream=raw_stream0)
            buf527 = reinterpret_tensor(buf508, (24, 512, 512), (262144, 512, 1), 0); del buf508  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_168, view_72, query_states_24, linear_169, view_73, key_states_48, mul_244, x1_48, x2_48, neg_48, cat_49, mul_245, q_embed_24, mul_246, x1_49, x2_49, neg_49, cat_50, mul_247, k_embed_24, getitem_175, hidden_states_243, key_states_49, transpose_124, matmul_49], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf525, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf526, (24, 128, 512), (65536, 1, 128), 0), out=buf527)
            buf530 = reinterpret_tensor(buf527, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf527  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_49, attn_weights_96, attn_weights_97, softmax_24, attn_weights_98], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf530, buf3, 12288, 512, stream=raw_stream0)
            buf531 = buf526; del buf526  # reuse
            # Topologically Sorted Source Nodes: [linear_170, view_74, value_states_48, getitem_176, hidden_states_244, value_states_49], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf524, buf531, 1572864, stream=raw_stream0)
            buf532 = reinterpret_tensor(buf525, (24, 512, 128), (65536, 128, 1), 0); del buf525  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_170, view_74, value_states_48, getitem_176, hidden_states_244, value_states_49, matmul_49, attn_weights_96, attn_weights_97, softmax_24, attn_weights_98, attn_output_96], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf530, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf531, (24, 512, 128), (65536, 128, 1), 0), out=buf532)
            buf533 = reinterpret_tensor(buf531, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf531  # reuse
            # Topologically Sorted Source Nodes: [attn_output_96, transpose_125, attn_output_97], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf532, buf533, 1572864, stream=raw_stream0)
            assert_size_stride(primals_224, (3072, 3072), (3072, 1), 'input')
            buf534 = reinterpret_tensor(buf532, (512, 3072), (3072, 1), 0); del buf532  # reuse
            # Topologically Sorted Source Nodes: [attn_output_96, transpose_125, attn_output_97, reshape_74, attn_output_99], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf533, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_224, (3072, 3072), (1, 3072), 0), out=buf534)
            del primals_224
            assert_size_stride(primals_225, (3072, ), (1, ), 'input')
            buf536 = reinterpret_tensor(buf533, (1, 512, 3072), (1572864, 3072, 1), 0); del buf533  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, hidden_states_246, pow_50, variance_49, add_173, rsqrt_49, hidden_states_247, to_129, hidden_states_248], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf513, buf519, buf534, primals_225, buf536, 512, 3072, stream=raw_stream0)
            del primals_225
            assert_size_stride(primals_226, (8192, 3072), (3072, 1), 'input')
            buf537 = reinterpret_tensor(buf518, (512, 8192), (8192, 1), 0); del buf518  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, hidden_states_246, pow_50, variance_49, add_173, rsqrt_49, hidden_states_247, to_129, hidden_states_248, linear_172], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf536, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_226, (3072, 8192), (1, 3072), 0), out=buf537)
            del primals_226
            assert_size_stride(primals_227, (8192, 3072), (3072, 1), 'input')
            buf538 = buf517; del buf517  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, hidden_states_246, pow_50, variance_49, add_173, rsqrt_49, hidden_states_247, to_129, hidden_states_248, linear_172, linear_173], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf536, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_227, (3072, 8192), (1, 3072), 0), out=buf538)
            del primals_227
            buf539 = reinterpret_tensor(buf537, (1, 512, 8192), (4194304, 8192, 1), 0); del buf537  # reuse
            # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_251], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf539, buf538, 4194304, stream=raw_stream0)
            assert_size_stride(primals_228, (3072, 8192), (8192, 1), 'input')
            buf540 = reinterpret_tensor(buf536, (512, 3072), (3072, 1), 0); del buf536  # reuse
            # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_251, down_proj_24], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf539, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_228, (8192, 3072), (1, 8192), 0), out=buf540)
            del primals_228
            assert_size_stride(primals_229, (3072, ), (1, ), 'input')
            buf542 = reinterpret_tensor(buf522, (1, 512, 3072), (1572864, 3072, 1), 0); del buf522  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, down_proj_24, hidden_states_249, hidden_states_250, pow_51, variance_50, add_175, rsqrt_50, hidden_states_251, to_131, hidden_states_252], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf513, buf519, buf534, buf540, primals_229, buf542, 512, 3072, stream=raw_stream0)
            del primals_229
            assert_size_stride(primals_230, (3072, 3072), (3072, 1), 'input')
            buf543 = buf476; del buf476  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, down_proj_24, hidden_states_249, hidden_states_250, pow_51, variance_50, add_175, rsqrt_50, hidden_states_251, to_131, hidden_states_252, linear_175], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf542, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_230, (3072, 3072), (1, 3072), 0), out=buf543)
            del primals_230
            assert_size_stride(primals_231, (1024, 3072), (3072, 1), 'input')
            buf544 = buf524; del buf524  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, down_proj_24, hidden_states_249, hidden_states_250, pow_51, variance_50, add_175, rsqrt_50, hidden_states_251, to_131, hidden_states_252, linear_175, linear_176], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf542, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_231, (3072, 1024), (1, 3072), 0), out=buf544)
            del primals_231
            assert_size_stride(primals_232, (1024, 3072), (3072, 1), 'input')
            buf545 = buf523; del buf523  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, down_proj_24, hidden_states_249, hidden_states_250, pow_51, variance_50, add_175, rsqrt_50, hidden_states_251, to_131, hidden_states_252, linear_175, linear_177], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf542, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_232, (3072, 1024), (1, 3072), 0), out=buf545)
            del primals_232
            buf546 = reinterpret_tensor(buf542, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf542  # reuse
            buf547 = reinterpret_tensor(buf500, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf500  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_175, view_75, query_states_25, linear_176, view_76, key_states_50, mul_254, x1_50, x2_50, neg_50, cat_51, mul_255, q_embed_25, mul_256, x1_51, x2_51, neg_51, cat_52, mul_257, k_embed_25, getitem_182, hidden_states_253, key_states_51], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf543, primals_3, buf544, buf546, buf547, 1572864, stream=raw_stream0)
            buf548 = reinterpret_tensor(buf530, (24, 512, 512), (262144, 512, 1), 0); del buf530  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_175, view_75, query_states_25, linear_176, view_76, key_states_50, mul_254, x1_50, x2_50, neg_50, cat_51, mul_255, q_embed_25, mul_256, x1_51, x2_51, neg_51, cat_52, mul_257, k_embed_25, getitem_182, hidden_states_253, key_states_51, transpose_129, matmul_51], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf546, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf547, (24, 128, 512), (65536, 1, 128), 0), out=buf548)
            buf551 = reinterpret_tensor(buf548, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf548  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_51, attn_weights_100, attn_weights_101, softmax_25, attn_weights_102], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf551, buf3, 12288, 512, stream=raw_stream0)
            buf552 = buf547; del buf547  # reuse
            # Topologically Sorted Source Nodes: [linear_177, view_77, value_states_50, getitem_183, hidden_states_254, value_states_51], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf545, buf552, 1572864, stream=raw_stream0)
            buf553 = reinterpret_tensor(buf546, (24, 512, 128), (65536, 128, 1), 0); del buf546  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_177, view_77, value_states_50, getitem_183, hidden_states_254, value_states_51, matmul_51, attn_weights_100, attn_weights_101, softmax_25, attn_weights_102, attn_output_100], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf551, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf552, (24, 512, 128), (65536, 128, 1), 0), out=buf553)
            buf554 = reinterpret_tensor(buf552, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf552  # reuse
            # Topologically Sorted Source Nodes: [attn_output_100, transpose_130, attn_output_101], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf553, buf554, 1572864, stream=raw_stream0)
            assert_size_stride(primals_233, (3072, 3072), (3072, 1), 'input')
            buf555 = reinterpret_tensor(buf553, (512, 3072), (3072, 1), 0); del buf553  # reuse
            # Topologically Sorted Source Nodes: [attn_output_100, transpose_130, attn_output_101, reshape_77, attn_output_103], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf554, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_233, (3072, 3072), (1, 3072), 0), out=buf555)
            del primals_233
            assert_size_stride(primals_234, (3072, ), (1, ), 'input')
            buf556 = buf513; del buf513  # reuse
            buf558 = reinterpret_tensor(buf554, (1, 512, 3072), (1572864, 3072, 1), 0); del buf554  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_239, attn_output_99, hidden_states_245, down_proj_24, hidden_states_249, attn_output_103, hidden_states_255, hidden_states_256, pow_52, variance_51, add_180, rsqrt_51, hidden_states_257, to_134, hidden_states_258], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf556, buf519, buf534, buf540, buf555, primals_234, buf558, 512, 3072, stream=raw_stream0)
            del primals_234
            assert_size_stride(primals_235, (8192, 3072), (3072, 1), 'input')
            buf559 = reinterpret_tensor(buf539, (512, 8192), (8192, 1), 0); del buf539  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_256, pow_52, variance_51, add_180, rsqrt_51, hidden_states_257, to_134, hidden_states_258, linear_179], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf558, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_235, (3072, 8192), (1, 3072), 0), out=buf559)
            del primals_235
            assert_size_stride(primals_236, (8192, 3072), (3072, 1), 'input')
            buf560 = buf538; del buf538  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_256, pow_52, variance_51, add_180, rsqrt_51, hidden_states_257, to_134, hidden_states_258, linear_179, linear_180], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf558, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_236, (3072, 8192), (1, 3072), 0), out=buf560)
            del primals_236
            buf561 = reinterpret_tensor(buf559, (1, 512, 8192), (4194304, 8192, 1), 0); del buf559  # reuse
            # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_261], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf561, buf560, 4194304, stream=raw_stream0)
            assert_size_stride(primals_237, (3072, 8192), (8192, 1), 'input')
            buf562 = reinterpret_tensor(buf558, (512, 3072), (3072, 1), 0); del buf558  # reuse
            # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_261, down_proj_25], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf561, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_237, (8192, 3072), (1, 8192), 0), out=buf562)
            del primals_237
            assert_size_stride(primals_238, (3072, ), (1, ), 'input')
            buf564 = reinterpret_tensor(buf555, (1, 512, 3072), (1572864, 3072, 1), 0); del buf555  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, hidden_states_260, pow_53, variance_52, add_182, rsqrt_52, hidden_states_261, to_136, hidden_states_262], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf556, buf562, primals_238, buf564, 512, 3072, stream=raw_stream0)
            del primals_238
            assert_size_stride(primals_239, (3072, 3072), (3072, 1), 'input')
            buf565 = buf540; del buf540  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, hidden_states_260, pow_53, variance_52, add_182, rsqrt_52, hidden_states_261, to_136, hidden_states_262, linear_182], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf564, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_239, (3072, 3072), (1, 3072), 0), out=buf565)
            del primals_239
            assert_size_stride(primals_240, (1024, 3072), (3072, 1), 'input')
            buf566 = buf545; del buf545  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, hidden_states_260, pow_53, variance_52, add_182, rsqrt_52, hidden_states_261, to_136, hidden_states_262, linear_182, linear_183], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf564, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_240, (3072, 1024), (1, 3072), 0), out=buf566)
            del primals_240
            assert_size_stride(primals_241, (1024, 3072), (3072, 1), 'input')
            buf567 = buf544; del buf544  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, hidden_states_260, pow_53, variance_52, add_182, rsqrt_52, hidden_states_261, to_136, hidden_states_262, linear_182, linear_184], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf564, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_241, (3072, 1024), (1, 3072), 0), out=buf567)
            del primals_241
            buf568 = reinterpret_tensor(buf564, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf564  # reuse
            buf569 = reinterpret_tensor(buf534, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf534  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_182, view_78, query_states_26, linear_183, view_79, key_states_52, mul_264, x1_52, x2_52, neg_52, cat_53, mul_265, q_embed_26, mul_266, x1_53, x2_53, neg_53, cat_54, mul_267, k_embed_26, getitem_189, hidden_states_263, key_states_53], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf565, primals_3, buf566, buf568, buf569, 1572864, stream=raw_stream0)
            buf570 = reinterpret_tensor(buf551, (24, 512, 512), (262144, 512, 1), 0); del buf551  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_182, view_78, query_states_26, linear_183, view_79, key_states_52, mul_264, x1_52, x2_52, neg_52, cat_53, mul_265, q_embed_26, mul_266, x1_53, x2_53, neg_53, cat_54, mul_267, k_embed_26, getitem_189, hidden_states_263, key_states_53, transpose_134, matmul_53], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf568, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf569, (24, 128, 512), (65536, 1, 128), 0), out=buf570)
            buf573 = reinterpret_tensor(buf570, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf570  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_53, attn_weights_104, attn_weights_105, softmax_26, attn_weights_106], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf573, buf3, 12288, 512, stream=raw_stream0)
            buf574 = buf569; del buf569  # reuse
            # Topologically Sorted Source Nodes: [linear_184, view_80, value_states_52, getitem_190, hidden_states_264, value_states_53], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf567, buf574, 1572864, stream=raw_stream0)
            buf575 = reinterpret_tensor(buf568, (24, 512, 128), (65536, 128, 1), 0); del buf568  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_184, view_80, value_states_52, getitem_190, hidden_states_264, value_states_53, matmul_53, attn_weights_104, attn_weights_105, softmax_26, attn_weights_106, attn_output_104], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf573, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf574, (24, 512, 128), (65536, 128, 1), 0), out=buf575)
            buf576 = reinterpret_tensor(buf574, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf574  # reuse
            # Topologically Sorted Source Nodes: [attn_output_104, transpose_135, attn_output_105], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf575, buf576, 1572864, stream=raw_stream0)
            assert_size_stride(primals_242, (3072, 3072), (3072, 1), 'input')
            buf577 = reinterpret_tensor(buf575, (512, 3072), (3072, 1), 0); del buf575  # reuse
            # Topologically Sorted Source Nodes: [attn_output_104, transpose_135, attn_output_105, reshape_80, attn_output_107], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf576, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_242, (3072, 3072), (1, 3072), 0), out=buf577)
            del primals_242
            assert_size_stride(primals_243, (3072, ), (1, ), 'input')
            buf579 = reinterpret_tensor(buf576, (1, 512, 3072), (1572864, 3072, 1), 0); del buf576  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, hidden_states_266, pow_54, variance_53, add_187, rsqrt_53, hidden_states_267, to_139, hidden_states_268], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf556, buf562, buf577, primals_243, buf579, 512, 3072, stream=raw_stream0)
            del primals_243
            assert_size_stride(primals_244, (8192, 3072), (3072, 1), 'input')
            buf580 = reinterpret_tensor(buf561, (512, 8192), (8192, 1), 0); del buf561  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, hidden_states_266, pow_54, variance_53, add_187, rsqrt_53, hidden_states_267, to_139, hidden_states_268, linear_186], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf579, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_244, (3072, 8192), (1, 3072), 0), out=buf580)
            del primals_244
            assert_size_stride(primals_245, (8192, 3072), (3072, 1), 'input')
            buf581 = buf560; del buf560  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, hidden_states_266, pow_54, variance_53, add_187, rsqrt_53, hidden_states_267, to_139, hidden_states_268, linear_186, linear_187], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf579, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_245, (3072, 8192), (1, 3072), 0), out=buf581)
            del primals_245
            buf582 = reinterpret_tensor(buf580, (1, 512, 8192), (4194304, 8192, 1), 0); del buf580  # reuse
            # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_271], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf582, buf581, 4194304, stream=raw_stream0)
            assert_size_stride(primals_246, (3072, 8192), (8192, 1), 'input')
            buf583 = reinterpret_tensor(buf579, (512, 3072), (3072, 1), 0); del buf579  # reuse
            # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_271, down_proj_26], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf582, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_246, (8192, 3072), (1, 8192), 0), out=buf583)
            del primals_246
            assert_size_stride(primals_247, (3072, ), (1, ), 'input')
            buf585 = reinterpret_tensor(buf565, (1, 512, 3072), (1572864, 3072, 1), 0); del buf565  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, down_proj_26, hidden_states_269, hidden_states_270, pow_55, variance_54, add_189, rsqrt_54, hidden_states_271, to_141, hidden_states_272], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf556, buf562, buf577, buf583, primals_247, buf585, 512, 3072, stream=raw_stream0)
            del primals_247
            assert_size_stride(primals_248, (3072, 3072), (3072, 1), 'input')
            buf586 = buf519; del buf519  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, down_proj_26, hidden_states_269, hidden_states_270, pow_55, variance_54, add_189, rsqrt_54, hidden_states_271, to_141, hidden_states_272, linear_189], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf585, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_248, (3072, 3072), (1, 3072), 0), out=buf586)
            del primals_248
            assert_size_stride(primals_249, (1024, 3072), (3072, 1), 'input')
            buf587 = buf567; del buf567  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, down_proj_26, hidden_states_269, hidden_states_270, pow_55, variance_54, add_189, rsqrt_54, hidden_states_271, to_141, hidden_states_272, linear_189, linear_190], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf585, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_249, (3072, 1024), (1, 3072), 0), out=buf587)
            del primals_249
            assert_size_stride(primals_250, (1024, 3072), (3072, 1), 'input')
            buf588 = buf566; del buf566  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, down_proj_26, hidden_states_269, hidden_states_270, pow_55, variance_54, add_189, rsqrt_54, hidden_states_271, to_141, hidden_states_272, linear_189, linear_191], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf585, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_250, (3072, 1024), (1, 3072), 0), out=buf588)
            del primals_250
            buf589 = reinterpret_tensor(buf585, (1, 24, 512, 128), (1572864, 65536, 128, 1), 0); del buf585  # reuse
            buf590 = reinterpret_tensor(buf543, (1, 8, 3, 512, 128), (1572864, 196608, 65536, 128, 1), 0); del buf543  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_189, view_81, query_states_27, linear_190, view_82, key_states_54, mul_274, x1_54, x2_54, neg_54, cat_55, mul_275, q_embed_27, mul_276, x1_55, x2_55, neg_55, cat_56, mul_277, k_embed_27, getitem_196, hidden_states_273, key_states_55], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(buf586, primals_3, buf587, buf589, buf590, 1572864, stream=raw_stream0)
            del buf586
            del buf587
            del primals_3
            buf591 = reinterpret_tensor(buf573, (24, 512, 512), (262144, 512, 1), 0); del buf573  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, cos, cos_1, sin, sin_1, cos_2, sin_2, cos_3, sin_3, linear_189, view_81, query_states_27, linear_190, view_82, key_states_54, mul_274, x1_54, x2_54, neg_54, cat_55, mul_275, q_embed_27, mul_276, x1_55, x2_55, neg_55, cat_56, mul_277, k_embed_27, getitem_196, hidden_states_273, key_states_55, transpose_139, matmul_55], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            extern_kernels.bmm(reinterpret_tensor(buf589, (24, 512, 128), (65536, 128, 1), 0), reinterpret_tensor(buf590, (24, 128, 512), (65536, 1, 128), 0), out=buf591)
            buf594 = reinterpret_tensor(buf591, (1, 24, 512, 512), (6291456, 262144, 512, 1), 0); del buf591  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, matmul_55, attn_weights_108, attn_weights_109, softmax_27, attn_weights_110], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten.mul, aten._to_copy, prims.prepare_softmax_online, aten._softmax]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__to_copy_add_arange_bitwise_and_eq_index_le_lift_fresh_mul_prepare_softmax_online_scalar_tensor_view_where_5.run(buf594, buf3, 12288, 512, stream=raw_stream0)
            del buf3
            buf595 = buf590; del buf590  # reuse
            # Topologically Sorted Source Nodes: [linear_191, view_83, value_states_54, getitem_197, hidden_states_274, value_states_55], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_expand_transpose_unsqueeze_view_6.run(buf588, buf595, 1572864, stream=raw_stream0)
            del buf588
            buf596 = reinterpret_tensor(buf589, (24, 512, 128), (65536, 128, 1), 0); del buf589  # reuse
            # Topologically Sorted Source Nodes: [cache_position, kv_arange_1, batch_arange, le, result_1, index, index_1, eq, result_2, batched_outputs_2, tensor, mask, linear_191, view_83, value_states_54, getitem_197, hidden_states_274, value_states_55, matmul_55, attn_weights_108, attn_weights_109, softmax_27, attn_weights_110, attn_output_108], Original ATen: [aten.arange, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.eq, aten.lift_fresh, aten.scalar_tensor, aten.where, aten._unsafe_view, aten.transpose, aten.unsqueeze, aten.expand, aten.clone, aten.mul, aten._to_copy, aten._softmax, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf594, (24, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf595, (24, 512, 128), (65536, 128, 1), 0), out=buf596)
            del buf594
            buf597 = reinterpret_tensor(buf595, (1, 512, 24, 128), (1572864, 3072, 128, 1), 0); del buf595  # reuse
            # Topologically Sorted Source Nodes: [attn_output_108, transpose_140, attn_output_109], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_7.run(buf596, buf597, 1572864, stream=raw_stream0)
            assert_size_stride(primals_251, (3072, 3072), (3072, 1), 'input')
            buf598 = reinterpret_tensor(buf596, (512, 3072), (3072, 1), 0); del buf596  # reuse
            # Topologically Sorted Source Nodes: [attn_output_108, transpose_140, attn_output_109, reshape_83, attn_output_111], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf597, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_251, (3072, 3072), (1, 3072), 0), out=buf598)
            del primals_251
            assert_size_stride(primals_252, (3072, ), (1, ), 'input')
            buf599 = buf556; del buf556  # reuse
            buf601 = reinterpret_tensor(buf597, (1, 512, 3072), (1572864, 3072, 1), 0); del buf597  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_259, attn_output_107, hidden_states_265, down_proj_26, hidden_states_269, attn_output_111, hidden_states_275, hidden_states_276, pow_56, variance_55, add_194, rsqrt_55, hidden_states_277, to_144, hidden_states_278], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf599, buf562, buf577, buf583, buf598, primals_252, buf601, 512, 3072, stream=raw_stream0)
            del buf562
            del buf577
            del buf583
            del primals_252
            assert_size_stride(primals_253, (8192, 3072), (3072, 1), 'input')
            buf602 = reinterpret_tensor(buf582, (512, 8192), (8192, 1), 0); del buf582  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_276, pow_56, variance_55, add_194, rsqrt_55, hidden_states_277, to_144, hidden_states_278, linear_193], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf601, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_253, (3072, 8192), (1, 3072), 0), out=buf602)
            del primals_253
            assert_size_stride(primals_254, (8192, 3072), (3072, 1), 'input')
            buf603 = buf581; del buf581  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_276, pow_56, variance_55, add_194, rsqrt_55, hidden_states_277, to_144, hidden_states_278, linear_193, linear_194], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf601, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_254, (3072, 8192), (1, 3072), 0), out=buf603)
            del primals_254
            buf604 = reinterpret_tensor(buf602, (1, 512, 8192), (4194304, 8192, 1), 0); del buf602  # reuse
            # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_281], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_9.run(buf604, buf603, 4194304, stream=raw_stream0)
            del buf603
            assert_size_stride(primals_255, (3072, 8192), (8192, 1), 'input')
            buf605 = reinterpret_tensor(buf601, (512, 3072), (3072, 1), 0); del buf601  # reuse
            # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_281, down_proj_27], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf604, (512, 8192), (8192, 1), 0), reinterpret_tensor(primals_255, (8192, 3072), (1, 8192), 0), out=buf605)
            del buf604
            del primals_255
            assert_size_stride(primals_256, (3072, ), (1, ), 'input')
            buf607 = buf599; del buf599  # reuse
            buf608 = reinterpret_tensor(buf598, (1, 512, 3072), (1572864, 3072, 1), 0); del buf598  # reuse
            # Topologically Sorted Source Nodes: [down_proj_27, hidden_states_279, hidden_states_280, pow_57, variance_56, add_196, rsqrt_56, hidden_states_281, to_146, hidden_states_282], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_16.run(buf607, buf605, primals_256, buf608, 512, 3072, stream=raw_stream0)
            del buf605
            del primals_256
            buf609 = empty_strided_cuda((512, 128256), (128256, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_282, logits], Original ATen: [aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf608, (512, 3072), (3072, 1), 0), reinterpret_tensor(primals_2, (3072, 128256), (1, 3072), 0), out=buf609)
            del buf608
            buf610 = empty_strided_cuda((1, 513), (513, 1), torch.int64)
            # Topologically Sorted Source Nodes: [labels], Original ATen: [aten.constant_pad_nd]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_constant_pad_nd_17.run(primals_1, buf610, 513, stream=raw_stream0)
            del primals_1
            buf611 = empty_strided_cuda((512, 1), (1, 1), torch.float32)
            buf612 = empty_strided_cuda((512, 1), (1, 512), torch.float32)
            buf613 = reinterpret_tensor(buf612, (512, 1), (1, 1), 0); del buf612  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_1, logits_2, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.view, prims.prepare_softmax_online, aten._log_softmax]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__to_copy__unsafe_view_prepare_softmax_online_view_18.run(buf613, buf609, buf611, 512, 128256, stream=raw_stream0)
            buf616 = empty_strided_cuda((), (), torch.float32)
            buf615 = empty_strided_cuda((), (), torch.float32)
            buf617 = buf616; del buf616  # reuse
            # Topologically Sorted Source Nodes: [logits, logits_1, getitem_200, logits_2, shift_labels_1, loss], Original ATen: [aten._unsafe_view, aten._to_copy, aten.slice, aten.view, aten._log_softmax, aten.nll_loss_forward]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax__to_copy__unsafe_view_nll_loss_forward_slice_view_19.run(buf617, buf610, buf609, buf611, buf613, buf615, 1, 512, stream=raw_stream0)
        return (buf617, primals_2, buf607, buf609, buf610, buf611, buf613, buf615, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
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
