# AOT ID: ['0_backward']
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


# kernel path: /tmp/torchinductor_tzh/y3/cy3ppphpwg4xh6wsoulkqbkoxqxlfcdwk5dv2shvo3zb6364ker3.py
# Topologically Sorted Source Nodes: [div_57, getitem_200, shift_labels_1, unsqueeze_118, ne_4, loss, where_3, where_self, where_4, mul_285, logits, logits_1, logits_2, exp_57, sum_32, mul_286, sub_32, view_743, convert_element_type_737], Original ATen: [aten.nll_loss_backward, aten.slice, aten.view, aten.nll_loss_forward, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._unsafe_view, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   convert_element_type_737 => convert_element_type_737
#   div_57 => div_57
#   exp_57 => exp_57
#   getitem_200 => slice_116
#   logits => view_740
#   logits_1 => convert_element_type_735
#   logits_2 => view_741
#   loss => full_default_3, full_default_4, sub_30, sub_31
#   mul_285 => mul_285
#   mul_286 => mul_286
#   ne_4 => ne_4
#   shift_labels_1 => view_742
#   sub_32 => sub_32
#   sum_32 => sum_32
#   unsqueeze_118 => unsqueeze_118
#   view_743 => view_743
#   where_3 => where_3
#   where_4 => where_4
#   where_self => where_self
# Graph fragment:
#   %constant_pad_nd : Tensor "i64[1, 129][129, 1]cuda:0" = PlaceHolder[target=constant_pad_nd]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type_736 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_736]
#   %mm_196 : Tensor "bf16[128, 128256][128256, 1]cuda:0" = PlaceHolder[target=mm_196]
#   %amax_28 : Tensor "f32[128, 1][1, 1]cuda:0" = PlaceHolder[target=amax_28]
#   %log : Tensor "f32[128, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_32 : Tensor "f32[128, 1][1, 128]cuda:0" = PlaceHolder[target=sum_32]
#   %div_57 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_736), kwargs = {})
#   %slice_116 : Tensor "i64[1, 128][129, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, 1, 1, 9223372036854775807), kwargs = {})
#   %view_742 : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_116, [-1]), kwargs = {})
#   %unsqueeze_118 : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_742, 1), kwargs = {})
#   %ne_4 : Tensor "b8[128, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_118, -100), kwargs = {})
#   %full_default_3 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "i64[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %unsqueeze_118, %full_default_3), kwargs = {})
#   %iota_default : Tensor "i64[128256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_default : Tensor "i64[1, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_default, [1, 128256]), kwargs = {})
#   %expand_default : Tensor "i64[128, 128256][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_3, [128, 128256]), kwargs = {})
#   %eq_tensor : Tensor "b8[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_default, %view_default), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0})
#   %scalar_tensor_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0})
#   %where_self : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %div_57, %full_default_4), kwargs = {})
#   %mul_285 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_4), kwargs = {})
#   %view_740 : Tensor "bf16[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 128, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %view_741 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %sub_30 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_741, %amax_28), kwargs = {})
#   %sub_31 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_30, %log), kwargs = {})
#   %exp_57 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_31,), kwargs = {})
#   %sum_32 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_285, [1], True), kwargs = {})
#   %mul_286 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_57, %sum_32), kwargs = {})
#   %sub_32 : Tensor "f32[128, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_285, %mul_286), kwargs = {})
#   %view_743 : Tensor "f32[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_32, [1, 128, 128256]), kwargs = {})
#   %convert_element_type_737 : Tensor "bf16[1, 128, 128256][16416768, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_743, torch.bfloat16), kwargs = {})
#   return %sum_32,%convert_element_type_737
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 2048, 'r0_': 98500608}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 128256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [1, 1])
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [1, 1])
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = r0_1
        tmp6 = tmp4 == tmp5
        tmp7 = tl.full([1, 1], -1.0, tl.float32)
        tmp8 = tl.full([1, 1], 0.0, tl.float32)
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp14 = (tmp11 / tmp13)
        tmp15 = tl.where(tmp2, tmp14, tmp8)
        tmp16 = tmp9 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tmp29 = tl.load(in_ptr1 + (0))
    tmp30 = tl.broadcast_to(tmp29, [1, 1])
    tmp31 = tl.load(in_ptr2 + (0))
    tmp32 = tl.broadcast_to(tmp31, [1, 1])
    tmp38 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp36 = tl.load(in_out_ptr0 + (r0_1 + 128256*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.full([1, 1], -100, tl.int64)
        tmp21 = tmp0 != tmp20
        tmp22 = tl.full([1, 1], 0, tl.int64)
        tmp23 = tl.where(tmp21, tmp0, tmp22)
        tmp24 = r0_1
        tmp25 = tmp23 == tmp24
        tmp26 = tl.full([1, 1], -1.0, tl.float32)
        tmp27 = tl.full([1, 1], 0.0, tl.float32)
        tmp28 = tl.where(tmp25, tmp26, tmp27)
        tmp33 = (tmp30 / tmp32)
        tmp34 = tl.where(tmp21, tmp33, tmp27)
        tmp35 = tmp28 * tmp34
        tmp37 = tmp36.to(tl.float32)
        tmp39 = tmp37 - tmp38
        tmp41 = tmp39 - tmp40
        tmp42 = libdevice.exp(tmp41)
        tmp43 = tmp42 * tmp18
        tmp44 = tmp35 - tmp43
        tmp45 = tmp44.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 128256*x0), tmp45, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/zw/czwnhg7dghe6q7u3jxlymhhulbqhgovfd2ey5pa6s7tvle5qln3e.py
# Topologically Sorted Source Nodes: [full_default_121], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_default_121 => full_default_121
# Graph fragment:
#   %full_default_121 : Tensor "f32[128256, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([128256, 3072], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %index_put
triton_poi_fused_embedding_dense_backward_1 = async_compile.triton('triton_poi_fused_embedding_dense_backward_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_embedding_dense_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 3152019456}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 394002432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/q4/cq45dwryp3sepqu5ilbad4t7gs7mv6bsmwzr553uu2aoh3nboojx.py
# Topologically Sorted Source Nodes: [view_745, hidden_states_280, hidden_states_281, to_146, mul_288, sum_33], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   hidden_states_280 => convert_element_type_731
#   hidden_states_281 => mul_283
#   mul_288 => mul_288
#   sum_33 => sum_33
#   to_146 => convert_element_type_732
#   view_745 => view_745
# Graph fragment:
#   %mm_198 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_198]
#   %add_224 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_224]
#   %rsqrt_56 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_56]
#   %view_745 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_198, [1, 128, 3072]), kwargs = {})
#   %convert_element_type_731 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_224, torch.float32), kwargs = {})
#   %mul_283 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_731, %rsqrt_56), kwargs = {})
#   %convert_element_type_732 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_283, torch.bfloat16), kwargs = {})
#   %mul_288 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_745, %convert_element_type_732), kwargs = {})
#   %sum_33 : Tensor "bf16[1, 1, 3072][3072, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_288, [0, 1], True), kwargs = {})
#   return %sum_33
triton_red_fused__to_copy_mul_sum_view_2 = async_compile.triton('triton_red_fused__to_copy_mul_sum_view_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_mul_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 1585152, 'r0_': 512}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_mul_sum_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3072
    r0_numel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/a3/ca37fch5lhefmsdjj7f3p5qqjd5kt2se2ili2w6axzqr7dfvqghy.py
# Topologically Sorted Source Nodes: [view_745, mul_287, hidden_states_280, convert_element_type_742, mul_289, mul_290, sum_34, pow_58, mul_291, mul_292, expand_173, div_58, pow_59, mul_293, mul_294, add_226, convert_element_type_743], Original ATen: [aten.view, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div, aten.add]
# Source node to ATen node mapping:
#   add_226 => add_226
#   convert_element_type_742 => convert_element_type_742
#   convert_element_type_743 => convert_element_type_743
#   div_58 => div_58
#   expand_173 => expand_173
#   hidden_states_280 => convert_element_type_731
#   mul_287 => mul_287
#   mul_289 => mul_289
#   mul_290 => mul_290
#   mul_291 => mul_291
#   mul_292 => mul_292
#   mul_293 => mul_293
#   mul_294 => mul_294
#   pow_58 => pow_58
#   pow_59 => pow_59
#   sum_34 => sum_34
#   view_745 => view_745
# Graph fragment:
#   %mm_198 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_198]
#   %primals_256 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_256]
#   %add_224 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_224]
#   %rsqrt_56 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_56]
#   %sum_34 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=sum_34]
#   %view_745 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_198, [1, 128, 3072]), kwargs = {})
#   %mul_287 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_745, %primals_256), kwargs = {})
#   %convert_element_type_731 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_224, torch.float32), kwargs = {})
#   %convert_element_type_742 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_287, torch.float32), kwargs = {})
#   %mul_289 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_742, %convert_element_type_731), kwargs = {})
#   %mul_290 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_742, %rsqrt_56), kwargs = {})
#   %sum_34 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_289, [2], True), kwargs = {})
#   %pow_58 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt_56, 3), kwargs = {})
#   %mul_291 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_34, -0.5), kwargs = {})
#   %mul_292 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_291, %pow_58), kwargs = {})
#   %expand_173 : Tensor "f32[1, 128, 3072][128, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_292, [1, 128, 3072]), kwargs = {})
#   %div_58 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_173, 3072), kwargs = {})
#   %pow_59 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_731, 1.0), kwargs = {})
#   %mul_293 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_59, 2.0), kwargs = {})
#   %mul_294 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_58, %mul_293), kwargs = {})
#   %add_226 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_290, %mul_294), kwargs = {})
#   %convert_element_type_743 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_226, torch.bfloat16), kwargs = {})
#   return %sum_34,%convert_element_type_743
triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_3 = async_compile.triton('triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 512, 'r0_': 3151872}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp14 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp10 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 * tmp14
        tmp16 = tl.full([1, 1], -0.5, tl.float32)
        tmp17 = tmp8 * tmp16
        tmp18 = tmp14 * tmp14
        tmp19 = tmp18 * tmp14
        tmp20 = tmp17 * tmp19
        tmp21 = tl.full([1, 1], 0.0003255208333333333, tl.float32)
        tmp22 = tmp20 * tmp21
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tl.full([1, 1], 2.0, tl.float32)
        tmp26 = tmp24 * tmp25
        tmp27 = tmp22 * tmp26
        tmp28 = tmp15 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp29, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/b5/cb5n7knm2n3vnjlsyzrwmk2afidwsslcsw67admpsho5ioscgt2o.py
# Topologically Sorted Source Nodes: [view_748, linear_193, silu_27, mul_295, linear_194, mul_296, convert_element_type_752, reciprocal, mul_297, mul_298, sub_33, mul_299, add_228, mul_300, convert_element_type_754], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
# Source node to ATen node mapping:
#   add_228 => add_228
#   convert_element_type_752 => convert_element_type_752
#   convert_element_type_754 => convert_element_type_754
#   linear_193 => view_734
#   linear_194 => view_736
#   mul_295 => mul_295
#   mul_296 => mul_296
#   mul_297 => mul_297
#   mul_298 => mul_298
#   mul_299 => mul_299
#   mul_300 => mul_300
#   reciprocal => reciprocal
#   silu_27 => add_223, convert_element_type_725, convert_element_type_726, div_55, exp_55, neg_83
#   sub_33 => sub_33
#   view_748 => view_748
# Graph fragment:
#   %mm_200 : Tensor "bf16[128, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_200]
#   %mm_193 : Tensor "bf16[128, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_193]
#   %mm_194 : Tensor "bf16[128, 8192][8192, 1]cuda:0" = PlaceHolder[target=mm_194]
#   %view_748 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_200, [1, 128, 8192]), kwargs = {})
#   %view_734 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_193, [1, 128, 8192]), kwargs = {})
#   %convert_element_type_725 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_734, torch.float32), kwargs = {})
#   %neg_83 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_725,), kwargs = {})
#   %exp_55 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_83,), kwargs = {})
#   %add_223 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp_55, 1), kwargs = {})
#   %div_55 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_725, %add_223), kwargs = {})
#   %convert_element_type_726 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_55, torch.bfloat16), kwargs = {})
#   %mul_295 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_748, %convert_element_type_726), kwargs = {})
#   %view_736 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_194, [1, 128, 8192]), kwargs = {})
#   %mul_296 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_748, %view_736), kwargs = {})
#   %convert_element_type_752 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_296, torch.float32), kwargs = {})
#   %reciprocal : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_223,), kwargs = {})
#   %mul_297 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %mul_298 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_752, %mul_297), kwargs = {})
#   %sub_33 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_297), kwargs = {})
#   %mul_299 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_725, %sub_33), kwargs = {})
#   %add_228 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_299, 1), kwargs = {})
#   %mul_300 : Tensor "f32[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_298, %add_228), kwargs = {})
#   %convert_element_type_754 : Tensor "bf16[1, 128, 8192][1048576, 8192, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_300, torch.bfloat16), kwargs = {})
#   return %mul_295,%convert_element_type_754
triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 14680064}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp10 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = -tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp5 = tl.full([1], 1.0, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = (tmp2 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp0 * tmp8
    tmp11 = tmp0 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp5 / tmp6)
    tmp14 = tmp13 * tmp5
    tmp15 = tmp12 * tmp14
    tmp16 = tmp5 - tmp14
    tmp17 = tmp2 * tmp16
    tmp18 = tmp17 + tmp5
    tmp19 = tmp15 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp9, None)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/s2/cs2tknvslyv4dxxziq4r7fx4uzw4ckappd3fl7xrvax4zjenk6h5.py
# Topologically Sorted Source Nodes: [view_750, view_752, add_229, mul_301, hidden_states_276, convert_element_type_759, mul_303, mul_304, sum_36, pow_60, mul_305, mul_306, expand_174, div_59, pow_61, mul_307, mul_308, add_230, convert_element_type_760, add_231], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
# Source node to ATen node mapping:
#   add_229 => add_229
#   add_230 => add_230
#   add_231 => add_231
#   convert_element_type_759 => convert_element_type_759
#   convert_element_type_760 => convert_element_type_760
#   div_59 => div_59
#   expand_174 => expand_174
#   hidden_states_276 => convert_element_type_721
#   mul_301 => mul_301
#   mul_303 => mul_303
#   mul_304 => mul_304
#   mul_305 => mul_305
#   mul_306 => mul_306
#   mul_307 => mul_307
#   mul_308 => mul_308
#   pow_60 => pow_60
#   pow_61 => pow_61
#   sum_36 => sum_36
#   view_750 => view_750
#   view_752 => view_752
# Graph fragment:
#   %mm_202 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_202]
#   %mm_204 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_204]
#   %primals_252 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_252]
#   %add_221 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_221]
#   %convert_element_type_743 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=convert_element_type_743]
#   %rsqrt_55 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_55]
#   %sum_36 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=sum_36]
#   %view_750 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_202, [1, 128, 3072]), kwargs = {})
#   %view_752 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_204, [1, 128, 3072]), kwargs = {})
#   %add_229 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_750, %view_752), kwargs = {})
#   %mul_301 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_229, %primals_252), kwargs = {})
#   %convert_element_type_721 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_221, torch.float32), kwargs = {})
#   %convert_element_type_759 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_301, torch.float32), kwargs = {})
#   %mul_303 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_759, %convert_element_type_721), kwargs = {})
#   %mul_304 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_759, %rsqrt_55), kwargs = {})
#   %sum_36 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_303, [2], True), kwargs = {})
#   %pow_60 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt_55, 3), kwargs = {})
#   %mul_305 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_36, -0.5), kwargs = {})
#   %mul_306 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_305, %pow_60), kwargs = {})
#   %expand_174 : Tensor "f32[1, 128, 3072][128, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_306, [1, 128, 3072]), kwargs = {})
#   %div_59 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_174, 3072), kwargs = {})
#   %pow_61 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_721, 1.0), kwargs = {})
#   %mul_307 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_61, 2.0), kwargs = {})
#   %mul_308 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_59, %mul_307), kwargs = {})
#   %add_230 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_304, %mul_308), kwargs = {})
#   %convert_element_type_760 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_230, torch.bfloat16), kwargs = {})
#   %add_231 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_743, %convert_element_type_760), kwargs = {})
#   return %sum_36,%add_231
triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5 = async_compile.triton('triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 512, 'r0_': 4724736}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp19 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tmp21 = tl.full([1, 1], -0.5, tl.float32)
        tmp22 = tmp10 * tmp21
        tmp23 = tmp19 * tmp19
        tmp24 = tmp23 * tmp19
        tmp25 = tmp22 * tmp24
        tmp26 = tl.full([1, 1], 0.0003255208333333333, tl.float32)
        tmp27 = tmp25 * tmp26
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.full([1, 1], 2.0, tl.float32)
        tmp31 = tmp29 * tmp30
        tmp32 = tmp27 * tmp31
        tmp33 = tmp20 + tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp12 + tmp34
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/vl/cvlhlmuyy3wuy3nwhjijrp5472gs5lrelnquh7xf5t63sn4z2uk4.py
# Topologically Sorted Source Nodes: [view_750, view_752, add_229, hidden_states_276, hidden_states_277, to_144, mul_302, sum_35], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add_229 => add_229
#   hidden_states_276 => convert_element_type_721
#   hidden_states_277 => mul_280
#   mul_302 => mul_302
#   sum_35 => sum_35
#   to_144 => convert_element_type_722
#   view_750 => view_750
#   view_752 => view_752
# Graph fragment:
#   %mm_202 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_202]
#   %mm_204 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_204]
#   %add_221 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_221]
#   %rsqrt_55 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_55]
#   %view_750 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_202, [1, 128, 3072]), kwargs = {})
#   %view_752 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_204, [1, 128, 3072]), kwargs = {})
#   %add_229 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_750, %view_752), kwargs = {})
#   %convert_element_type_721 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_221, torch.float32), kwargs = {})
#   %mul_280 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_721, %rsqrt_55), kwargs = {})
#   %convert_element_type_722 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_280, torch.bfloat16), kwargs = {})
#   %mul_302 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_229, %convert_element_type_722), kwargs = {})
#   %sum_35 : Tensor "bf16[1, 1, 3072][3072, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_302, [0, 1], True), kwargs = {})
#   return %sum_35
triton_red_fused__to_copy_add_mul_sum_view_6 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_view_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 2371584, 'r0_': 512}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3072
    r0_numel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp2 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/7p/c7p5jrubskhpinyu6uptuwscwfojr3vnf7rcydvrfra4hoyuzbi3.py
# Topologically Sorted Source Nodes: [view_758, view_763, sum_38, squeeze_1, permute_364, clone_114], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   clone_114 => clone_114
#   permute_364 => permute_364
#   squeeze_1 => squeeze_1
#   sum_38 => sum_38
#   view_758 => view_758
#   view_763 => view_763
# Graph fragment:
#   %bmm_56 : Tensor "bf16[24, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_56]
#   %view_758 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_56, [1, 24, 128, 128]), kwargs = {})
#   %view_763 : Tensor "bf16[1, 8, 3, 128, 128][393216, 49152, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_758, [1, 8, 3, 128, 128]), kwargs = {})
#   %sum_38 : Tensor "bf16[1, 8, 1, 128, 128][131072, 16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_763, [2], True), kwargs = {})
#   %squeeze_1 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_38, 2), kwargs = {})
#   %permute_364 : Tensor "bf16[1, 128, 8, 128][131072, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%squeeze_1, [0, 2, 1, 3]), kwargs = {})
#   %clone_114 : Tensor "bf16[1, 128, 8, 128][131072, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_364,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_114
triton_poi_fused_clone_squeeze_sum_transpose_view_7 = async_compile.triton('triton_poi_fused_clone_squeeze_sum_transpose_view_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_squeeze_sum_transpose_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 1310720}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_squeeze_sum_transpose_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex // 16384
    x3 = (xindex % 16384)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 128)
    tmp0 = tl.load(in_ptr0 + (x3 + 49152*x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (16384 + x3 + 49152*x2), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (32768 + x3 + 49152*x2), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x0 + 128*x2 + 1024*x1), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/o6/co6rrs6wek6joncrvirzs4ciizaj5kx2jp7amcg47t7dsjvynate.py
# Topologically Sorted Source Nodes: [view_759, convert_element_type_769, softmax_27, mul_309, sum_37, neg_86, fma, convert_element_type_770, mul_310], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
# Source node to ATen node mapping:
#   convert_element_type_769 => convert_element_type_769
#   convert_element_type_770 => convert_element_type_770
#   fma => fma
#   mul_309 => mul_309
#   mul_310 => mul_310
#   neg_86 => neg_86
#   softmax_27 => convert_element_type_715, div_54, exp_54, sub_29
#   sum_37 => sum_37
#   view_759 => view_759
# Graph fragment:
#   %bmm_57 : Tensor "bf16[24, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_57]
#   %add_220 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0" = PlaceHolder[target=add_220]
#   %amax_27 : Tensor "f32[1, 24, 128, 1][3072, 128, 1, 1]cuda:0" = PlaceHolder[target=amax_27]
#   %sum_28 : Tensor "f32[1, 24, 128, 1][3072, 128, 1, 1]cuda:0" = PlaceHolder[target=sum_28]
#   %sum_37 : Tensor "f32[1, 24, 128, 1][3072, 128, 1, 3072]cuda:0" = PlaceHolder[target=sum_37]
#   %view_759 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_57, [1, 24, 128, 128]), kwargs = {})
#   %convert_element_type_769 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_759, torch.float32), kwargs = {})
#   %convert_element_type_715 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_220, torch.float32), kwargs = {})
#   %sub_29 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_715, %amax_27), kwargs = {})
#   %exp_54 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_29,), kwargs = {})
#   %div_54 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_54, %sum_28), kwargs = {})
#   %mul_309 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_769, %div_54), kwargs = {})
#   %sum_37 : Tensor "f32[1, 24, 128, 1][3072, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_309, [-1], True), kwargs = {})
#   %neg_86 : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div_54,), kwargs = {})
#   %fma : Tensor "f32[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.fma.default](args = (%neg_86, %sum_37, %mul_309), kwargs = {})
#   %convert_element_type_770 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%fma, torch.bfloat16), kwargs = {})
#   %mul_310 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_770, 0.08838834764831845), kwargs = {})
#   return %sum_37,%mul_310
triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8 = async_compile.triton('triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 24576, 'r0_': 3145728}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp6 = libdevice.exp(tmp5)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tmp1 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None].to(tl.float32)
    tmp14 = -tmp8
    tmp15 = tl.fma(tmp14, tmp13, tmp9)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.full([1, 1], 0.08838834764831845, tl.float32)
    tmp18 = tmp16 * tmp17
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/3g/c3gijcfr5epyyv64567z57vx3bol6kcup7fhxqlxai5lboqbjgeq.py
# Topologically Sorted Source Nodes: [view_761, permute_363, view_764, sum_39, squeeze_2, cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, mul_311, slice_117, slice_118, neg_87, full_default_8, add_232, cos, cos_1, cos_2, cos_3, mul_312, add_233], Original ATen: [aten.view, aten.transpose, aten.sum, aten.squeeze, aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.cat, aten.sin, aten.mul, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.cos]
# Source node to ATen node mapping:
#   add_232 => add_232
#   add_233 => add_233
#   cache_position => iota
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_10
#   expand => expand_1
#   freqs => permute
#   full_default_8 => full_default_8
#   getitem_1 => unsqueeze_1, unsqueeze_2
#   getitem_2 => unsqueeze_3
#   matmul => mul
#   mul_311 => mul_311
#   mul_312 => mul_312
#   neg_87 => neg_87
#   permute_363 => permute_363
#   position_ids => unsqueeze
#   position_ids_expanded => convert_element_type
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_6
#   slice_117 => slice_117
#   slice_118 => slice_118
#   squeeze_2 => squeeze_2
#   sum_39 => sum_39
#   view_761 => view_761
#   view_764 => view_764
# Graph fragment:
#   %bmm_58 : Tensor "bf16[24, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_58]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %view_761 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_58, [1, 24, 128, 128]), kwargs = {})
#   %permute_363 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_761, [0, 1, 3, 2]), kwargs = {})
#   %view_764 : Tensor "bf16[1, 8, 3, 128, 128][393216, 49152, 16384, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_363, [1, 8, 3, 128, 128]), kwargs = {})
#   %sum_39 : Tensor "bf16[1, 8, 1, 128, 128][131072, 16384, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_764, [2], True), kwargs = {})
#   %squeeze_2 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_39, 2), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
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
#   %sin : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_10,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_311 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, %unsqueeze_6), kwargs = {})
#   %slice_117 : Tensor "bf16[1, 8, 128, 64][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_311, 3, 0, 64), kwargs = {})
#   %slice_118 : Tensor "bf16[1, 8, 128, 64][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_311, 3, 64, 128), kwargs = {})
#   %neg_87 : Tensor "bf16[1, 8, 128, 64][65536, 8192, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_117,), kwargs = {})
#   %full_default_8 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.full.default](args = ([1, 8, 128, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_8, %neg_87, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_8, %slice_118, 3, 0, 64), kwargs = {})
#   %add_232 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
#   %cos : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_10,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %mul_312 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, %unsqueeze_5), kwargs = {})
#   %add_233 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_232, %mul_312), kwargs = {})
#   return %add_233
triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9 = async_compile.triton('triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 2359552}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 128) % 128)
    x2 = xindex // 16384
    x3 = (xindex % 16384)
    x4 = xindex
    x0 = (xindex % 128)
    tmp41 = tl.load(in_ptr0 + (x3 + 49152*x2), None).to(tl.float32)
    tmp42 = tl.load(in_ptr0 + (16384 + x3 + 49152*x2), None).to(tl.float32)
    tmp44 = tl.load(in_ptr0 + (32768 + x3 + 49152*x2), None).to(tl.float32)
    tmp46 = tl.load(in_ptr1 + (((((x4 // 128) % 128)) % 64)), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-8192) + x3 + 49152*x2), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (8192 + x3 + 49152*x2), tmp2, other=0.0).to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.load(in_ptr0 + (24576 + x3 + 49152*x2), tmp2, other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr1 + (((((x4 // 128) % 128)) % 64)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp12 = tl_math.sin(tmp11)
    tmp13 = tl.full([1], 1.0, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp7 * tmp15
    tmp17 = -tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp2, tmp17, tmp18)
    tmp20 = tl.full([1], 0.0, tl.float32)
    tmp21 = tl.where(tmp2, tmp19, tmp20)
    tmp22 = tmp0 < tmp1
    tmp23 = tl.load(in_ptr0 + (8192 + x3 + 49152*x2), tmp22, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr0 + (24576 + x3 + 49152*x2), tmp22, other=0.0).to(tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = tl.load(in_ptr0 + (40960 + x3 + 49152*x2), tmp22, other=0.0).to(tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.load(in_ptr1 + (((((x4 // 128) % 128)) % 64)), tmp22, eviction_policy='evict_last', other=0.0)
    tmp29 = x0
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp32 = tl_math.sin(tmp31)
    tmp33 = tl.full([1], 1.0, tl.float32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp27 * tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp22, tmp36, tmp37)
    tmp39 = tl.where(tmp22, tmp38, tmp20)
    tmp40 = tmp21 + tmp39
    tmp43 = tmp41 + tmp42
    tmp45 = tmp43 + tmp44
    tmp47 = x0
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp46 * tmp48
    tmp50 = tl_math.cos(tmp49)
    tmp51 = tl.full([1], 1.0, tl.float32)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tmp45 * tmp53
    tmp55 = tmp40 + tmp54
    tl.store(out_ptr0 + (x4), tmp55, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/dm/cdmw7iqyv4lsnjidp7lcpqtzuwnhk3jvphmpentho3qnxmbqspf7.py
# Topologically Sorted Source Nodes: [view_762, cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_313, slice_119, slice_120, neg_88, full_default_10, add_234, mul_314, add_235, permute_374, clone_116], Original ATen: [aten.view, aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone]
# Source node to ATen node mapping:
#   add_234 => add_234
#   add_235 => add_235
#   cache_position => iota
#   clone_116 => clone_116
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_1
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_10
#   expand => expand_1
#   freqs => permute
#   full_default_10 => full_default_10
#   getitem_1 => unsqueeze_1, unsqueeze_2
#   getitem_2 => unsqueeze_3
#   matmul => mul
#   mul_313 => mul_313
#   mul_314 => mul_314
#   neg_88 => neg_88
#   permute_374 => permute_374
#   position_ids => unsqueeze
#   position_ids_expanded => convert_element_type
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_2
#   sin_3 => unsqueeze_6
#   slice_119 => slice_119
#   slice_120 => slice_120
#   view_762 => view_762
# Graph fragment:
#   %bmm_59 : Tensor "bf16[24, 128, 128][16384, 128, 1]cuda:0" = PlaceHolder[target=bmm_59]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %view_762 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_59, [1, 24, 128, 128]), kwargs = {})
#   %iota : Tensor "i64[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
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
#   %sin : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_10,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %cos : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_10,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, 128, 128][16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 128, 128][16384, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_1, 1), kwargs = {})
#   %mul_313 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_762, %unsqueeze_6), kwargs = {})
#   %slice_119 : Tensor "bf16[1, 24, 128, 64][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_313, 3, 0, 64), kwargs = {})
#   %slice_120 : Tensor "bf16[1, 24, 128, 64][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_313, 3, 64, 128), kwargs = {})
#   %neg_88 : Tensor "bf16[1, 24, 128, 64][196608, 8192, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_119,), kwargs = {})
#   %full_default_10 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=56] = call_function[target=torch.ops.aten.full.default](args = ([1, 24, 128, 128], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_2 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %neg_88, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_3 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_10, %slice_120, 3, 0, 64), kwargs = {})
#   %add_234 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %slice_scatter_default_3), kwargs = {})
#   %mul_314 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_762, %unsqueeze_5), kwargs = {})
#   %add_235 : Tensor "bf16[1, 24, 128, 128][393216, 16384, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_234, %mul_314), kwargs = {})
#   %permute_374 : Tensor "bf16[1, 128, 24, 128][393216, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_235, [0, 2, 1, 3]), kwargs = {})
#   %clone_116 : Tensor "bf16[1, 128, 24, 128][393216, 3072, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_374,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_116
triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10 = async_compile.triton('triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 3932416}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x3 = xindex
    x1 = ((xindex // 128) % 128)
    x2 = xindex // 16384
    tmp33 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp34 = tl.load(in_ptr1 + ((x3 % 64)), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-64) + x3), tmp2, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + ((x3 % 64)), tmp2, eviction_policy='evict_last', other=0.0)
    tmp5 = x1
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp8 = tl_math.sin(tmp7)
    tmp9 = tl.full([1], 1.0, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp3 * tmp11
    tmp13 = -tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp2, tmp13, tmp14)
    tmp16 = tl.full([1], 0.0, tl.float32)
    tmp17 = tl.where(tmp2, tmp15, tmp16)
    tmp18 = tmp0 < tmp1
    tmp19 = tl.load(in_ptr0 + (64 + x3), tmp18, other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr1 + ((x3 % 64)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp21 = x1
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp24 = tl_math.sin(tmp23)
    tmp25 = tl.full([1], 1.0, tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp19 * tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp18, tmp28, tmp29)
    tmp31 = tl.where(tmp18, tmp30, tmp16)
    tmp32 = tmp17 + tmp31
    tmp35 = x1
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp34 * tmp36
    tmp38 = tl_math.cos(tmp37)
    tmp39 = tl.full([1], 1.0, tl.float32)
    tmp40 = tmp38 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp33 * tmp41
    tmp43 = tmp32 + tmp42
    tl.store(out_ptr0 + (x0 + 128*x2 + 3072*x1), tmp43, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/4c/c4ckk5t7ppcisovsxjwqie5way5bbgoaf2bmfkqsqmohtvbkjxid.py
# Topologically Sorted Source Nodes: [permute_369, clone_115, view_768, view_769], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
# Source node to ATen node mapping:
#   clone_115 => clone_115
#   permute_369 => permute_369
#   view_768 => view_768
#   view_769 => view_769
# Graph fragment:
#   %add_233 : Tensor "bf16[1, 8, 128, 128][131072, 16384, 1, 128]cuda:0" = PlaceHolder[target=add_233]
#   %permute_369 : Tensor "bf16[1, 128, 8, 128][131072, 128, 16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_233, [0, 2, 1, 3]), kwargs = {})
#   %clone_115 : Tensor "bf16[1, 128, 8, 128][131072, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_369,), kwargs = {memory_format: torch.contiguous_format})
#   %view_768 : Tensor "bf16[1, 128, 1024][131072, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_115, [1, 128, 1024]), kwargs = {})
#   %view_769 : Tensor "bf16[128, 1024][1024, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_768, [128, 1024]), kwargs = {})
#   return %view_769
triton_poi_fused__unsafe_view_clone_transpose_view_11 = async_compile.triton('triton_poi_fused__unsafe_view_clone_transpose_view_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__unsafe_view_clone_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'y': 262144, 'x': 524288}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_transpose_view_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x1), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (x1 + 1024*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/yn/cynrx7osqpqzjbuzkodkgz5amuf4b57jbq6v3w4cfyfs4xzjryiu.py
# Topologically Sorted Source Nodes: [view_767, view_770, add_236, view_773, add_237, mul_315, hidden_states_270, convert_element_type_787, mul_317, mul_318, sum_41, pow_62, mul_319, mul_320, expand_175, div_60, pow_63, mul_321, mul_322, add_238, convert_element_type_788, add_239], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
# Source node to ATen node mapping:
#   add_236 => add_236
#   add_237 => add_237
#   add_238 => add_238
#   add_239 => add_239
#   convert_element_type_787 => convert_element_type_787
#   convert_element_type_788 => convert_element_type_788
#   div_60 => div_60
#   expand_175 => expand_175
#   hidden_states_270 => convert_element_type_705
#   mul_315 => mul_315
#   mul_317 => mul_317
#   mul_318 => mul_318
#   mul_319 => mul_319
#   mul_320 => mul_320
#   mul_321 => mul_321
#   mul_322 => mul_322
#   pow_62 => pow_62
#   pow_63 => pow_63
#   sum_41 => sum_41
#   view_767 => view_767
#   view_770 => view_770
#   view_773 => view_773
# Graph fragment:
#   %mm_208 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_208]
#   %mm_210 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_210]
#   %mm_212 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_212]
#   %primals_247 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_247]
#   %add_216 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_216]
#   %add_231 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_231]
#   %rsqrt_54 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_54]
#   %sum_41 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=sum_41]
#   %view_767 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_208, [1, 128, 3072]), kwargs = {})
#   %view_770 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_210, [1, 128, 3072]), kwargs = {})
#   %add_236 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_767, %view_770), kwargs = {})
#   %view_773 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_212, [1, 128, 3072]), kwargs = {})
#   %add_237 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_236, %view_773), kwargs = {})
#   %mul_315 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_237, %primals_247), kwargs = {})
#   %convert_element_type_705 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_216, torch.float32), kwargs = {})
#   %convert_element_type_787 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_315, torch.float32), kwargs = {})
#   %mul_317 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_787, %convert_element_type_705), kwargs = {})
#   %mul_318 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_787, %rsqrt_54), kwargs = {})
#   %sum_41 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_317, [2], True), kwargs = {})
#   %pow_62 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt_54, 3), kwargs = {})
#   %mul_319 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_41, -0.5), kwargs = {})
#   %mul_320 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_319, %pow_62), kwargs = {})
#   %expand_175 : Tensor "f32[1, 128, 3072][128, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_320, [1, 128, 3072]), kwargs = {})
#   %div_60 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_175, 3072), kwargs = {})
#   %pow_63 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_705, 1.0), kwargs = {})
#   %mul_321 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_63, 2.0), kwargs = {})
#   %mul_322 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_60, %mul_321), kwargs = {})
#   %add_238 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %mul_322), kwargs = {})
#   %convert_element_type_788 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_238, torch.bfloat16), kwargs = {})
#   %add_239 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_231, %convert_element_type_788), kwargs = {})
#   return %sum_41,%add_239
triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12 = async_compile.triton('triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 512, 'r0_': 5511168}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp23 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr4 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tmp25 = tl.full([1, 1], -0.5, tl.float32)
        tmp26 = tmp12 * tmp25
        tmp27 = tmp23 * tmp23
        tmp28 = tmp27 * tmp23
        tmp29 = tmp26 * tmp28
        tmp30 = tl.full([1, 1], 0.0003255208333333333, tl.float32)
        tmp31 = tmp29 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl.full([1, 1], 2.0, tl.float32)
        tmp35 = tmp33 * tmp34
        tmp36 = tmp31 * tmp35
        tmp37 = tmp24 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp14 + tmp38
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/uv/cuv7yqm43yqnvtkkgeqiclqk5vwyg6jscfn26rmmlr2evn6kh43s.py
# Topologically Sorted Source Nodes: [view_767, view_770, add_236, view_773, add_237, hidden_states_270, hidden_states_271, to_141, mul_316, sum_40], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add_236 => add_236
#   add_237 => add_237
#   hidden_states_270 => convert_element_type_705
#   hidden_states_271 => mul_273
#   mul_316 => mul_316
#   sum_40 => sum_40
#   to_141 => convert_element_type_706
#   view_767 => view_767
#   view_770 => view_770
#   view_773 => view_773
# Graph fragment:
#   %mm_208 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_208]
#   %mm_210 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_210]
#   %mm_212 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_212]
#   %add_216 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_216]
#   %rsqrt_54 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_54]
#   %view_767 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_208, [1, 128, 3072]), kwargs = {})
#   %view_770 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_210, [1, 128, 3072]), kwargs = {})
#   %add_236 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_767, %view_770), kwargs = {})
#   %view_773 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_212, [1, 128, 3072]), kwargs = {})
#   %add_237 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_236, %view_773), kwargs = {})
#   %convert_element_type_705 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_216, torch.float32), kwargs = {})
#   %mul_273 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_705, %rsqrt_54), kwargs = {})
#   %convert_element_type_706 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_273, torch.bfloat16), kwargs = {})
#   %mul_316 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_237, %convert_element_type_706), kwargs = {})
#   %sum_40 : Tensor "bf16[1, 1, 3072][3072, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_316, [0, 1], True), kwargs = {})
#   return %sum_40
triton_red_fused__to_copy_add_mul_sum_view_13 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_view_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 3158016, 'r0_': 512}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_view_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3072
    r0_numel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/cu/ccuivxzapvi7g7o37dau42fzqxj7nc6t74nsyn2osbqnf4yfutaz.py
# Topologically Sorted Source Nodes: [view_1506, view_1508, add_580, mul_1057, attn_output_3, hidden_states_5, hidden_states_6, convert_element_type_1974, mul_1059, mul_1060, sum_225, pow_168, mul_1061, mul_1062, expand_228, div_113, pow_169, mul_1063, mul_1064, add_581, convert_element_type_1975, add_582], Original ATen: [aten.view, aten.add, aten.mul, aten._unsafe_view, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
# Source node to ATen node mapping:
#   add_580 => add_580
#   add_581 => add_581
#   add_582 => add_582
#   attn_output_3 => view_30
#   convert_element_type_1974 => convert_element_type_1974
#   convert_element_type_1975 => convert_element_type_1975
#   div_113 => div_113
#   expand_228 => expand_228
#   hidden_states_5 => add_5
#   hidden_states_6 => convert_element_type_19
#   mul_1057 => mul_1057
#   mul_1059 => mul_1059
#   mul_1060 => mul_1060
#   mul_1061 => mul_1061
#   mul_1062 => mul_1062
#   mul_1063 => mul_1063
#   mul_1064 => mul_1064
#   pow_168 => pow_168
#   pow_169 => pow_169
#   sum_225 => sum_225
#   view_1506 => view_1506
#   view_1508 => view_1508
# Graph fragment:
#   %mm_580 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_580]
#   %mm_582 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_582]
#   %primals_9 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_9]
#   %embedding : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=embedding]
#   %mm_3 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %add_577 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_577]
#   %rsqrt_1 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %sum_225 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=sum_225]
#   %view_1506 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_580, [1, 128, 3072]), kwargs = {})
#   %view_1508 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_582, [1, 128, 3072]), kwargs = {})
#   %add_580 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1506, %view_1508), kwargs = {})
#   %mul_1057 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_580, %primals_9), kwargs = {})
#   %view_30 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 128, 3072]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_30), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %convert_element_type_1974 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1057, torch.float32), kwargs = {})
#   %mul_1059 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1974, %convert_element_type_19), kwargs = {})
#   %mul_1060 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1974, %rsqrt_1), kwargs = {})
#   %sum_225 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1059, [2], True), kwargs = {})
#   %pow_168 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt_1, 3), kwargs = {})
#   %mul_1061 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_225, -0.5), kwargs = {})
#   %mul_1062 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1061, %pow_168), kwargs = {})
#   %expand_228 : Tensor "f32[1, 128, 3072][128, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_1062, [1, 128, 3072]), kwargs = {})
#   %div_113 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_228, 3072), kwargs = {})
#   %pow_169 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_19, 1.0), kwargs = {})
#   %mul_1063 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_169, 2.0), kwargs = {})
#   %mul_1064 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_113, %mul_1063), kwargs = {})
#   %add_581 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1060, %mul_1064), kwargs = {})
#   %convert_element_type_1975 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_581, torch.bfloat16), kwargs = {})
#   %add_582 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_577, %convert_element_type_1975), kwargs = {})
#   return %sum_225,%add_582
triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_14 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 12, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 512, 'r0_': 5511168}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp5 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr3 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr4 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = tl.full([1, 1], -0.5, tl.float32)
        tmp24 = tmp12 * tmp23
        tmp25 = tmp21 * tmp21
        tmp26 = tmp25 * tmp21
        tmp27 = tmp24 * tmp26
        tmp28 = tl.full([1, 1], 0.0003255208333333333, tl.float32)
        tmp29 = tmp27 * tmp28
        tmp32 = tmp30 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tl.full([1, 1], 2.0, tl.float32)
        tmp35 = tmp33 * tmp34
        tmp36 = tmp29 * tmp35
        tmp37 = tmp22 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp14 + tmp38
        tl.store(in_out_ptr0 + (r0_1 + 3072*x0), tmp39, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/wg/cwgtbsdah6qrfg3wlyqhyfalo4hm7er5c4etc5rxr6t3jjw2fsz4.py
# Topologically Sorted Source Nodes: [view_1506, view_1508, add_580, attn_output_3, hidden_states_5, hidden_states_6, hidden_states_7, to_9, mul_1058, sum_224, view_1523, view_1526, add_587, view_1529, add_588, hidden_states, hidden_states_1, to_6, mul_1072, sum_229], Original ATen: [aten.view, aten.add, aten._unsafe_view, aten._to_copy, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add_580 => add_580
#   add_587 => add_587
#   add_588 => add_588
#   attn_output_3 => view_30
#   hidden_states => convert_element_type_3
#   hidden_states_1 => mul_3
#   hidden_states_5 => add_5
#   hidden_states_6 => convert_element_type_19
#   hidden_states_7 => mul_10
#   mul_1058 => mul_1058
#   mul_1072 => mul_1072
#   sum_224 => sum_224
#   sum_229 => sum_229
#   to_6 => convert_element_type_4
#   to_9 => convert_element_type_20
#   view_1506 => view_1506
#   view_1508 => view_1508
#   view_1523 => view_1523
#   view_1526 => view_1526
#   view_1529 => view_1529
# Graph fragment:
#   %mm_580 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_580]
#   %mm_582 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_582]
#   %embedding : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=embedding]
#   %mm_3 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %rsqrt_1 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %mm_586 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_586]
#   %mm_588 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_588]
#   %mm_590 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_590]
#   %rsqrt : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %view_1506 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_580, [1, 128, 3072]), kwargs = {})
#   %view_1508 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_582, [1, 128, 3072]), kwargs = {})
#   %add_580 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1506, %view_1508), kwargs = {})
#   %view_30 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 128, 3072]), kwargs = {})
#   %add_5 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_30), kwargs = {})
#   %convert_element_type_19 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.float32), kwargs = {})
#   %mul_10 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_19, %rsqrt_1), kwargs = {})
#   %convert_element_type_20 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
#   %mul_1058 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_580, %convert_element_type_20), kwargs = {})
#   %sum_224 : Tensor "bf16[1, 1, 3072][3072, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1058, [0, 1], True), kwargs = {})
#   %view_1523 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_586, [1, 128, 3072]), kwargs = {})
#   %view_1526 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_588, [1, 128, 3072]), kwargs = {})
#   %add_587 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1523, %view_1526), kwargs = {})
#   %view_1529 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_590, [1, 128, 3072]), kwargs = {})
#   %add_588 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_587, %view_1529), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %mul_3 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_3, %rsqrt), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %mul_1072 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_588, %convert_element_type_4), kwargs = {})
#   %sum_229 : Tensor "bf16[1, 1, 3072][3072, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1072, [0, 1], True), kwargs = {})
#   return %sum_224,%sum_229
triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_15 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 5529600, 'r0_': 1024}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 3072
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr7 + (x0 + 3072*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr8 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp2 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp3.to(tl.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp18 * tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, R0_BLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(r0_mask & xmask, tmp26, _tmp25)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/xc/cxcivyghf43k6mijx4sw5pcda36om7tqskok7gng67vcdnab62vo.py
# Topologically Sorted Source Nodes: [loss, view_1523, view_1526, add_587, view_1529, add_588, mul_1071, hidden_states, convert_element_type_2002, mul_1073, mul_1074, sum_230, pow_170, mul_1075, mul_1076, expand_229, div_114, pow_171, mul_1077, mul_1078, add_589, convert_element_type_2003, add_590, convert_element_type_2004, eq_1, unsqueeze_119, where_5], Original ATen: [aten.nll_loss_forward, aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   add_587 => add_587
#   add_588 => add_588
#   add_589 => add_589
#   add_590 => add_590
#   convert_element_type_2002 => convert_element_type_2002
#   convert_element_type_2003 => convert_element_type_2003
#   convert_element_type_2004 => convert_element_type_2004
#   div_114 => div_114
#   eq_1 => eq_1
#   expand_229 => expand_229
#   hidden_states => convert_element_type_3
#   loss => full_default_4
#   mul_1071 => mul_1071
#   mul_1073 => mul_1073
#   mul_1074 => mul_1074
#   mul_1075 => mul_1075
#   mul_1076 => mul_1076
#   mul_1077 => mul_1077
#   mul_1078 => mul_1078
#   pow_170 => pow_170
#   pow_171 => pow_171
#   sum_230 => sum_230
#   unsqueeze_119 => unsqueeze_119
#   view_1523 => view_1523
#   view_1526 => view_1526
#   view_1529 => view_1529
#   where_5 => where_5
# Graph fragment:
#   %mm_586 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_586]
#   %mm_588 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_588]
#   %mm_590 : Tensor "bf16[128, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_590]
#   %primals_4 : Tensor "bf16[3072][1]cuda:0" = PlaceHolder[target=primals_4]
#   %embedding : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=embedding]
#   %primals_1 : Tensor "i64[1, 128][128, 1]cuda:0" = PlaceHolder[target=primals_1]
#   %add_582 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0" = PlaceHolder[target=add_582]
#   %rsqrt : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0" = PlaceHolder[target=rsqrt]
#   %sum_230 : Tensor "f32[1, 128, 1][128, 1, 128]cuda:0" = PlaceHolder[target=sum_230]
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_1523 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_586, [1, 128, 3072]), kwargs = {})
#   %view_1526 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_588, [1, 128, 3072]), kwargs = {})
#   %add_587 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1523, %view_1526), kwargs = {})
#   %view_1529 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_590, [1, 128, 3072]), kwargs = {})
#   %add_588 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_587, %view_1529), kwargs = {})
#   %mul_1071 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_588, %primals_4), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %convert_element_type_2002 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1071, torch.float32), kwargs = {})
#   %mul_1073 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2002, %convert_element_type_3), kwargs = {})
#   %mul_1074 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2002, %rsqrt), kwargs = {})
#   %sum_230 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1073, [2], True), kwargs = {})
#   %pow_170 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt, 3), kwargs = {})
#   %mul_1075 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_230, -0.5), kwargs = {})
#   %mul_1076 : Tensor "f32[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1075, %pow_170), kwargs = {})
#   %expand_229 : Tensor "f32[1, 128, 3072][128, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_1076, [1, 128, 3072]), kwargs = {})
#   %div_114 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_229, 3072), kwargs = {})
#   %pow_171 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_3, 1.0), kwargs = {})
#   %mul_1077 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_171, 2.0), kwargs = {})
#   %mul_1078 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_114, %mul_1077), kwargs = {})
#   %add_589 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1074, %mul_1078), kwargs = {})
#   %convert_element_type_2003 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_589, torch.bfloat16), kwargs = {})
#   %add_590 : Tensor "bf16[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_582, %convert_element_type_2003), kwargs = {})
#   %convert_element_type_2004 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_590, torch.float32), kwargs = {})
#   %eq_1 : Tensor "b8[1, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%primals_1, -1), kwargs = {})
#   %unsqueeze_119 : Tensor "b8[1, 128, 1][128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%eq_1, -1), kwargs = {})
#   %where_5 : Tensor "f32[1, 128, 3072][393216, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_119, %full_default_4, %convert_element_type_2004), kwargs = {})
#   return %sum_230,%where_5
triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_16 = async_compile.triton('triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*i64', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 13, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 1536, 'r0_': 7084032}, 'add_persistent_rblock': True, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp17 = tl.load(in_ptr6 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr0 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr4 + (r0_1 + 3072*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.full([1, 1], -1, tl.int64)
        tmp16 = tmp14 == tmp15
        tmp20 = tmp18 + tmp19
        tmp22 = tmp20 + tmp21
        tmp24 = tmp22 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp25 * tmp26
        tmp28 = tl.full([1, 1], -0.5, tl.float32)
        tmp29 = tmp12 * tmp28
        tmp30 = tmp26 * tmp26
        tmp31 = tmp30 * tmp26
        tmp32 = tmp29 * tmp31
        tmp33 = tl.full([1, 1], 0.0003255208333333333, tl.float32)
        tmp34 = tmp32 * tmp33
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tl.full([1, 1], 2.0, tl.float32)
        tmp38 = tmp36 * tmp37
        tmp39 = tmp34 * tmp38
        tmp40 = tmp27 + tmp39
        tmp41 = tmp40.to(tl.float32)
        tmp42 = tmp17 + tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tl.full([1, 1], 0.0, tl.float32)
        tmp45 = tl.where(tmp16, tmp44, tmp43)
        tl.store(out_ptr1 + (r0_1 + 3072*x0), tmp45, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/qn/cqnp7ee3vpweqajqwoba3ac4kp3nuynysn7oheosaiskrt66bf2v.py
# Topologically Sorted Source Nodes: [convert_element_type_2005], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   convert_element_type_2005 => convert_element_type_2005
# Graph fragment:
#   %buf903 : Tensor  = PlaceHolder[target=buf903]
#   %convert_element_type_2005 : Tensor "bf16[128256, 3072][3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.bfloat16), kwargs = {})
#   return %convert_element_type_2005
triton_poi_fused_embedding_dense_backward_17 = async_compile.triton('triton_poi_fused_embedding_dense_backward_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 536870912}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_embedding_dense_backward_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 1576009728}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 394002432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, embedding, rsqrt, view_11, add_4, amax, sum_1, view_29, mm_3, rsqrt_1, view_31, mm_4, mm_5, view_35, add_8, rsqrt_2, view_37, add_12, amax_1, sum_2, view_55, add_13, rsqrt_3, view_57, mm_11, mm_12, view_61, add_16, rsqrt_4, view_63, add_20, amax_2, sum_3, view_81, add_21, rsqrt_5, view_83, mm_18, mm_19, view_87, add_24, rsqrt_6, view_89, add_28, amax_3, sum_4, view_107, add_29, rsqrt_7, view_109, mm_25, mm_26, view_113, add_32, rsqrt_8, view_115, add_36, amax_4, sum_5, view_133, add_37, rsqrt_9, view_135, mm_32, mm_33, view_139, add_40, rsqrt_10, view_141, add_44, amax_5, sum_6, view_159, add_45, rsqrt_11, view_161, mm_39, mm_40, view_165, add_48, rsqrt_12, view_167, add_52, amax_6, sum_7, view_185, add_53, rsqrt_13, view_187, mm_46, mm_47, view_191, add_56, rsqrt_14, view_193, add_60, amax_7, sum_8, view_211, add_61, rsqrt_15, view_213, mm_53, mm_54, view_217, add_64, rsqrt_16, view_219, add_68, amax_8, sum_9, view_237, add_69, rsqrt_17, view_239, mm_60, mm_61, view_243, add_72, rsqrt_18, view_245, add_76, amax_9, sum_10, view_263, add_77, rsqrt_19, view_265, mm_67, mm_68, view_269, add_80, rsqrt_20, view_271, add_84, amax_10, sum_11, view_289, add_85, rsqrt_21, view_291, mm_74, mm_75, view_295, add_88, rsqrt_22, view_297, add_92, amax_11, sum_12, view_315, add_93, rsqrt_23, view_317, mm_81, mm_82, view_321, add_96, rsqrt_24, view_323, add_100, amax_12, sum_13, view_341, add_101, rsqrt_25, view_343, mm_88, mm_89, view_347, add_104, rsqrt_26, view_349, add_108, amax_13, sum_14, view_367, add_109, rsqrt_27, view_369, mm_95, mm_96, view_373, add_112, rsqrt_28, view_375, add_116, amax_14, sum_15, view_393, add_117, rsqrt_29, view_395, mm_102, mm_103, view_399, add_120, rsqrt_30, view_401, add_124, amax_15, sum_16, view_419, add_125, rsqrt_31, view_421, mm_109, mm_110, view_425, add_128, rsqrt_32, view_427, add_132, amax_16, sum_17, view_445, add_133, rsqrt_33, view_447, mm_116, mm_117, view_451, add_136, rsqrt_34, view_453, add_140, amax_17, sum_18, view_471, add_141, rsqrt_35, view_473, mm_123, mm_124, view_477, add_144, rsqrt_36, view_479, add_148, amax_18, sum_19, view_497, add_149, rsqrt_37, view_499, mm_130, mm_131, view_503, add_152, rsqrt_38, view_505, add_156, amax_19, sum_20, view_523, add_157, rsqrt_39, view_525, mm_137, mm_138, view_529, add_160, rsqrt_40, view_531, add_164, amax_20, sum_21, view_549, add_165, rsqrt_41, view_551, mm_144, mm_145, view_555, add_168, rsqrt_42, view_557, add_172, amax_21, sum_22, view_575, add_173, rsqrt_43, view_577, mm_151, mm_152, view_581, add_176, rsqrt_44, view_583, add_180, amax_22, sum_23, view_601, add_181, rsqrt_45, view_603, mm_158, mm_159, view_607, add_184, rsqrt_46, view_609, add_188, amax_23, sum_24, view_627, add_189, rsqrt_47, view_629, mm_165, mm_166, view_633, add_192, rsqrt_48, view_635, add_196, amax_24, sum_25, view_653, add_197, rsqrt_49, view_655, mm_172, mm_173, view_659, add_200, rsqrt_50, view_661, add_204, amax_25, sum_26, view_679, add_205, rsqrt_51, view_681, mm_179, mm_180, view_685, add_208, rsqrt_52, view_687, add_212, amax_26, sum_27, view_705, add_213, rsqrt_53, view_707, mm_186, mm_187, view_711, add_216, rsqrt_54, view_713, add_220, amax_27, sum_28, view_731, add_221, rsqrt_55, view_733, mm_193, mm_194, view_737, add_224, rsqrt_56, view_739, mm_196, constant_pad_nd, amax_28, log, convert_element_type_736, permute_359, permute_360, permute_361, permute_362, permute_396, permute_397, permute_398, permute_399, permute_433, permute_434, permute_435, permute_436, permute_470, permute_471, permute_472, permute_473, permute_507, permute_508, permute_509, permute_510, permute_544, permute_545, permute_546, permute_547, permute_581, permute_582, permute_583, permute_584, permute_618, permute_619, permute_620, permute_621, permute_655, permute_656, permute_657, permute_658, permute_692, permute_693, permute_694, permute_695, permute_729, permute_730, permute_731, permute_732, permute_766, permute_767, permute_768, permute_769, permute_803, permute_804, permute_805, permute_806, permute_840, permute_841, permute_842, permute_843, permute_877, permute_878, permute_879, permute_880, permute_914, permute_915, permute_916, permute_917, permute_951, permute_952, permute_953, permute_954, permute_988, permute_989, permute_990, permute_991, permute_1025, permute_1026, permute_1027, permute_1028, permute_1062, permute_1063, permute_1064, permute_1065, permute_1099, permute_1100, permute_1101, permute_1102, permute_1136, permute_1137, permute_1138, permute_1139, permute_1173, permute_1174, permute_1175, permute_1176, permute_1210, permute_1211, permute_1212, permute_1213, permute_1247, permute_1248, permute_1249, permute_1250, permute_1284, permute_1285, permute_1286, permute_1287, permute_1321, permute_1322, permute_1323, permute_1324, permute_1358, permute_1359, permute_1360, permute_1361, tangents_1 = args
        args.clear()
        assert_size_stride(constant_pad_nd, (1, 129), (129, 1), 'input')
        assert_size_stride(tangents_1, (), (), 'input')
        assert_size_stride(convert_element_type_736, (), (), 'input')
        assert_size_stride(mm_196, (128, 128256), (128256, 1), 'input')
        assert_size_stride(amax_28, (128, 1), (1, 1), 'input')
        assert_size_stride(log, (128, 1), (1, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf1 = reinterpret_tensor(mm_196, (1, 128, 128256), (16416768, 128256, 1), 0); del mm_196  # reuse
            # Topologically Sorted Source Nodes: [div_57, getitem_200, shift_labels_1, unsqueeze_118, ne_4, loss, where_3, where_self, where_4, mul_285, logits, logits_1, logits_2, exp_57, sum_32, mul_286, sub_32, view_743, convert_element_type_737], Original ATen: [aten.nll_loss_backward, aten.slice, aten.view, aten.nll_loss_forward, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._unsafe_view, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0.run(buf1, constant_pad_nd, tangents_1, convert_element_type_736, amax_28, log, 128, 128256, stream=raw_stream0)
            del amax_28
            del constant_pad_nd
            del convert_element_type_736
            del log
            del tangents_1
            buf902 = empty_strided_cuda((128256, 3072), (3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [full_default_121], Original ATen: [aten.embedding_dense_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_1.run(buf902, 394002432, stream=raw_stream0)
            assert_size_stride(primals_2, (128256, 3072), (3072, 1), 'input')
            buf2 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [div_57, getitem_200, shift_labels_1, unsqueeze_118, ne_4, loss, where_3, where_self, where_4, mul_285, logits, logits_1, logits_2, exp_57, mul_286, sub_32, view_743, convert_element_type_737, view_744, permute_340, mm_198], Original ATen: [aten.nll_loss_backward, aten.slice, aten.view, aten.nll_loss_forward, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._unsafe_view, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1, (128, 128256), (128256, 1), 0), primals_2, out=buf2)
            del primals_2
            assert_size_stride(add_224, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_56, (1, 128, 1), (128, 1, 1), 'input')
            buf3 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_745, hidden_states_280, hidden_states_281, to_146, mul_288, sum_33], Original ATen: [aten.view, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_mul_sum_view_2.run(buf2, add_224, rsqrt_56, buf3, 3072, 128, stream=raw_stream0)
            assert_size_stride(primals_256, (3072, ), (1, ), 'input')
            buf5 = reinterpret_tensor(buf2, (1, 128, 3072), (393216, 3072, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [view_745, mul_287, hidden_states_280, convert_element_type_742, mul_289, mul_290, sum_34, pow_58, mul_291, mul_292, expand_173, div_58, pow_59, mul_293, mul_294, add_226, convert_element_type_743], Original ATen: [aten.view, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_3.run(buf5, primals_256, add_224, rsqrt_56, 128, 3072, stream=raw_stream0)
            del add_224
            del primals_256
            del rsqrt_56
            assert_size_stride(view_737, (128, 8192), (8192, 1), 'input')
            buf6 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_747, permute_342, mm_199], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (3072, 128), (1, 3072), 0), view_737, out=buf6)
            del view_737
            assert_size_stride(primals_255, (3072, 8192), (8192, 1), 'input')
            buf7 = empty_strided_cuda((128, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_747, down_proj_27, permute_344, mm_200], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (128, 3072), (3072, 1), 0), primals_255, out=buf7)
            del primals_255
            assert_size_stride(mm_193, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_194, (128, 8192), (8192, 1), 'input')
            buf8 = empty_strided_cuda((1, 128, 8192), (1048576, 8192, 1), torch.bfloat16)
            buf11 = reinterpret_tensor(mm_194, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_194  # reuse
            # Topologically Sorted Source Nodes: [view_748, linear_193, silu_27, mul_295, linear_194, mul_296, convert_element_type_752, reciprocal, mul_297, mul_298, sub_33, mul_299, add_228, mul_300, convert_element_type_754], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf11, buf7, mm_193, buf8, 1048576, stream=raw_stream0)
            del buf7
            del mm_193
            assert_size_stride(view_733, (128, 3072), (3072, 1), 'input')
            buf9 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_748, linear_193, silu_27, mul_295, view_749, permute_346, mm_201], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf8, (8192, 128), (1, 8192), 0), view_733, out=buf9)
            assert_size_stride(primals_254, (8192, 3072), (3072, 1), 'input')
            buf10 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_748, linear_193, silu_27, mul_295, view_749, linear_194, permute_348, mm_202], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf8, (128, 8192), (8192, 1), 0), primals_254, out=buf10)
            del primals_254
            buf12 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_748, linear_193, silu_27, linear_194, mul_296, convert_element_type_752, reciprocal, mul_297, mul_298, sub_33, mul_299, add_228, mul_300, convert_element_type_754, view_751, permute_350, mm_203], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf11, (8192, 128), (1, 8192), 0), view_733, out=buf12)
            del view_733
            assert_size_stride(primals_253, (8192, 3072), (3072, 1), 'input')
            buf13 = empty_strided_cuda((128, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_748, linear_193, silu_27, linear_194, mul_296, convert_element_type_752, reciprocal, mul_297, mul_298, sub_33, mul_299, add_228, mul_300, convert_element_type_754, view_751, permute_352, mm_204], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf11, (128, 8192), (8192, 1), 0), primals_253, out=buf13)
            del primals_253
            assert_size_stride(primals_252, (3072, ), (1, ), 'input')
            assert_size_stride(add_221, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_55, (1, 128, 1), (128, 1, 1), 'input')
            buf16 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [view_750, view_752, add_229, mul_301, hidden_states_276, convert_element_type_759, mul_303, mul_304, sum_36, pow_60, mul_305, mul_306, expand_174, div_59, pow_61, mul_307, mul_308, add_230, convert_element_type_760, add_231], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf16, buf10, buf13, primals_252, add_221, rsqrt_55, 128, 3072, stream=raw_stream0)
            del primals_252
            buf14 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_750, view_752, add_229, hidden_states_276, hidden_states_277, to_144, mul_302, sum_35], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf10, buf13, add_221, rsqrt_55, buf14, 3072, 128, stream=raw_stream0)
            del add_221
            del rsqrt_55
            assert_size_stride(view_731, (128, 3072), (3072, 1), 'input')
            buf17 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_754, permute_354, mm_205], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf16, (3072, 128), (1, 3072), 0), view_731, out=buf17)
            del view_731
            assert_size_stride(primals_251, (3072, 3072), (3072, 1), 'input')
            buf18 = buf13; del buf13  # reuse
            # Topologically Sorted Source Nodes: [view_754, attn_output_111, permute_356, mm_206], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf16, (128, 3072), (3072, 1), 0), primals_251, out=buf18)
            del primals_251
            assert_size_stride(permute_359, (24, 128, 128), (16384, 1, 128), 'input')
            buf19 = reinterpret_tensor(buf10, (24, 128, 128), (16384, 128, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [view_755, view_756, permute_358, view_757, bmm_56], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_359, reinterpret_tensor(buf18, (24, 128, 128), (128, 3072, 1), 0), out=buf19)
            del permute_359
            assert_size_stride(permute_360, (24, 128, 128), (16384, 1, 128), 'input')
            buf20 = empty_strided_cuda((24, 128, 128), (16384, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_755, view_756, permute_358, view_757, bmm_57], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf18, (24, 128, 128), (128, 3072, 1), 0), permute_360, out=buf20)
            del permute_360
            buf26 = empty_strided_cuda((1, 128, 8, 128), (131072, 1024, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_758, view_763, sum_38, squeeze_1, permute_364, clone_114], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf19, buf26, 131072, stream=raw_stream0)
            assert_size_stride(add_220, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_27, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_28, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf22 = add_220; del add_220  # reuse
            # Topologically Sorted Source Nodes: [view_759, convert_element_type_769, softmax_27, mul_309, sum_37, neg_86, fma, convert_element_type_770, mul_310], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf22, buf20, amax_27, sum_28, 3072, 128, stream=raw_stream0)
            del amax_27
            del sum_28
            assert_size_stride(view_713, (128, 3072), (3072, 1), 'input')
            buf27 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_758, view_763, sum_38, squeeze_1, permute_364, clone_114, view_765, view_766, permute_365, mm_207], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf26, (1024, 128), (1, 1024), 0), view_713, out=buf27)
            assert_size_stride(primals_250, (1024, 3072), (3072, 1), 'input')
            buf28 = reinterpret_tensor(buf20, (128, 3072), (3072, 1), 0); del buf20  # reuse
            # Topologically Sorted Source Nodes: [view_758, view_763, sum_38, squeeze_1, permute_364, clone_114, view_765, view_766, linear_191, permute_367, mm_208], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf26, (128, 1024), (1024, 1), 0), primals_250, out=buf28)
            del primals_250
            assert_size_stride(permute_361, (24, 128, 128), (128, 1, 3072), 'input')
            buf23 = buf19; del buf19  # reuse
            # Topologically Sorted Source Nodes: [view_759, convert_element_type_769, softmax_27, mul_309, neg_86, fma, convert_element_type_770, mul_310, view_760, bmm_58], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_361, reinterpret_tensor(buf22, (24, 128, 128), (16384, 128, 1), 0), out=buf23)
            del permute_361
            assert_size_stride(permute_362, (24, 128, 128), (16384, 128, 1), 'input')
            buf24 = reinterpret_tensor(buf18, (24, 128, 128), (16384, 128, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [view_759, convert_element_type_769, softmax_27, mul_309, neg_86, fma, convert_element_type_770, mul_310, view_760, bmm_59], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf22, (24, 128, 128), (16384, 128, 1), 0), permute_362, out=buf24)
            del buf22
            del permute_362
            assert_size_stride(primals_3, (64, ), (1, ), 'input')
            buf25 = reinterpret_tensor(buf26, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [view_761, permute_363, view_764, sum_39, squeeze_2, cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, mul_311, slice_117, slice_118, neg_87, full_default_8, add_232, cos, cos_1, cos_2, cos_3, mul_312, add_233], Original ATen: [aten.view, aten.transpose, aten.sum, aten.squeeze, aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.cat, aten.sin, aten.mul, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.cos]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf23, primals_3, buf25, 131072, stream=raw_stream0)
            buf32 = reinterpret_tensor(buf23, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [view_762, cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_313, slice_119, slice_120, neg_88, full_default_10, add_234, mul_314, add_235, permute_374, clone_116], Original ATen: [aten.view, aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf24, primals_3, buf32, 393216, stream=raw_stream0)
            buf29 = empty_strided_cuda((128, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_369, clone_115, view_768, view_769], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf25, buf29, 128, 1024, stream=raw_stream0)
            buf33 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_762, cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_313, slice_119, slice_120, neg_88, full_default_10, add_234, mul_314, add_235, permute_374, clone_116, view_771, view_772, permute_375, mm_211], Original ATen: [aten.view, aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (3072, 128), (1, 3072), 0), view_713, out=buf33)
            assert_size_stride(primals_248, (3072, 3072), (3072, 1), 'input')
            buf34 = reinterpret_tensor(buf24, (128, 3072), (3072, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [view_762, cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_313, slice_119, slice_120, neg_88, full_default_10, add_234, mul_314, add_235, permute_374, clone_116, view_771, view_772, linear_189, permute_377, mm_212], Original ATen: [aten.view, aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf32, (128, 3072), (3072, 1), 0), primals_248, out=buf34)
            del primals_248
            buf30 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_370, mm_209], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf29, (1024, 128), (1, 1024), 0), view_713, out=buf30)
            del view_713
            assert_size_stride(primals_249, (1024, 3072), (3072, 1), 'input')
            buf31 = reinterpret_tensor(buf32, (128, 3072), (3072, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [linear_190, permute_372, mm_210], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf29, primals_249, out=buf31)
            del primals_249
            assert_size_stride(primals_247, (3072, ), (1, ), 'input')
            assert_size_stride(add_216, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_54, (1, 128, 1), (128, 1, 1), 'input')
            buf37 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [view_767, view_770, add_236, view_773, add_237, mul_315, hidden_states_270, convert_element_type_787, mul_317, mul_318, sum_41, pow_62, mul_319, mul_320, expand_175, div_60, pow_63, mul_321, mul_322, add_238, convert_element_type_788, add_239], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf37, buf28, buf31, buf34, primals_247, add_216, rsqrt_54, 128, 3072, stream=raw_stream0)
            del primals_247
            buf35 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_767, view_770, add_236, view_773, add_237, hidden_states_270, hidden_states_271, to_141, mul_316, sum_40], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf28, buf31, buf34, add_216, rsqrt_54, buf35, 3072, 128, stream=raw_stream0)
            del add_216
            del rsqrt_54
            assert_size_stride(view_711, (128, 8192), (8192, 1), 'input')
            buf38 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_775, permute_379, mm_213], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf37, (3072, 128), (1, 3072), 0), view_711, out=buf38)
            del view_711
            assert_size_stride(primals_246, (3072, 8192), (8192, 1), 'input')
            buf39 = reinterpret_tensor(buf11, (128, 8192), (8192, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [view_775, down_proj_26, permute_381, mm_214], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf37, (128, 3072), (3072, 1), 0), primals_246, out=buf39)
            del primals_246
            assert_size_stride(mm_186, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_187, (128, 8192), (8192, 1), 'input')
            buf40 = buf8; del buf8  # reuse
            buf43 = reinterpret_tensor(mm_187, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_187  # reuse
            # Topologically Sorted Source Nodes: [view_776, linear_186, silu_26, mul_323, linear_187, mul_324, convert_element_type_797, reciprocal_1, mul_325, mul_326, sub_34, mul_327, add_241, mul_328, convert_element_type_799], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf43, buf39, mm_186, buf40, 1048576, stream=raw_stream0)
            del buf39
            del mm_186
            assert_size_stride(view_707, (128, 3072), (3072, 1), 'input')
            buf41 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_776, linear_186, silu_26, mul_323, view_777, permute_383, mm_215], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (8192, 128), (1, 8192), 0), view_707, out=buf41)
            assert_size_stride(primals_245, (8192, 3072), (3072, 1), 'input')
            buf42 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [view_776, linear_186, silu_26, mul_323, view_777, linear_187, permute_385, mm_216], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (128, 8192), (8192, 1), 0), primals_245, out=buf42)
            del primals_245
            buf44 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_776, linear_186, silu_26, linear_187, mul_324, convert_element_type_797, reciprocal_1, mul_325, mul_326, sub_34, mul_327, add_241, mul_328, convert_element_type_799, view_779, permute_387, mm_217], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf43, (8192, 128), (1, 8192), 0), view_707, out=buf44)
            del view_707
            assert_size_stride(primals_244, (8192, 3072), (3072, 1), 'input')
            buf45 = buf31; del buf31  # reuse
            # Topologically Sorted Source Nodes: [view_776, linear_186, silu_26, linear_187, mul_324, convert_element_type_797, reciprocal_1, mul_325, mul_326, sub_34, mul_327, add_241, mul_328, convert_element_type_799, view_779, permute_389, mm_218], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf43, (128, 8192), (8192, 1), 0), primals_244, out=buf45)
            del primals_244
            assert_size_stride(primals_243, (3072, ), (1, ), 'input')
            assert_size_stride(add_213, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_53, (1, 128, 1), (128, 1, 1), 'input')
            buf48 = buf37; del buf37  # reuse
            # Topologically Sorted Source Nodes: [view_778, view_780, add_242, mul_329, hidden_states_266, convert_element_type_804, mul_331, mul_332, sum_43, pow_64, mul_333, mul_334, expand_176, div_61, pow_65, mul_335, mul_336, add_243, convert_element_type_805, add_244], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf48, buf42, buf45, primals_243, add_213, rsqrt_53, 128, 3072, stream=raw_stream0)
            del primals_243
            buf46 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_778, view_780, add_242, hidden_states_266, hidden_states_267, to_139, mul_330, sum_42], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf42, buf45, add_213, rsqrt_53, buf46, 3072, 128, stream=raw_stream0)
            del add_213
            del rsqrt_53
            assert_size_stride(view_705, (128, 3072), (3072, 1), 'input')
            buf49 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_782, permute_391, mm_219], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (3072, 128), (1, 3072), 0), view_705, out=buf49)
            del view_705
            assert_size_stride(primals_242, (3072, 3072), (3072, 1), 'input')
            buf50 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [view_782, attn_output_107, permute_393, mm_220], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (128, 3072), (3072, 1), 0), primals_242, out=buf50)
            del primals_242
            assert_size_stride(permute_396, (24, 128, 128), (16384, 1, 128), 'input')
            buf51 = reinterpret_tensor(buf42, (24, 128, 128), (16384, 128, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [view_783, view_784, permute_395, view_785, bmm_60], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_396, reinterpret_tensor(buf50, (24, 128, 128), (128, 3072, 1), 0), out=buf51)
            del permute_396
            assert_size_stride(permute_397, (24, 128, 128), (16384, 1, 128), 'input')
            buf52 = reinterpret_tensor(buf28, (24, 128, 128), (16384, 128, 1), 0); del buf28  # reuse
            # Topologically Sorted Source Nodes: [view_783, view_784, permute_395, view_785, bmm_61], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf50, (24, 128, 128), (128, 3072, 1), 0), permute_397, out=buf52)
            del permute_397
            buf58 = reinterpret_tensor(buf29, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf29  # reuse
            # Topologically Sorted Source Nodes: [view_786, view_791, sum_45, squeeze_3, permute_401, clone_117], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf51, buf58, 131072, stream=raw_stream0)
            assert_size_stride(add_212, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_26, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_27, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf54 = add_212; del add_212  # reuse
            # Topologically Sorted Source Nodes: [view_787, convert_element_type_814, softmax_26, mul_337, sum_44, neg_90, fma_1, convert_element_type_815, mul_338], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf54, buf52, amax_26, sum_27, 3072, 128, stream=raw_stream0)
            del amax_26
            del sum_27
            assert_size_stride(view_687, (128, 3072), (3072, 1), 'input')
            buf59 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_786, view_791, sum_45, squeeze_3, permute_401, clone_117, view_793, view_794, permute_402, mm_221], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf58, (1024, 128), (1, 1024), 0), view_687, out=buf59)
            assert_size_stride(primals_241, (1024, 3072), (3072, 1), 'input')
            buf60 = reinterpret_tensor(buf52, (128, 3072), (3072, 1), 0); del buf52  # reuse
            # Topologically Sorted Source Nodes: [view_786, view_791, sum_45, squeeze_3, permute_401, clone_117, view_793, view_794, linear_184, permute_404, mm_222], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf58, (128, 1024), (1024, 1), 0), primals_241, out=buf60)
            del primals_241
            assert_size_stride(permute_398, (24, 128, 128), (128, 1, 3072), 'input')
            buf55 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [view_787, convert_element_type_814, softmax_26, mul_337, neg_90, fma_1, convert_element_type_815, mul_338, view_788, bmm_62], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_398, reinterpret_tensor(buf54, (24, 128, 128), (16384, 128, 1), 0), out=buf55)
            del permute_398
            assert_size_stride(permute_399, (24, 128, 128), (16384, 128, 1), 'input')
            buf56 = reinterpret_tensor(buf50, (24, 128, 128), (16384, 128, 1), 0); del buf50  # reuse
            # Topologically Sorted Source Nodes: [view_787, convert_element_type_814, softmax_26, mul_337, neg_90, fma_1, convert_element_type_815, mul_338, view_788, bmm_63], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf54, (24, 128, 128), (16384, 128, 1), 0), permute_399, out=buf56)
            del buf54
            del permute_399
            buf57 = reinterpret_tensor(buf58, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf58  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_789, permute_400, view_792, sum_46, squeeze_4, mul_339, slice_121, slice_122, neg_91, add_245, mul_340, add_246], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf55, primals_3, buf57, 131072, stream=raw_stream0)
            buf64 = reinterpret_tensor(buf55, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_790, mul_341, slice_123, slice_124, neg_92, add_247, mul_342, add_248, permute_411, clone_119], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf56, primals_3, buf64, 393216, stream=raw_stream0)
            buf61 = reinterpret_tensor(buf25, (128, 1024), (1024, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [permute_406, clone_118, view_796, view_797], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf57, buf61, 128, 1024, stream=raw_stream0)
            buf65 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_790, mul_341, slice_123, slice_124, neg_92, add_247, mul_342, add_248, permute_411, clone_119, view_799, view_800, permute_412, mm_225], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf64, (3072, 128), (1, 3072), 0), view_687, out=buf65)
            assert_size_stride(primals_239, (3072, 3072), (3072, 1), 'input')
            buf66 = reinterpret_tensor(buf56, (128, 3072), (3072, 1), 0); del buf56  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_790, mul_341, slice_123, slice_124, neg_92, add_247, mul_342, add_248, permute_411, clone_119, view_799, view_800, linear_182, permute_414, mm_226], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf64, (128, 3072), (3072, 1), 0), primals_239, out=buf66)
            del primals_239
            buf62 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_407, mm_223], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf61, (1024, 128), (1, 1024), 0), view_687, out=buf62)
            del view_687
            assert_size_stride(primals_240, (1024, 3072), (3072, 1), 'input')
            buf63 = reinterpret_tensor(buf64, (128, 3072), (3072, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [linear_183, permute_409, mm_224], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf61, primals_240, out=buf63)
            del primals_240
            assert_size_stride(primals_238, (3072, ), (1, ), 'input')
            assert_size_stride(add_208, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_52, (1, 128, 1), (128, 1, 1), 'input')
            buf69 = buf48; del buf48  # reuse
            # Topologically Sorted Source Nodes: [view_795, view_798, add_249, view_801, add_250, mul_343, hidden_states_260, convert_element_type_832, mul_345, mul_346, sum_48, pow_66, mul_347, mul_348, expand_177, div_62, pow_67, mul_349, mul_350, add_251, convert_element_type_833, add_252], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf69, buf60, buf63, buf66, primals_238, add_208, rsqrt_52, 128, 3072, stream=raw_stream0)
            del primals_238
            buf67 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_795, view_798, add_249, view_801, add_250, hidden_states_260, hidden_states_261, to_136, mul_344, sum_47], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf60, buf63, buf66, add_208, rsqrt_52, buf67, 3072, 128, stream=raw_stream0)
            del add_208
            del rsqrt_52
            assert_size_stride(view_685, (128, 8192), (8192, 1), 'input')
            buf70 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_803, permute_416, mm_227], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf69, (3072, 128), (1, 3072), 0), view_685, out=buf70)
            del view_685
            assert_size_stride(primals_237, (3072, 8192), (8192, 1), 'input')
            buf71 = reinterpret_tensor(buf43, (128, 8192), (8192, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [view_803, down_proj_25, permute_418, mm_228], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf69, (128, 3072), (3072, 1), 0), primals_237, out=buf71)
            del primals_237
            assert_size_stride(mm_179, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_180, (128, 8192), (8192, 1), 'input')
            buf72 = buf40; del buf40  # reuse
            buf75 = reinterpret_tensor(mm_180, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_180  # reuse
            # Topologically Sorted Source Nodes: [view_804, linear_179, silu_25, mul_351, linear_180, mul_352, convert_element_type_842, reciprocal_2, mul_353, mul_354, sub_35, mul_355, add_254, mul_356, convert_element_type_844], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf75, buf71, mm_179, buf72, 1048576, stream=raw_stream0)
            del buf71
            del mm_179
            assert_size_stride(view_681, (128, 3072), (3072, 1), 'input')
            buf73 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_804, linear_179, silu_25, mul_351, view_805, permute_420, mm_229], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf72, (8192, 128), (1, 8192), 0), view_681, out=buf73)
            assert_size_stride(primals_236, (8192, 3072), (3072, 1), 'input')
            buf74 = buf66; del buf66  # reuse
            # Topologically Sorted Source Nodes: [view_804, linear_179, silu_25, mul_351, view_805, linear_180, permute_422, mm_230], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf72, (128, 8192), (8192, 1), 0), primals_236, out=buf74)
            del primals_236
            buf76 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_804, linear_179, silu_25, linear_180, mul_352, convert_element_type_842, reciprocal_2, mul_353, mul_354, sub_35, mul_355, add_254, mul_356, convert_element_type_844, view_807, permute_424, mm_231], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf75, (8192, 128), (1, 8192), 0), view_681, out=buf76)
            del view_681
            assert_size_stride(primals_235, (8192, 3072), (3072, 1), 'input')
            buf77 = buf63; del buf63  # reuse
            # Topologically Sorted Source Nodes: [view_804, linear_179, silu_25, linear_180, mul_352, convert_element_type_842, reciprocal_2, mul_353, mul_354, sub_35, mul_355, add_254, mul_356, convert_element_type_844, view_807, permute_426, mm_232], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf75, (128, 8192), (8192, 1), 0), primals_235, out=buf77)
            del primals_235
            assert_size_stride(primals_234, (3072, ), (1, ), 'input')
            assert_size_stride(add_205, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_51, (1, 128, 1), (128, 1, 1), 'input')
            buf80 = buf69; del buf69  # reuse
            # Topologically Sorted Source Nodes: [view_806, view_808, add_255, mul_357, hidden_states_256, convert_element_type_849, mul_359, mul_360, sum_50, pow_68, mul_361, mul_362, expand_178, div_63, pow_69, mul_363, mul_364, add_256, convert_element_type_850, add_257], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf80, buf74, buf77, primals_234, add_205, rsqrt_51, 128, 3072, stream=raw_stream0)
            del primals_234
            buf78 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_806, view_808, add_255, hidden_states_256, hidden_states_257, to_134, mul_358, sum_49], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf74, buf77, add_205, rsqrt_51, buf78, 3072, 128, stream=raw_stream0)
            del add_205
            del rsqrt_51
            assert_size_stride(view_679, (128, 3072), (3072, 1), 'input')
            buf81 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_810, permute_428, mm_233], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf80, (3072, 128), (1, 3072), 0), view_679, out=buf81)
            del view_679
            assert_size_stride(primals_233, (3072, 3072), (3072, 1), 'input')
            buf82 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [view_810, attn_output_103, permute_430, mm_234], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf80, (128, 3072), (3072, 1), 0), primals_233, out=buf82)
            del primals_233
            assert_size_stride(permute_433, (24, 128, 128), (16384, 1, 128), 'input')
            buf83 = reinterpret_tensor(buf74, (24, 128, 128), (16384, 128, 1), 0); del buf74  # reuse
            # Topologically Sorted Source Nodes: [view_811, view_812, permute_432, view_813, bmm_64], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_433, reinterpret_tensor(buf82, (24, 128, 128), (128, 3072, 1), 0), out=buf83)
            del permute_433
            assert_size_stride(permute_434, (24, 128, 128), (16384, 1, 128), 'input')
            buf84 = reinterpret_tensor(buf60, (24, 128, 128), (16384, 128, 1), 0); del buf60  # reuse
            # Topologically Sorted Source Nodes: [view_811, view_812, permute_432, view_813, bmm_65], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf82, (24, 128, 128), (128, 3072, 1), 0), permute_434, out=buf84)
            del permute_434
            buf90 = reinterpret_tensor(buf61, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [view_814, view_819, sum_52, squeeze_5, permute_438, clone_120], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf83, buf90, 131072, stream=raw_stream0)
            assert_size_stride(add_204, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_25, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_26, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf86 = add_204; del add_204  # reuse
            # Topologically Sorted Source Nodes: [view_815, convert_element_type_859, softmax_25, mul_365, sum_51, neg_94, fma_2, convert_element_type_860, mul_366], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf86, buf84, amax_25, sum_26, 3072, 128, stream=raw_stream0)
            del amax_25
            del sum_26
            assert_size_stride(view_661, (128, 3072), (3072, 1), 'input')
            buf91 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_814, view_819, sum_52, squeeze_5, permute_438, clone_120, view_821, view_822, permute_439, mm_235], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf90, (1024, 128), (1, 1024), 0), view_661, out=buf91)
            assert_size_stride(primals_232, (1024, 3072), (3072, 1), 'input')
            buf92 = reinterpret_tensor(buf84, (128, 3072), (3072, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [view_814, view_819, sum_52, squeeze_5, permute_438, clone_120, view_821, view_822, linear_177, permute_441, mm_236], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf90, (128, 1024), (1024, 1), 0), primals_232, out=buf92)
            del primals_232
            assert_size_stride(permute_435, (24, 128, 128), (128, 1, 3072), 'input')
            buf87 = buf83; del buf83  # reuse
            # Topologically Sorted Source Nodes: [view_815, convert_element_type_859, softmax_25, mul_365, neg_94, fma_2, convert_element_type_860, mul_366, view_816, bmm_66], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_435, reinterpret_tensor(buf86, (24, 128, 128), (16384, 128, 1), 0), out=buf87)
            del permute_435
            assert_size_stride(permute_436, (24, 128, 128), (16384, 128, 1), 'input')
            buf88 = reinterpret_tensor(buf82, (24, 128, 128), (16384, 128, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [view_815, convert_element_type_859, softmax_25, mul_365, neg_94, fma_2, convert_element_type_860, mul_366, view_816, bmm_67], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf86, (24, 128, 128), (16384, 128, 1), 0), permute_436, out=buf88)
            del buf86
            del permute_436
            buf89 = reinterpret_tensor(buf90, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf90  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_817, permute_437, view_820, sum_53, squeeze_6, mul_367, slice_125, slice_126, neg_95, add_258, mul_368, add_259], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf87, primals_3, buf89, 131072, stream=raw_stream0)
            buf96 = reinterpret_tensor(buf87, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_818, mul_369, slice_127, slice_128, neg_96, add_260, mul_370, add_261, permute_448, clone_122], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf88, primals_3, buf96, 393216, stream=raw_stream0)
            buf93 = reinterpret_tensor(buf57, (128, 1024), (1024, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [permute_443, clone_121, view_824, view_825], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf89, buf93, 128, 1024, stream=raw_stream0)
            buf97 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_818, mul_369, slice_127, slice_128, neg_96, add_260, mul_370, add_261, permute_448, clone_122, view_827, view_828, permute_449, mm_239], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf96, (3072, 128), (1, 3072), 0), view_661, out=buf97)
            assert_size_stride(primals_230, (3072, 3072), (3072, 1), 'input')
            buf98 = reinterpret_tensor(buf88, (128, 3072), (3072, 1), 0); del buf88  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_818, mul_369, slice_127, slice_128, neg_96, add_260, mul_370, add_261, permute_448, clone_122, view_827, view_828, linear_175, permute_451, mm_240], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf96, (128, 3072), (3072, 1), 0), primals_230, out=buf98)
            del primals_230
            buf94 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_444, mm_237], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (1024, 128), (1, 1024), 0), view_661, out=buf94)
            del view_661
            assert_size_stride(primals_231, (1024, 3072), (3072, 1), 'input')
            buf95 = reinterpret_tensor(buf96, (128, 3072), (3072, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [linear_176, permute_446, mm_238], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf93, primals_231, out=buf95)
            del primals_231
            assert_size_stride(primals_229, (3072, ), (1, ), 'input')
            assert_size_stride(add_200, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_50, (1, 128, 1), (128, 1, 1), 'input')
            buf101 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [view_823, view_826, add_262, view_829, add_263, mul_371, hidden_states_250, convert_element_type_877, mul_373, mul_374, sum_55, pow_70, mul_375, mul_376, expand_179, div_64, pow_71, mul_377, mul_378, add_264, convert_element_type_878, add_265], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf101, buf92, buf95, buf98, primals_229, add_200, rsqrt_50, 128, 3072, stream=raw_stream0)
            del primals_229
            buf99 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_823, view_826, add_262, view_829, add_263, hidden_states_250, hidden_states_251, to_131, mul_372, sum_54], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf92, buf95, buf98, add_200, rsqrt_50, buf99, 3072, 128, stream=raw_stream0)
            del add_200
            del rsqrt_50
            assert_size_stride(view_659, (128, 8192), (8192, 1), 'input')
            buf102 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_831, permute_453, mm_241], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf101, (3072, 128), (1, 3072), 0), view_659, out=buf102)
            del view_659
            assert_size_stride(primals_228, (3072, 8192), (8192, 1), 'input')
            buf103 = reinterpret_tensor(buf75, (128, 8192), (8192, 1), 0); del buf75  # reuse
            # Topologically Sorted Source Nodes: [view_831, down_proj_24, permute_455, mm_242], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf101, (128, 3072), (3072, 1), 0), primals_228, out=buf103)
            del primals_228
            assert_size_stride(mm_172, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_173, (128, 8192), (8192, 1), 'input')
            buf104 = buf72; del buf72  # reuse
            buf107 = reinterpret_tensor(mm_173, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_173  # reuse
            # Topologically Sorted Source Nodes: [view_832, linear_172, silu_24, mul_379, linear_173, mul_380, convert_element_type_887, reciprocal_3, mul_381, mul_382, sub_36, mul_383, add_267, mul_384, convert_element_type_889], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf107, buf103, mm_172, buf104, 1048576, stream=raw_stream0)
            del buf103
            del mm_172
            assert_size_stride(view_655, (128, 3072), (3072, 1), 'input')
            buf105 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_832, linear_172, silu_24, mul_379, view_833, permute_457, mm_243], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf104, (8192, 128), (1, 8192), 0), view_655, out=buf105)
            assert_size_stride(primals_227, (8192, 3072), (3072, 1), 'input')
            buf106 = buf98; del buf98  # reuse
            # Topologically Sorted Source Nodes: [view_832, linear_172, silu_24, mul_379, view_833, linear_173, permute_459, mm_244], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf104, (128, 8192), (8192, 1), 0), primals_227, out=buf106)
            del primals_227
            buf108 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_832, linear_172, silu_24, linear_173, mul_380, convert_element_type_887, reciprocal_3, mul_381, mul_382, sub_36, mul_383, add_267, mul_384, convert_element_type_889, view_835, permute_461, mm_245], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf107, (8192, 128), (1, 8192), 0), view_655, out=buf108)
            del view_655
            assert_size_stride(primals_226, (8192, 3072), (3072, 1), 'input')
            buf109 = buf95; del buf95  # reuse
            # Topologically Sorted Source Nodes: [view_832, linear_172, silu_24, linear_173, mul_380, convert_element_type_887, reciprocal_3, mul_381, mul_382, sub_36, mul_383, add_267, mul_384, convert_element_type_889, view_835, permute_463, mm_246], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf107, (128, 8192), (8192, 1), 0), primals_226, out=buf109)
            del primals_226
            assert_size_stride(primals_225, (3072, ), (1, ), 'input')
            assert_size_stride(add_197, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_49, (1, 128, 1), (128, 1, 1), 'input')
            buf112 = buf101; del buf101  # reuse
            # Topologically Sorted Source Nodes: [view_834, view_836, add_268, mul_385, hidden_states_246, convert_element_type_894, mul_387, mul_388, sum_57, pow_72, mul_389, mul_390, expand_180, div_65, pow_73, mul_391, mul_392, add_269, convert_element_type_895, add_270], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf112, buf106, buf109, primals_225, add_197, rsqrt_49, 128, 3072, stream=raw_stream0)
            del primals_225
            buf110 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_834, view_836, add_268, hidden_states_246, hidden_states_247, to_129, mul_386, sum_56], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf106, buf109, add_197, rsqrt_49, buf110, 3072, 128, stream=raw_stream0)
            del add_197
            del rsqrt_49
            assert_size_stride(view_653, (128, 3072), (3072, 1), 'input')
            buf113 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_838, permute_465, mm_247], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf112, (3072, 128), (1, 3072), 0), view_653, out=buf113)
            del view_653
            assert_size_stride(primals_224, (3072, 3072), (3072, 1), 'input')
            buf114 = buf109; del buf109  # reuse
            # Topologically Sorted Source Nodes: [view_838, attn_output_99, permute_467, mm_248], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf112, (128, 3072), (3072, 1), 0), primals_224, out=buf114)
            del primals_224
            assert_size_stride(permute_470, (24, 128, 128), (16384, 1, 128), 'input')
            buf115 = reinterpret_tensor(buf106, (24, 128, 128), (16384, 128, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [view_839, view_840, permute_469, view_841, bmm_68], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_470, reinterpret_tensor(buf114, (24, 128, 128), (128, 3072, 1), 0), out=buf115)
            del permute_470
            assert_size_stride(permute_471, (24, 128, 128), (16384, 1, 128), 'input')
            buf116 = reinterpret_tensor(buf92, (24, 128, 128), (16384, 128, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [view_839, view_840, permute_469, view_841, bmm_69], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf114, (24, 128, 128), (128, 3072, 1), 0), permute_471, out=buf116)
            del permute_471
            buf122 = reinterpret_tensor(buf93, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf93  # reuse
            # Topologically Sorted Source Nodes: [view_842, view_847, sum_59, squeeze_7, permute_475, clone_123], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf115, buf122, 131072, stream=raw_stream0)
            assert_size_stride(add_196, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_24, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_25, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf118 = add_196; del add_196  # reuse
            # Topologically Sorted Source Nodes: [view_843, convert_element_type_904, softmax_24, mul_393, sum_58, neg_98, fma_3, convert_element_type_905, mul_394], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf118, buf116, amax_24, sum_25, 3072, 128, stream=raw_stream0)
            del amax_24
            del sum_25
            assert_size_stride(view_635, (128, 3072), (3072, 1), 'input')
            buf123 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_842, view_847, sum_59, squeeze_7, permute_475, clone_123, view_849, view_850, permute_476, mm_249], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf122, (1024, 128), (1, 1024), 0), view_635, out=buf123)
            assert_size_stride(primals_223, (1024, 3072), (3072, 1), 'input')
            buf124 = reinterpret_tensor(buf116, (128, 3072), (3072, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [view_842, view_847, sum_59, squeeze_7, permute_475, clone_123, view_849, view_850, linear_170, permute_478, mm_250], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf122, (128, 1024), (1024, 1), 0), primals_223, out=buf124)
            del primals_223
            assert_size_stride(permute_472, (24, 128, 128), (128, 1, 3072), 'input')
            buf119 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [view_843, convert_element_type_904, softmax_24, mul_393, neg_98, fma_3, convert_element_type_905, mul_394, view_844, bmm_70], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_472, reinterpret_tensor(buf118, (24, 128, 128), (16384, 128, 1), 0), out=buf119)
            del permute_472
            assert_size_stride(permute_473, (24, 128, 128), (16384, 128, 1), 'input')
            buf120 = reinterpret_tensor(buf114, (24, 128, 128), (16384, 128, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [view_843, convert_element_type_904, softmax_24, mul_393, neg_98, fma_3, convert_element_type_905, mul_394, view_844, bmm_71], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf118, (24, 128, 128), (16384, 128, 1), 0), permute_473, out=buf120)
            del buf118
            del permute_473
            buf121 = reinterpret_tensor(buf122, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_845, permute_474, view_848, sum_60, squeeze_8, mul_395, slice_129, slice_130, neg_99, add_271, mul_396, add_272], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf119, primals_3, buf121, 131072, stream=raw_stream0)
            buf128 = reinterpret_tensor(buf119, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_846, mul_397, slice_131, slice_132, neg_100, add_273, mul_398, add_274, permute_485, clone_125], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf120, primals_3, buf128, 393216, stream=raw_stream0)
            buf125 = reinterpret_tensor(buf89, (128, 1024), (1024, 1), 0); del buf89  # reuse
            # Topologically Sorted Source Nodes: [permute_480, clone_124, view_852, view_853], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf121, buf125, 128, 1024, stream=raw_stream0)
            buf129 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_846, mul_397, slice_131, slice_132, neg_100, add_273, mul_398, add_274, permute_485, clone_125, view_855, view_856, permute_486, mm_253], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf128, (3072, 128), (1, 3072), 0), view_635, out=buf129)
            assert_size_stride(primals_221, (3072, 3072), (3072, 1), 'input')
            buf130 = reinterpret_tensor(buf120, (128, 3072), (3072, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_846, mul_397, slice_131, slice_132, neg_100, add_273, mul_398, add_274, permute_485, clone_125, view_855, view_856, linear_168, permute_488, mm_254], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf128, (128, 3072), (3072, 1), 0), primals_221, out=buf130)
            del primals_221
            buf126 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_481, mm_251], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf125, (1024, 128), (1, 1024), 0), view_635, out=buf126)
            del view_635
            assert_size_stride(primals_222, (1024, 3072), (3072, 1), 'input')
            buf127 = reinterpret_tensor(buf128, (128, 3072), (3072, 1), 0); del buf128  # reuse
            # Topologically Sorted Source Nodes: [linear_169, permute_483, mm_252], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf125, primals_222, out=buf127)
            del primals_222
            assert_size_stride(primals_220, (3072, ), (1, ), 'input')
            assert_size_stride(add_192, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_48, (1, 128, 1), (128, 1, 1), 'input')
            buf133 = buf112; del buf112  # reuse
            # Topologically Sorted Source Nodes: [view_851, view_854, add_275, view_857, add_276, mul_399, hidden_states_240, convert_element_type_922, mul_401, mul_402, sum_62, pow_74, mul_403, mul_404, expand_181, div_66, pow_75, mul_405, mul_406, add_277, convert_element_type_923, add_278], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf133, buf124, buf127, buf130, primals_220, add_192, rsqrt_48, 128, 3072, stream=raw_stream0)
            del primals_220
            buf131 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_851, view_854, add_275, view_857, add_276, hidden_states_240, hidden_states_241, to_126, mul_400, sum_61], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf124, buf127, buf130, add_192, rsqrt_48, buf131, 3072, 128, stream=raw_stream0)
            del add_192
            del rsqrt_48
            assert_size_stride(view_633, (128, 8192), (8192, 1), 'input')
            buf134 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_859, permute_490, mm_255], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf133, (3072, 128), (1, 3072), 0), view_633, out=buf134)
            del view_633
            assert_size_stride(primals_219, (3072, 8192), (8192, 1), 'input')
            buf135 = reinterpret_tensor(buf107, (128, 8192), (8192, 1), 0); del buf107  # reuse
            # Topologically Sorted Source Nodes: [view_859, down_proj_23, permute_492, mm_256], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf133, (128, 3072), (3072, 1), 0), primals_219, out=buf135)
            del primals_219
            assert_size_stride(mm_165, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_166, (128, 8192), (8192, 1), 'input')
            buf136 = buf104; del buf104  # reuse
            buf139 = reinterpret_tensor(mm_166, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_166  # reuse
            # Topologically Sorted Source Nodes: [view_860, linear_165, silu_23, mul_407, linear_166, mul_408, convert_element_type_932, reciprocal_4, mul_409, mul_410, sub_37, mul_411, add_280, mul_412, convert_element_type_934], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf139, buf135, mm_165, buf136, 1048576, stream=raw_stream0)
            del buf135
            del mm_165
            assert_size_stride(view_629, (128, 3072), (3072, 1), 'input')
            buf137 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_860, linear_165, silu_23, mul_407, view_861, permute_494, mm_257], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf136, (8192, 128), (1, 8192), 0), view_629, out=buf137)
            assert_size_stride(primals_218, (8192, 3072), (3072, 1), 'input')
            buf138 = buf130; del buf130  # reuse
            # Topologically Sorted Source Nodes: [view_860, linear_165, silu_23, mul_407, view_861, linear_166, permute_496, mm_258], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf136, (128, 8192), (8192, 1), 0), primals_218, out=buf138)
            del primals_218
            buf140 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_860, linear_165, silu_23, linear_166, mul_408, convert_element_type_932, reciprocal_4, mul_409, mul_410, sub_37, mul_411, add_280, mul_412, convert_element_type_934, view_863, permute_498, mm_259], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (8192, 128), (1, 8192), 0), view_629, out=buf140)
            del view_629
            assert_size_stride(primals_217, (8192, 3072), (3072, 1), 'input')
            buf141 = buf127; del buf127  # reuse
            # Topologically Sorted Source Nodes: [view_860, linear_165, silu_23, linear_166, mul_408, convert_element_type_932, reciprocal_4, mul_409, mul_410, sub_37, mul_411, add_280, mul_412, convert_element_type_934, view_863, permute_500, mm_260], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (128, 8192), (8192, 1), 0), primals_217, out=buf141)
            del primals_217
            assert_size_stride(primals_216, (3072, ), (1, ), 'input')
            assert_size_stride(add_189, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_47, (1, 128, 1), (128, 1, 1), 'input')
            buf144 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [view_862, view_864, add_281, mul_413, hidden_states_236, convert_element_type_939, mul_415, mul_416, sum_64, pow_76, mul_417, mul_418, expand_182, div_67, pow_77, mul_419, mul_420, add_282, convert_element_type_940, add_283], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf144, buf138, buf141, primals_216, add_189, rsqrt_47, 128, 3072, stream=raw_stream0)
            del primals_216
            buf142 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_862, view_864, add_281, hidden_states_236, hidden_states_237, to_124, mul_414, sum_63], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf138, buf141, add_189, rsqrt_47, buf142, 3072, 128, stream=raw_stream0)
            del add_189
            del rsqrt_47
            assert_size_stride(view_627, (128, 3072), (3072, 1), 'input')
            buf145 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_866, permute_502, mm_261], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf144, (3072, 128), (1, 3072), 0), view_627, out=buf145)
            del view_627
            assert_size_stride(primals_215, (3072, 3072), (3072, 1), 'input')
            buf146 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [view_866, attn_output_95, permute_504, mm_262], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf144, (128, 3072), (3072, 1), 0), primals_215, out=buf146)
            del primals_215
            assert_size_stride(permute_507, (24, 128, 128), (16384, 1, 128), 'input')
            buf147 = reinterpret_tensor(buf138, (24, 128, 128), (16384, 128, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [view_867, view_868, permute_506, view_869, bmm_72], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_507, reinterpret_tensor(buf146, (24, 128, 128), (128, 3072, 1), 0), out=buf147)
            del permute_507
            assert_size_stride(permute_508, (24, 128, 128), (16384, 1, 128), 'input')
            buf148 = reinterpret_tensor(buf124, (24, 128, 128), (16384, 128, 1), 0); del buf124  # reuse
            # Topologically Sorted Source Nodes: [view_867, view_868, permute_506, view_869, bmm_73], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf146, (24, 128, 128), (128, 3072, 1), 0), permute_508, out=buf148)
            del permute_508
            buf154 = reinterpret_tensor(buf125, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [view_870, view_875, sum_66, squeeze_9, permute_512, clone_126], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf147, buf154, 131072, stream=raw_stream0)
            assert_size_stride(add_188, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_23, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_24, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf150 = add_188; del add_188  # reuse
            # Topologically Sorted Source Nodes: [view_871, convert_element_type_949, softmax_23, mul_421, sum_65, neg_102, fma_4, convert_element_type_950, mul_422], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf150, buf148, amax_23, sum_24, 3072, 128, stream=raw_stream0)
            del amax_23
            del sum_24
            assert_size_stride(view_609, (128, 3072), (3072, 1), 'input')
            buf155 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_870, view_875, sum_66, squeeze_9, permute_512, clone_126, view_877, view_878, permute_513, mm_263], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (1024, 128), (1, 1024), 0), view_609, out=buf155)
            assert_size_stride(primals_214, (1024, 3072), (3072, 1), 'input')
            buf156 = reinterpret_tensor(buf148, (128, 3072), (3072, 1), 0); del buf148  # reuse
            # Topologically Sorted Source Nodes: [view_870, view_875, sum_66, squeeze_9, permute_512, clone_126, view_877, view_878, linear_163, permute_515, mm_264], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf154, (128, 1024), (1024, 1), 0), primals_214, out=buf156)
            del primals_214
            assert_size_stride(permute_509, (24, 128, 128), (128, 1, 3072), 'input')
            buf151 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [view_871, convert_element_type_949, softmax_23, mul_421, neg_102, fma_4, convert_element_type_950, mul_422, view_872, bmm_74], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_509, reinterpret_tensor(buf150, (24, 128, 128), (16384, 128, 1), 0), out=buf151)
            del permute_509
            assert_size_stride(permute_510, (24, 128, 128), (16384, 128, 1), 'input')
            buf152 = reinterpret_tensor(buf146, (24, 128, 128), (16384, 128, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [view_871, convert_element_type_949, softmax_23, mul_421, neg_102, fma_4, convert_element_type_950, mul_422, view_872, bmm_75], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf150, (24, 128, 128), (16384, 128, 1), 0), permute_510, out=buf152)
            del buf150
            del permute_510
            buf153 = reinterpret_tensor(buf154, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf154  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_873, permute_511, view_876, sum_67, squeeze_10, mul_423, slice_133, slice_134, neg_103, add_284, mul_424, add_285], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf151, primals_3, buf153, 131072, stream=raw_stream0)
            buf160 = reinterpret_tensor(buf151, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_874, mul_425, slice_135, slice_136, neg_104, add_286, mul_426, add_287, permute_522, clone_128], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf152, primals_3, buf160, 393216, stream=raw_stream0)
            buf157 = reinterpret_tensor(buf121, (128, 1024), (1024, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [permute_517, clone_127, view_880, view_881], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf153, buf157, 128, 1024, stream=raw_stream0)
            buf161 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_874, mul_425, slice_135, slice_136, neg_104, add_286, mul_426, add_287, permute_522, clone_128, view_883, view_884, permute_523, mm_267], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf160, (3072, 128), (1, 3072), 0), view_609, out=buf161)
            assert_size_stride(primals_212, (3072, 3072), (3072, 1), 'input')
            buf162 = reinterpret_tensor(buf152, (128, 3072), (3072, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_874, mul_425, slice_135, slice_136, neg_104, add_286, mul_426, add_287, permute_522, clone_128, view_883, view_884, linear_161, permute_525, mm_268], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf160, (128, 3072), (3072, 1), 0), primals_212, out=buf162)
            del primals_212
            buf158 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_518, mm_265], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf157, (1024, 128), (1, 1024), 0), view_609, out=buf158)
            del view_609
            assert_size_stride(primals_213, (1024, 3072), (3072, 1), 'input')
            buf159 = reinterpret_tensor(buf160, (128, 3072), (3072, 1), 0); del buf160  # reuse
            # Topologically Sorted Source Nodes: [linear_162, permute_520, mm_266], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf157, primals_213, out=buf159)
            del primals_213
            assert_size_stride(primals_211, (3072, ), (1, ), 'input')
            assert_size_stride(add_184, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_46, (1, 128, 1), (128, 1, 1), 'input')
            buf165 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [view_879, view_882, add_288, view_885, add_289, mul_427, hidden_states_230, convert_element_type_967, mul_429, mul_430, sum_69, pow_78, mul_431, mul_432, expand_183, div_68, pow_79, mul_433, mul_434, add_290, convert_element_type_968, add_291], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf165, buf156, buf159, buf162, primals_211, add_184, rsqrt_46, 128, 3072, stream=raw_stream0)
            del primals_211
            buf163 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_879, view_882, add_288, view_885, add_289, hidden_states_230, hidden_states_231, to_121, mul_428, sum_68], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf156, buf159, buf162, add_184, rsqrt_46, buf163, 3072, 128, stream=raw_stream0)
            del add_184
            del rsqrt_46
            assert_size_stride(view_607, (128, 8192), (8192, 1), 'input')
            buf166 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_887, permute_527, mm_269], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf165, (3072, 128), (1, 3072), 0), view_607, out=buf166)
            del view_607
            assert_size_stride(primals_210, (3072, 8192), (8192, 1), 'input')
            buf167 = reinterpret_tensor(buf139, (128, 8192), (8192, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [view_887, down_proj_22, permute_529, mm_270], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf165, (128, 3072), (3072, 1), 0), primals_210, out=buf167)
            del primals_210
            assert_size_stride(mm_158, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_159, (128, 8192), (8192, 1), 'input')
            buf168 = buf136; del buf136  # reuse
            buf171 = reinterpret_tensor(mm_159, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_159  # reuse
            # Topologically Sorted Source Nodes: [view_888, linear_158, silu_22, mul_435, linear_159, mul_436, convert_element_type_977, reciprocal_5, mul_437, mul_438, sub_38, mul_439, add_293, mul_440, convert_element_type_979], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf171, buf167, mm_158, buf168, 1048576, stream=raw_stream0)
            del buf167
            del mm_158
            assert_size_stride(view_603, (128, 3072), (3072, 1), 'input')
            buf169 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_888, linear_158, silu_22, mul_435, view_889, permute_531, mm_271], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf168, (8192, 128), (1, 8192), 0), view_603, out=buf169)
            assert_size_stride(primals_209, (8192, 3072), (3072, 1), 'input')
            buf170 = buf162; del buf162  # reuse
            # Topologically Sorted Source Nodes: [view_888, linear_158, silu_22, mul_435, view_889, linear_159, permute_533, mm_272], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf168, (128, 8192), (8192, 1), 0), primals_209, out=buf170)
            del primals_209
            buf172 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_888, linear_158, silu_22, linear_159, mul_436, convert_element_type_977, reciprocal_5, mul_437, mul_438, sub_38, mul_439, add_293, mul_440, convert_element_type_979, view_891, permute_535, mm_273], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf171, (8192, 128), (1, 8192), 0), view_603, out=buf172)
            del view_603
            assert_size_stride(primals_208, (8192, 3072), (3072, 1), 'input')
            buf173 = buf159; del buf159  # reuse
            # Topologically Sorted Source Nodes: [view_888, linear_158, silu_22, linear_159, mul_436, convert_element_type_977, reciprocal_5, mul_437, mul_438, sub_38, mul_439, add_293, mul_440, convert_element_type_979, view_891, permute_537, mm_274], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf171, (128, 8192), (8192, 1), 0), primals_208, out=buf173)
            del primals_208
            assert_size_stride(primals_207, (3072, ), (1, ), 'input')
            assert_size_stride(add_181, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_45, (1, 128, 1), (128, 1, 1), 'input')
            buf176 = buf165; del buf165  # reuse
            # Topologically Sorted Source Nodes: [view_890, view_892, add_294, mul_441, hidden_states_226, convert_element_type_984, mul_443, mul_444, sum_71, pow_80, mul_445, mul_446, expand_184, div_69, pow_81, mul_447, mul_448, add_295, convert_element_type_985, add_296], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf176, buf170, buf173, primals_207, add_181, rsqrt_45, 128, 3072, stream=raw_stream0)
            del primals_207
            buf174 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_890, view_892, add_294, hidden_states_226, hidden_states_227, to_119, mul_442, sum_70], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf170, buf173, add_181, rsqrt_45, buf174, 3072, 128, stream=raw_stream0)
            del add_181
            del rsqrt_45
            assert_size_stride(view_601, (128, 3072), (3072, 1), 'input')
            buf177 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_894, permute_539, mm_275], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf176, (3072, 128), (1, 3072), 0), view_601, out=buf177)
            del view_601
            assert_size_stride(primals_206, (3072, 3072), (3072, 1), 'input')
            buf178 = buf173; del buf173  # reuse
            # Topologically Sorted Source Nodes: [view_894, attn_output_91, permute_541, mm_276], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf176, (128, 3072), (3072, 1), 0), primals_206, out=buf178)
            del primals_206
            assert_size_stride(permute_544, (24, 128, 128), (16384, 1, 128), 'input')
            buf179 = reinterpret_tensor(buf170, (24, 128, 128), (16384, 128, 1), 0); del buf170  # reuse
            # Topologically Sorted Source Nodes: [view_895, view_896, permute_543, view_897, bmm_76], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_544, reinterpret_tensor(buf178, (24, 128, 128), (128, 3072, 1), 0), out=buf179)
            del permute_544
            assert_size_stride(permute_545, (24, 128, 128), (16384, 1, 128), 'input')
            buf180 = reinterpret_tensor(buf156, (24, 128, 128), (16384, 128, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [view_895, view_896, permute_543, view_897, bmm_77], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf178, (24, 128, 128), (128, 3072, 1), 0), permute_545, out=buf180)
            del permute_545
            buf186 = reinterpret_tensor(buf157, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf157  # reuse
            # Topologically Sorted Source Nodes: [view_898, view_903, sum_73, squeeze_11, permute_549, clone_129], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf179, buf186, 131072, stream=raw_stream0)
            assert_size_stride(add_180, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_22, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_23, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf182 = add_180; del add_180  # reuse
            # Topologically Sorted Source Nodes: [view_899, convert_element_type_994, softmax_22, mul_449, sum_72, neg_106, fma_5, convert_element_type_995, mul_450], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf182, buf180, amax_22, sum_23, 3072, 128, stream=raw_stream0)
            del amax_22
            del sum_23
            assert_size_stride(view_583, (128, 3072), (3072, 1), 'input')
            buf187 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_898, view_903, sum_73, squeeze_11, permute_549, clone_129, view_905, view_906, permute_550, mm_277], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf186, (1024, 128), (1, 1024), 0), view_583, out=buf187)
            assert_size_stride(primals_205, (1024, 3072), (3072, 1), 'input')
            buf188 = reinterpret_tensor(buf180, (128, 3072), (3072, 1), 0); del buf180  # reuse
            # Topologically Sorted Source Nodes: [view_898, view_903, sum_73, squeeze_11, permute_549, clone_129, view_905, view_906, linear_156, permute_552, mm_278], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf186, (128, 1024), (1024, 1), 0), primals_205, out=buf188)
            del primals_205
            assert_size_stride(permute_546, (24, 128, 128), (128, 1, 3072), 'input')
            buf183 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [view_899, convert_element_type_994, softmax_22, mul_449, neg_106, fma_5, convert_element_type_995, mul_450, view_900, bmm_78], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_546, reinterpret_tensor(buf182, (24, 128, 128), (16384, 128, 1), 0), out=buf183)
            del permute_546
            assert_size_stride(permute_547, (24, 128, 128), (16384, 128, 1), 'input')
            buf184 = reinterpret_tensor(buf178, (24, 128, 128), (16384, 128, 1), 0); del buf178  # reuse
            # Topologically Sorted Source Nodes: [view_899, convert_element_type_994, softmax_22, mul_449, neg_106, fma_5, convert_element_type_995, mul_450, view_900, bmm_79], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf182, (24, 128, 128), (16384, 128, 1), 0), permute_547, out=buf184)
            del buf182
            del permute_547
            buf185 = reinterpret_tensor(buf186, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf186  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_901, permute_548, view_904, sum_74, squeeze_12, mul_451, slice_137, slice_138, neg_107, add_297, mul_452, add_298], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf183, primals_3, buf185, 131072, stream=raw_stream0)
            buf192 = reinterpret_tensor(buf183, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_902, mul_453, slice_139, slice_140, neg_108, add_299, mul_454, add_300, permute_559, clone_131], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf184, primals_3, buf192, 393216, stream=raw_stream0)
            buf189 = reinterpret_tensor(buf153, (128, 1024), (1024, 1), 0); del buf153  # reuse
            # Topologically Sorted Source Nodes: [permute_554, clone_130, view_908, view_909], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf185, buf189, 128, 1024, stream=raw_stream0)
            buf193 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_902, mul_453, slice_139, slice_140, neg_108, add_299, mul_454, add_300, permute_559, clone_131, view_911, view_912, permute_560, mm_281], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf192, (3072, 128), (1, 3072), 0), view_583, out=buf193)
            assert_size_stride(primals_203, (3072, 3072), (3072, 1), 'input')
            buf194 = reinterpret_tensor(buf184, (128, 3072), (3072, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_902, mul_453, slice_139, slice_140, neg_108, add_299, mul_454, add_300, permute_559, clone_131, view_911, view_912, linear_154, permute_562, mm_282], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf192, (128, 3072), (3072, 1), 0), primals_203, out=buf194)
            del primals_203
            buf190 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_555, mm_279], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf189, (1024, 128), (1, 1024), 0), view_583, out=buf190)
            del view_583
            assert_size_stride(primals_204, (1024, 3072), (3072, 1), 'input')
            buf191 = reinterpret_tensor(buf192, (128, 3072), (3072, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [linear_155, permute_557, mm_280], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf189, primals_204, out=buf191)
            del primals_204
            assert_size_stride(primals_202, (3072, ), (1, ), 'input')
            assert_size_stride(add_176, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_44, (1, 128, 1), (128, 1, 1), 'input')
            buf197 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [view_907, view_910, add_301, view_913, add_302, mul_455, hidden_states_220, convert_element_type_1012, mul_457, mul_458, sum_76, pow_82, mul_459, mul_460, expand_185, div_70, pow_83, mul_461, mul_462, add_303, convert_element_type_1013, add_304], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf197, buf188, buf191, buf194, primals_202, add_176, rsqrt_44, 128, 3072, stream=raw_stream0)
            del primals_202
            buf195 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_907, view_910, add_301, view_913, add_302, hidden_states_220, hidden_states_221, to_116, mul_456, sum_75], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf188, buf191, buf194, add_176, rsqrt_44, buf195, 3072, 128, stream=raw_stream0)
            del add_176
            del rsqrt_44
            assert_size_stride(view_581, (128, 8192), (8192, 1), 'input')
            buf198 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_915, permute_564, mm_283], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (3072, 128), (1, 3072), 0), view_581, out=buf198)
            del view_581
            assert_size_stride(primals_201, (3072, 8192), (8192, 1), 'input')
            buf199 = reinterpret_tensor(buf171, (128, 8192), (8192, 1), 0); del buf171  # reuse
            # Topologically Sorted Source Nodes: [view_915, down_proj_21, permute_566, mm_284], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (128, 3072), (3072, 1), 0), primals_201, out=buf199)
            del primals_201
            assert_size_stride(mm_151, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_152, (128, 8192), (8192, 1), 'input')
            buf200 = buf168; del buf168  # reuse
            buf203 = reinterpret_tensor(mm_152, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_152  # reuse
            # Topologically Sorted Source Nodes: [view_916, linear_151, silu_21, mul_463, linear_152, mul_464, convert_element_type_1022, reciprocal_6, mul_465, mul_466, sub_39, mul_467, add_306, mul_468, convert_element_type_1024], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf203, buf199, mm_151, buf200, 1048576, stream=raw_stream0)
            del buf199
            del mm_151
            assert_size_stride(view_577, (128, 3072), (3072, 1), 'input')
            buf201 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_916, linear_151, silu_21, mul_463, view_917, permute_568, mm_285], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf200, (8192, 128), (1, 8192), 0), view_577, out=buf201)
            assert_size_stride(primals_200, (8192, 3072), (3072, 1), 'input')
            buf202 = buf194; del buf194  # reuse
            # Topologically Sorted Source Nodes: [view_916, linear_151, silu_21, mul_463, view_917, linear_152, permute_570, mm_286], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf200, (128, 8192), (8192, 1), 0), primals_200, out=buf202)
            del primals_200
            buf204 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_916, linear_151, silu_21, linear_152, mul_464, convert_element_type_1022, reciprocal_6, mul_465, mul_466, sub_39, mul_467, add_306, mul_468, convert_element_type_1024, view_919, permute_572, mm_287], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf203, (8192, 128), (1, 8192), 0), view_577, out=buf204)
            del view_577
            assert_size_stride(primals_199, (8192, 3072), (3072, 1), 'input')
            buf205 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [view_916, linear_151, silu_21, linear_152, mul_464, convert_element_type_1022, reciprocal_6, mul_465, mul_466, sub_39, mul_467, add_306, mul_468, convert_element_type_1024, view_919, permute_574, mm_288], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf203, (128, 8192), (8192, 1), 0), primals_199, out=buf205)
            del primals_199
            assert_size_stride(primals_198, (3072, ), (1, ), 'input')
            assert_size_stride(add_173, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_43, (1, 128, 1), (128, 1, 1), 'input')
            buf208 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [view_918, view_920, add_307, mul_469, hidden_states_216, convert_element_type_1029, mul_471, mul_472, sum_78, pow_84, mul_473, mul_474, expand_186, div_71, pow_85, mul_475, mul_476, add_308, convert_element_type_1030, add_309], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf208, buf202, buf205, primals_198, add_173, rsqrt_43, 128, 3072, stream=raw_stream0)
            del primals_198
            buf206 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_918, view_920, add_307, hidden_states_216, hidden_states_217, to_114, mul_470, sum_77], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf202, buf205, add_173, rsqrt_43, buf206, 3072, 128, stream=raw_stream0)
            del add_173
            del rsqrt_43
            assert_size_stride(view_575, (128, 3072), (3072, 1), 'input')
            buf209 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_922, permute_576, mm_289], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf208, (3072, 128), (1, 3072), 0), view_575, out=buf209)
            del view_575
            assert_size_stride(primals_197, (3072, 3072), (3072, 1), 'input')
            buf210 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [view_922, attn_output_87, permute_578, mm_290], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf208, (128, 3072), (3072, 1), 0), primals_197, out=buf210)
            del primals_197
            assert_size_stride(permute_581, (24, 128, 128), (16384, 1, 128), 'input')
            buf211 = reinterpret_tensor(buf202, (24, 128, 128), (16384, 128, 1), 0); del buf202  # reuse
            # Topologically Sorted Source Nodes: [view_923, view_924, permute_580, view_925, bmm_80], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_581, reinterpret_tensor(buf210, (24, 128, 128), (128, 3072, 1), 0), out=buf211)
            del permute_581
            assert_size_stride(permute_582, (24, 128, 128), (16384, 1, 128), 'input')
            buf212 = reinterpret_tensor(buf188, (24, 128, 128), (16384, 128, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [view_923, view_924, permute_580, view_925, bmm_81], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf210, (24, 128, 128), (128, 3072, 1), 0), permute_582, out=buf212)
            del permute_582
            buf218 = reinterpret_tensor(buf189, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf189  # reuse
            # Topologically Sorted Source Nodes: [view_926, view_931, sum_80, squeeze_13, permute_586, clone_132], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf211, buf218, 131072, stream=raw_stream0)
            assert_size_stride(add_172, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_21, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_22, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf214 = add_172; del add_172  # reuse
            # Topologically Sorted Source Nodes: [view_927, convert_element_type_1039, softmax_21, mul_477, sum_79, neg_110, fma_6, convert_element_type_1040, mul_478], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf214, buf212, amax_21, sum_22, 3072, 128, stream=raw_stream0)
            del amax_21
            del sum_22
            assert_size_stride(view_557, (128, 3072), (3072, 1), 'input')
            buf219 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_926, view_931, sum_80, squeeze_13, permute_586, clone_132, view_933, view_934, permute_587, mm_291], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf218, (1024, 128), (1, 1024), 0), view_557, out=buf219)
            assert_size_stride(primals_196, (1024, 3072), (3072, 1), 'input')
            buf220 = reinterpret_tensor(buf212, (128, 3072), (3072, 1), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [view_926, view_931, sum_80, squeeze_13, permute_586, clone_132, view_933, view_934, linear_149, permute_589, mm_292], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf218, (128, 1024), (1024, 1), 0), primals_196, out=buf220)
            del primals_196
            assert_size_stride(permute_583, (24, 128, 128), (128, 1, 3072), 'input')
            buf215 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [view_927, convert_element_type_1039, softmax_21, mul_477, neg_110, fma_6, convert_element_type_1040, mul_478, view_928, bmm_82], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_583, reinterpret_tensor(buf214, (24, 128, 128), (16384, 128, 1), 0), out=buf215)
            del permute_583
            assert_size_stride(permute_584, (24, 128, 128), (16384, 128, 1), 'input')
            buf216 = reinterpret_tensor(buf210, (24, 128, 128), (16384, 128, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [view_927, convert_element_type_1039, softmax_21, mul_477, neg_110, fma_6, convert_element_type_1040, mul_478, view_928, bmm_83], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf214, (24, 128, 128), (16384, 128, 1), 0), permute_584, out=buf216)
            del buf214
            del permute_584
            buf217 = reinterpret_tensor(buf218, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf218  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_929, permute_585, view_932, sum_81, squeeze_14, mul_479, slice_141, slice_142, neg_111, add_310, mul_480, add_311], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf215, primals_3, buf217, 131072, stream=raw_stream0)
            buf224 = reinterpret_tensor(buf215, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf215  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_930, mul_481, slice_143, slice_144, neg_112, add_312, mul_482, add_313, permute_596, clone_134], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf216, primals_3, buf224, 393216, stream=raw_stream0)
            buf221 = reinterpret_tensor(buf185, (128, 1024), (1024, 1), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [permute_591, clone_133, view_936, view_937], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf217, buf221, 128, 1024, stream=raw_stream0)
            buf225 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_930, mul_481, slice_143, slice_144, neg_112, add_312, mul_482, add_313, permute_596, clone_134, view_939, view_940, permute_597, mm_295], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf224, (3072, 128), (1, 3072), 0), view_557, out=buf225)
            assert_size_stride(primals_194, (3072, 3072), (3072, 1), 'input')
            buf226 = reinterpret_tensor(buf216, (128, 3072), (3072, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_930, mul_481, slice_143, slice_144, neg_112, add_312, mul_482, add_313, permute_596, clone_134, view_939, view_940, linear_147, permute_599, mm_296], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf224, (128, 3072), (3072, 1), 0), primals_194, out=buf226)
            del primals_194
            buf222 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_592, mm_293], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf221, (1024, 128), (1, 1024), 0), view_557, out=buf222)
            del view_557
            assert_size_stride(primals_195, (1024, 3072), (3072, 1), 'input')
            buf223 = reinterpret_tensor(buf224, (128, 3072), (3072, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [linear_148, permute_594, mm_294], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf221, primals_195, out=buf223)
            del primals_195
            assert_size_stride(primals_193, (3072, ), (1, ), 'input')
            assert_size_stride(add_168, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_42, (1, 128, 1), (128, 1, 1), 'input')
            buf229 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [view_935, view_938, add_314, view_941, add_315, mul_483, hidden_states_210, convert_element_type_1057, mul_485, mul_486, sum_83, pow_86, mul_487, mul_488, expand_187, div_72, pow_87, mul_489, mul_490, add_316, convert_element_type_1058, add_317], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf229, buf220, buf223, buf226, primals_193, add_168, rsqrt_42, 128, 3072, stream=raw_stream0)
            del primals_193
            buf227 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_935, view_938, add_314, view_941, add_315, hidden_states_210, hidden_states_211, to_111, mul_484, sum_82], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf220, buf223, buf226, add_168, rsqrt_42, buf227, 3072, 128, stream=raw_stream0)
            del add_168
            del rsqrt_42
            assert_size_stride(view_555, (128, 8192), (8192, 1), 'input')
            buf230 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_943, permute_601, mm_297], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf229, (3072, 128), (1, 3072), 0), view_555, out=buf230)
            del view_555
            assert_size_stride(primals_192, (3072, 8192), (8192, 1), 'input')
            buf231 = reinterpret_tensor(buf203, (128, 8192), (8192, 1), 0); del buf203  # reuse
            # Topologically Sorted Source Nodes: [view_943, down_proj_20, permute_603, mm_298], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf229, (128, 3072), (3072, 1), 0), primals_192, out=buf231)
            del primals_192
            assert_size_stride(mm_144, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_145, (128, 8192), (8192, 1), 'input')
            buf232 = buf200; del buf200  # reuse
            buf235 = reinterpret_tensor(mm_145, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_145  # reuse
            # Topologically Sorted Source Nodes: [view_944, linear_144, silu_20, mul_491, linear_145, mul_492, convert_element_type_1067, reciprocal_7, mul_493, mul_494, sub_40, mul_495, add_319, mul_496, convert_element_type_1069], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf235, buf231, mm_144, buf232, 1048576, stream=raw_stream0)
            del buf231
            del mm_144
            assert_size_stride(view_551, (128, 3072), (3072, 1), 'input')
            buf233 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_944, linear_144, silu_20, mul_491, view_945, permute_605, mm_299], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (8192, 128), (1, 8192), 0), view_551, out=buf233)
            assert_size_stride(primals_191, (8192, 3072), (3072, 1), 'input')
            buf234 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [view_944, linear_144, silu_20, mul_491, view_945, linear_145, permute_607, mm_300], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (128, 8192), (8192, 1), 0), primals_191, out=buf234)
            del primals_191
            buf236 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_944, linear_144, silu_20, linear_145, mul_492, convert_element_type_1067, reciprocal_7, mul_493, mul_494, sub_40, mul_495, add_319, mul_496, convert_element_type_1069, view_947, permute_609, mm_301], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf235, (8192, 128), (1, 8192), 0), view_551, out=buf236)
            del view_551
            assert_size_stride(primals_190, (8192, 3072), (3072, 1), 'input')
            buf237 = buf223; del buf223  # reuse
            # Topologically Sorted Source Nodes: [view_944, linear_144, silu_20, linear_145, mul_492, convert_element_type_1067, reciprocal_7, mul_493, mul_494, sub_40, mul_495, add_319, mul_496, convert_element_type_1069, view_947, permute_611, mm_302], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf235, (128, 8192), (8192, 1), 0), primals_190, out=buf237)
            del primals_190
            assert_size_stride(primals_189, (3072, ), (1, ), 'input')
            assert_size_stride(add_165, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_41, (1, 128, 1), (128, 1, 1), 'input')
            buf240 = buf229; del buf229  # reuse
            # Topologically Sorted Source Nodes: [view_946, view_948, add_320, mul_497, hidden_states_206, convert_element_type_1074, mul_499, mul_500, sum_85, pow_88, mul_501, mul_502, expand_188, div_73, pow_89, mul_503, mul_504, add_321, convert_element_type_1075, add_322], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf240, buf234, buf237, primals_189, add_165, rsqrt_41, 128, 3072, stream=raw_stream0)
            del primals_189
            buf238 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_946, view_948, add_320, hidden_states_206, hidden_states_207, to_109, mul_498, sum_84], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf234, buf237, add_165, rsqrt_41, buf238, 3072, 128, stream=raw_stream0)
            del add_165
            del rsqrt_41
            assert_size_stride(view_549, (128, 3072), (3072, 1), 'input')
            buf241 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_950, permute_613, mm_303], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (3072, 128), (1, 3072), 0), view_549, out=buf241)
            del view_549
            assert_size_stride(primals_188, (3072, 3072), (3072, 1), 'input')
            buf242 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [view_950, attn_output_83, permute_615, mm_304], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf240, (128, 3072), (3072, 1), 0), primals_188, out=buf242)
            del primals_188
            assert_size_stride(permute_618, (24, 128, 128), (16384, 1, 128), 'input')
            buf243 = reinterpret_tensor(buf234, (24, 128, 128), (16384, 128, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [view_951, view_952, permute_617, view_953, bmm_84], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_618, reinterpret_tensor(buf242, (24, 128, 128), (128, 3072, 1), 0), out=buf243)
            del permute_618
            assert_size_stride(permute_619, (24, 128, 128), (16384, 1, 128), 'input')
            buf244 = reinterpret_tensor(buf220, (24, 128, 128), (16384, 128, 1), 0); del buf220  # reuse
            # Topologically Sorted Source Nodes: [view_951, view_952, permute_617, view_953, bmm_85], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf242, (24, 128, 128), (128, 3072, 1), 0), permute_619, out=buf244)
            del permute_619
            buf250 = reinterpret_tensor(buf221, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf221  # reuse
            # Topologically Sorted Source Nodes: [view_954, view_959, sum_87, squeeze_15, permute_623, clone_135], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf243, buf250, 131072, stream=raw_stream0)
            assert_size_stride(add_164, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_20, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_21, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf246 = add_164; del add_164  # reuse
            # Topologically Sorted Source Nodes: [view_955, convert_element_type_1084, softmax_20, mul_505, sum_86, neg_114, fma_7, convert_element_type_1085, mul_506], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf246, buf244, amax_20, sum_21, 3072, 128, stream=raw_stream0)
            del amax_20
            del sum_21
            assert_size_stride(view_531, (128, 3072), (3072, 1), 'input')
            buf251 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_954, view_959, sum_87, squeeze_15, permute_623, clone_135, view_961, view_962, permute_624, mm_305], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf250, (1024, 128), (1, 1024), 0), view_531, out=buf251)
            assert_size_stride(primals_187, (1024, 3072), (3072, 1), 'input')
            buf252 = reinterpret_tensor(buf244, (128, 3072), (3072, 1), 0); del buf244  # reuse
            # Topologically Sorted Source Nodes: [view_954, view_959, sum_87, squeeze_15, permute_623, clone_135, view_961, view_962, linear_142, permute_626, mm_306], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf250, (128, 1024), (1024, 1), 0), primals_187, out=buf252)
            del primals_187
            assert_size_stride(permute_620, (24, 128, 128), (128, 1, 3072), 'input')
            buf247 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [view_955, convert_element_type_1084, softmax_20, mul_505, neg_114, fma_7, convert_element_type_1085, mul_506, view_956, bmm_86], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_620, reinterpret_tensor(buf246, (24, 128, 128), (16384, 128, 1), 0), out=buf247)
            del permute_620
            assert_size_stride(permute_621, (24, 128, 128), (16384, 128, 1), 'input')
            buf248 = reinterpret_tensor(buf242, (24, 128, 128), (16384, 128, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [view_955, convert_element_type_1084, softmax_20, mul_505, neg_114, fma_7, convert_element_type_1085, mul_506, view_956, bmm_87], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf246, (24, 128, 128), (16384, 128, 1), 0), permute_621, out=buf248)
            del buf246
            del permute_621
            buf249 = reinterpret_tensor(buf250, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf250  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_957, permute_622, view_960, sum_88, squeeze_16, mul_507, slice_145, slice_146, neg_115, add_323, mul_508, add_324], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf247, primals_3, buf249, 131072, stream=raw_stream0)
            buf256 = reinterpret_tensor(buf247, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf247  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_958, mul_509, slice_147, slice_148, neg_116, add_325, mul_510, add_326, permute_633, clone_137], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf248, primals_3, buf256, 393216, stream=raw_stream0)
            buf253 = reinterpret_tensor(buf217, (128, 1024), (1024, 1), 0); del buf217  # reuse
            # Topologically Sorted Source Nodes: [permute_628, clone_136, view_964, view_965], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf249, buf253, 128, 1024, stream=raw_stream0)
            buf257 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_958, mul_509, slice_147, slice_148, neg_116, add_325, mul_510, add_326, permute_633, clone_137, view_967, view_968, permute_634, mm_309], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf256, (3072, 128), (1, 3072), 0), view_531, out=buf257)
            assert_size_stride(primals_185, (3072, 3072), (3072, 1), 'input')
            buf258 = reinterpret_tensor(buf248, (128, 3072), (3072, 1), 0); del buf248  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_958, mul_509, slice_147, slice_148, neg_116, add_325, mul_510, add_326, permute_633, clone_137, view_967, view_968, linear_140, permute_636, mm_310], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf256, (128, 3072), (3072, 1), 0), primals_185, out=buf258)
            del primals_185
            buf254 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_629, mm_307], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf253, (1024, 128), (1, 1024), 0), view_531, out=buf254)
            del view_531
            assert_size_stride(primals_186, (1024, 3072), (3072, 1), 'input')
            buf255 = reinterpret_tensor(buf256, (128, 3072), (3072, 1), 0); del buf256  # reuse
            # Topologically Sorted Source Nodes: [linear_141, permute_631, mm_308], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf253, primals_186, out=buf255)
            del primals_186
            assert_size_stride(primals_184, (3072, ), (1, ), 'input')
            assert_size_stride(add_160, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_40, (1, 128, 1), (128, 1, 1), 'input')
            buf261 = buf240; del buf240  # reuse
            # Topologically Sorted Source Nodes: [view_963, view_966, add_327, view_969, add_328, mul_511, hidden_states_200, convert_element_type_1102, mul_513, mul_514, sum_90, pow_90, mul_515, mul_516, expand_189, div_74, pow_91, mul_517, mul_518, add_329, convert_element_type_1103, add_330], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf261, buf252, buf255, buf258, primals_184, add_160, rsqrt_40, 128, 3072, stream=raw_stream0)
            del primals_184
            buf259 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_963, view_966, add_327, view_969, add_328, hidden_states_200, hidden_states_201, to_106, mul_512, sum_89], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf252, buf255, buf258, add_160, rsqrt_40, buf259, 3072, 128, stream=raw_stream0)
            del add_160
            del rsqrt_40
            assert_size_stride(view_529, (128, 8192), (8192, 1), 'input')
            buf262 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_971, permute_638, mm_311], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf261, (3072, 128), (1, 3072), 0), view_529, out=buf262)
            del view_529
            assert_size_stride(primals_183, (3072, 8192), (8192, 1), 'input')
            buf263 = reinterpret_tensor(buf235, (128, 8192), (8192, 1), 0); del buf235  # reuse
            # Topologically Sorted Source Nodes: [view_971, down_proj_19, permute_640, mm_312], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf261, (128, 3072), (3072, 1), 0), primals_183, out=buf263)
            del primals_183
            assert_size_stride(mm_137, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_138, (128, 8192), (8192, 1), 'input')
            buf264 = buf232; del buf232  # reuse
            buf267 = reinterpret_tensor(mm_138, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_138  # reuse
            # Topologically Sorted Source Nodes: [view_972, linear_137, silu_19, mul_519, linear_138, mul_520, convert_element_type_1112, reciprocal_8, mul_521, mul_522, sub_41, mul_523, add_332, mul_524, convert_element_type_1114], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf267, buf263, mm_137, buf264, 1048576, stream=raw_stream0)
            del buf263
            del mm_137
            assert_size_stride(view_525, (128, 3072), (3072, 1), 'input')
            buf265 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_972, linear_137, silu_19, mul_519, view_973, permute_642, mm_313], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf264, (8192, 128), (1, 8192), 0), view_525, out=buf265)
            assert_size_stride(primals_182, (8192, 3072), (3072, 1), 'input')
            buf266 = buf258; del buf258  # reuse
            # Topologically Sorted Source Nodes: [view_972, linear_137, silu_19, mul_519, view_973, linear_138, permute_644, mm_314], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf264, (128, 8192), (8192, 1), 0), primals_182, out=buf266)
            del primals_182
            buf268 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_972, linear_137, silu_19, linear_138, mul_520, convert_element_type_1112, reciprocal_8, mul_521, mul_522, sub_41, mul_523, add_332, mul_524, convert_element_type_1114, view_975, permute_646, mm_315], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf267, (8192, 128), (1, 8192), 0), view_525, out=buf268)
            del view_525
            assert_size_stride(primals_181, (8192, 3072), (3072, 1), 'input')
            buf269 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [view_972, linear_137, silu_19, linear_138, mul_520, convert_element_type_1112, reciprocal_8, mul_521, mul_522, sub_41, mul_523, add_332, mul_524, convert_element_type_1114, view_975, permute_648, mm_316], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf267, (128, 8192), (8192, 1), 0), primals_181, out=buf269)
            del primals_181
            assert_size_stride(primals_180, (3072, ), (1, ), 'input')
            assert_size_stride(add_157, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_39, (1, 128, 1), (128, 1, 1), 'input')
            buf272 = buf261; del buf261  # reuse
            # Topologically Sorted Source Nodes: [view_974, view_976, add_333, mul_525, hidden_states_196, convert_element_type_1119, mul_527, mul_528, sum_92, pow_92, mul_529, mul_530, expand_190, div_75, pow_93, mul_531, mul_532, add_334, convert_element_type_1120, add_335], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf272, buf266, buf269, primals_180, add_157, rsqrt_39, 128, 3072, stream=raw_stream0)
            del primals_180
            buf270 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_974, view_976, add_333, hidden_states_196, hidden_states_197, to_104, mul_526, sum_91], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf266, buf269, add_157, rsqrt_39, buf270, 3072, 128, stream=raw_stream0)
            del add_157
            del rsqrt_39
            assert_size_stride(view_523, (128, 3072), (3072, 1), 'input')
            buf273 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_978, permute_650, mm_317], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf272, (3072, 128), (1, 3072), 0), view_523, out=buf273)
            del view_523
            assert_size_stride(primals_179, (3072, 3072), (3072, 1), 'input')
            buf274 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [view_978, attn_output_79, permute_652, mm_318], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf272, (128, 3072), (3072, 1), 0), primals_179, out=buf274)
            del primals_179
            assert_size_stride(permute_655, (24, 128, 128), (16384, 1, 128), 'input')
            buf275 = reinterpret_tensor(buf266, (24, 128, 128), (16384, 128, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [view_979, view_980, permute_654, view_981, bmm_88], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_655, reinterpret_tensor(buf274, (24, 128, 128), (128, 3072, 1), 0), out=buf275)
            del permute_655
            assert_size_stride(permute_656, (24, 128, 128), (16384, 1, 128), 'input')
            buf276 = reinterpret_tensor(buf252, (24, 128, 128), (16384, 128, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [view_979, view_980, permute_654, view_981, bmm_89], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf274, (24, 128, 128), (128, 3072, 1), 0), permute_656, out=buf276)
            del permute_656
            buf282 = reinterpret_tensor(buf253, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf253  # reuse
            # Topologically Sorted Source Nodes: [view_982, view_987, sum_94, squeeze_17, permute_660, clone_138], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf275, buf282, 131072, stream=raw_stream0)
            assert_size_stride(add_156, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_19, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_20, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf278 = add_156; del add_156  # reuse
            # Topologically Sorted Source Nodes: [view_983, convert_element_type_1129, softmax_19, mul_533, sum_93, neg_118, fma_8, convert_element_type_1130, mul_534], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf278, buf276, amax_19, sum_20, 3072, 128, stream=raw_stream0)
            del amax_19
            del sum_20
            assert_size_stride(view_505, (128, 3072), (3072, 1), 'input')
            buf283 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_982, view_987, sum_94, squeeze_17, permute_660, clone_138, view_989, view_990, permute_661, mm_319], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf282, (1024, 128), (1, 1024), 0), view_505, out=buf283)
            assert_size_stride(primals_178, (1024, 3072), (3072, 1), 'input')
            buf284 = reinterpret_tensor(buf276, (128, 3072), (3072, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [view_982, view_987, sum_94, squeeze_17, permute_660, clone_138, view_989, view_990, linear_135, permute_663, mm_320], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf282, (128, 1024), (1024, 1), 0), primals_178, out=buf284)
            del primals_178
            assert_size_stride(permute_657, (24, 128, 128), (128, 1, 3072), 'input')
            buf279 = buf275; del buf275  # reuse
            # Topologically Sorted Source Nodes: [view_983, convert_element_type_1129, softmax_19, mul_533, neg_118, fma_8, convert_element_type_1130, mul_534, view_984, bmm_90], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_657, reinterpret_tensor(buf278, (24, 128, 128), (16384, 128, 1), 0), out=buf279)
            del permute_657
            assert_size_stride(permute_658, (24, 128, 128), (16384, 128, 1), 'input')
            buf280 = reinterpret_tensor(buf274, (24, 128, 128), (16384, 128, 1), 0); del buf274  # reuse
            # Topologically Sorted Source Nodes: [view_983, convert_element_type_1129, softmax_19, mul_533, neg_118, fma_8, convert_element_type_1130, mul_534, view_984, bmm_91], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf278, (24, 128, 128), (16384, 128, 1), 0), permute_658, out=buf280)
            del buf278
            del permute_658
            buf281 = reinterpret_tensor(buf282, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf282  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_985, permute_659, view_988, sum_95, squeeze_18, mul_535, slice_149, slice_150, neg_119, add_336, mul_536, add_337], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf279, primals_3, buf281, 131072, stream=raw_stream0)
            buf288 = reinterpret_tensor(buf279, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf279  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_986, mul_537, slice_151, slice_152, neg_120, add_338, mul_538, add_339, permute_670, clone_140], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf280, primals_3, buf288, 393216, stream=raw_stream0)
            buf285 = reinterpret_tensor(buf249, (128, 1024), (1024, 1), 0); del buf249  # reuse
            # Topologically Sorted Source Nodes: [permute_665, clone_139, view_992, view_993], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf281, buf285, 128, 1024, stream=raw_stream0)
            buf289 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_986, mul_537, slice_151, slice_152, neg_120, add_338, mul_538, add_339, permute_670, clone_140, view_995, view_996, permute_671, mm_323], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf288, (3072, 128), (1, 3072), 0), view_505, out=buf289)
            assert_size_stride(primals_176, (3072, 3072), (3072, 1), 'input')
            buf290 = reinterpret_tensor(buf280, (128, 3072), (3072, 1), 0); del buf280  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_986, mul_537, slice_151, slice_152, neg_120, add_338, mul_538, add_339, permute_670, clone_140, view_995, view_996, linear_133, permute_673, mm_324], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf288, (128, 3072), (3072, 1), 0), primals_176, out=buf290)
            del primals_176
            buf286 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_666, mm_321], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf285, (1024, 128), (1, 1024), 0), view_505, out=buf286)
            del view_505
            assert_size_stride(primals_177, (1024, 3072), (3072, 1), 'input')
            buf287 = reinterpret_tensor(buf288, (128, 3072), (3072, 1), 0); del buf288  # reuse
            # Topologically Sorted Source Nodes: [linear_134, permute_668, mm_322], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf285, primals_177, out=buf287)
            del primals_177
            assert_size_stride(primals_175, (3072, ), (1, ), 'input')
            assert_size_stride(add_152, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_38, (1, 128, 1), (128, 1, 1), 'input')
            buf293 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [view_991, view_994, add_340, view_997, add_341, mul_539, hidden_states_190, convert_element_type_1147, mul_541, mul_542, sum_97, pow_94, mul_543, mul_544, expand_191, div_76, pow_95, mul_545, mul_546, add_342, convert_element_type_1148, add_343], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf293, buf284, buf287, buf290, primals_175, add_152, rsqrt_38, 128, 3072, stream=raw_stream0)
            del primals_175
            buf291 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_991, view_994, add_340, view_997, add_341, hidden_states_190, hidden_states_191, to_101, mul_540, sum_96], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf284, buf287, buf290, add_152, rsqrt_38, buf291, 3072, 128, stream=raw_stream0)
            del add_152
            del rsqrt_38
            assert_size_stride(view_503, (128, 8192), (8192, 1), 'input')
            buf294 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_999, permute_675, mm_325], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf293, (3072, 128), (1, 3072), 0), view_503, out=buf294)
            del view_503
            assert_size_stride(primals_174, (3072, 8192), (8192, 1), 'input')
            buf295 = reinterpret_tensor(buf267, (128, 8192), (8192, 1), 0); del buf267  # reuse
            # Topologically Sorted Source Nodes: [view_999, down_proj_18, permute_677, mm_326], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf293, (128, 3072), (3072, 1), 0), primals_174, out=buf295)
            del primals_174
            assert_size_stride(mm_130, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_131, (128, 8192), (8192, 1), 'input')
            buf296 = buf264; del buf264  # reuse
            buf299 = reinterpret_tensor(mm_131, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_131  # reuse
            # Topologically Sorted Source Nodes: [view_1000, linear_130, silu_18, mul_547, linear_131, mul_548, convert_element_type_1157, reciprocal_9, mul_549, mul_550, sub_42, mul_551, add_345, mul_552, convert_element_type_1159], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf299, buf295, mm_130, buf296, 1048576, stream=raw_stream0)
            del buf295
            del mm_130
            assert_size_stride(view_499, (128, 3072), (3072, 1), 'input')
            buf297 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1000, linear_130, silu_18, mul_547, view_1001, permute_679, mm_327], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf296, (8192, 128), (1, 8192), 0), view_499, out=buf297)
            assert_size_stride(primals_173, (8192, 3072), (3072, 1), 'input')
            buf298 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [view_1000, linear_130, silu_18, mul_547, view_1001, linear_131, permute_681, mm_328], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf296, (128, 8192), (8192, 1), 0), primals_173, out=buf298)
            del primals_173
            buf300 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1000, linear_130, silu_18, linear_131, mul_548, convert_element_type_1157, reciprocal_9, mul_549, mul_550, sub_42, mul_551, add_345, mul_552, convert_element_type_1159, view_1003, permute_683, mm_329], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf299, (8192, 128), (1, 8192), 0), view_499, out=buf300)
            del view_499
            assert_size_stride(primals_172, (8192, 3072), (3072, 1), 'input')
            buf301 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [view_1000, linear_130, silu_18, linear_131, mul_548, convert_element_type_1157, reciprocal_9, mul_549, mul_550, sub_42, mul_551, add_345, mul_552, convert_element_type_1159, view_1003, permute_685, mm_330], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf299, (128, 8192), (8192, 1), 0), primals_172, out=buf301)
            del primals_172
            assert_size_stride(primals_171, (3072, ), (1, ), 'input')
            assert_size_stride(add_149, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_37, (1, 128, 1), (128, 1, 1), 'input')
            buf304 = buf293; del buf293  # reuse
            # Topologically Sorted Source Nodes: [view_1002, view_1004, add_346, mul_553, hidden_states_186, convert_element_type_1164, mul_555, mul_556, sum_99, pow_96, mul_557, mul_558, expand_192, div_77, pow_97, mul_559, mul_560, add_347, convert_element_type_1165, add_348], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf304, buf298, buf301, primals_171, add_149, rsqrt_37, 128, 3072, stream=raw_stream0)
            del primals_171
            buf302 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1002, view_1004, add_346, hidden_states_186, hidden_states_187, to_99, mul_554, sum_98], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf298, buf301, add_149, rsqrt_37, buf302, 3072, 128, stream=raw_stream0)
            del add_149
            del rsqrt_37
            assert_size_stride(view_497, (128, 3072), (3072, 1), 'input')
            buf305 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1006, permute_687, mm_331], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf304, (3072, 128), (1, 3072), 0), view_497, out=buf305)
            del view_497
            assert_size_stride(primals_170, (3072, 3072), (3072, 1), 'input')
            buf306 = buf301; del buf301  # reuse
            # Topologically Sorted Source Nodes: [view_1006, attn_output_75, permute_689, mm_332], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf304, (128, 3072), (3072, 1), 0), primals_170, out=buf306)
            del primals_170
            assert_size_stride(permute_692, (24, 128, 128), (16384, 1, 128), 'input')
            buf307 = reinterpret_tensor(buf298, (24, 128, 128), (16384, 128, 1), 0); del buf298  # reuse
            # Topologically Sorted Source Nodes: [view_1007, view_1008, permute_691, view_1009, bmm_92], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_692, reinterpret_tensor(buf306, (24, 128, 128), (128, 3072, 1), 0), out=buf307)
            del permute_692
            assert_size_stride(permute_693, (24, 128, 128), (16384, 1, 128), 'input')
            buf308 = reinterpret_tensor(buf284, (24, 128, 128), (16384, 128, 1), 0); del buf284  # reuse
            # Topologically Sorted Source Nodes: [view_1007, view_1008, permute_691, view_1009, bmm_93], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf306, (24, 128, 128), (128, 3072, 1), 0), permute_693, out=buf308)
            del permute_693
            buf314 = reinterpret_tensor(buf285, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [view_1010, view_1015, sum_101, squeeze_19, permute_697, clone_141], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf307, buf314, 131072, stream=raw_stream0)
            assert_size_stride(add_148, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_18, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_19, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf310 = add_148; del add_148  # reuse
            # Topologically Sorted Source Nodes: [view_1011, convert_element_type_1174, softmax_18, mul_561, sum_100, neg_122, fma_9, convert_element_type_1175, mul_562], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf310, buf308, amax_18, sum_19, 3072, 128, stream=raw_stream0)
            del amax_18
            del sum_19
            assert_size_stride(view_479, (128, 3072), (3072, 1), 'input')
            buf315 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1010, view_1015, sum_101, squeeze_19, permute_697, clone_141, view_1017, view_1018, permute_698, mm_333], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf314, (1024, 128), (1, 1024), 0), view_479, out=buf315)
            assert_size_stride(primals_169, (1024, 3072), (3072, 1), 'input')
            buf316 = reinterpret_tensor(buf308, (128, 3072), (3072, 1), 0); del buf308  # reuse
            # Topologically Sorted Source Nodes: [view_1010, view_1015, sum_101, squeeze_19, permute_697, clone_141, view_1017, view_1018, linear_128, permute_700, mm_334], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf314, (128, 1024), (1024, 1), 0), primals_169, out=buf316)
            del primals_169
            assert_size_stride(permute_694, (24, 128, 128), (128, 1, 3072), 'input')
            buf311 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [view_1011, convert_element_type_1174, softmax_18, mul_561, neg_122, fma_9, convert_element_type_1175, mul_562, view_1012, bmm_94], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_694, reinterpret_tensor(buf310, (24, 128, 128), (16384, 128, 1), 0), out=buf311)
            del permute_694
            assert_size_stride(permute_695, (24, 128, 128), (16384, 128, 1), 'input')
            buf312 = reinterpret_tensor(buf306, (24, 128, 128), (16384, 128, 1), 0); del buf306  # reuse
            # Topologically Sorted Source Nodes: [view_1011, convert_element_type_1174, softmax_18, mul_561, neg_122, fma_9, convert_element_type_1175, mul_562, view_1012, bmm_95], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf310, (24, 128, 128), (16384, 128, 1), 0), permute_695, out=buf312)
            del buf310
            del permute_695
            buf313 = reinterpret_tensor(buf314, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1013, permute_696, view_1016, sum_102, squeeze_20, mul_563, slice_153, slice_154, neg_123, add_349, mul_564, add_350], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf311, primals_3, buf313, 131072, stream=raw_stream0)
            buf320 = reinterpret_tensor(buf311, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf311  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1014, mul_565, slice_155, slice_156, neg_124, add_351, mul_566, add_352, permute_707, clone_143], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf312, primals_3, buf320, 393216, stream=raw_stream0)
            buf317 = reinterpret_tensor(buf281, (128, 1024), (1024, 1), 0); del buf281  # reuse
            # Topologically Sorted Source Nodes: [permute_702, clone_142, view_1020, view_1021], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf313, buf317, 128, 1024, stream=raw_stream0)
            buf321 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1014, mul_565, slice_155, slice_156, neg_124, add_351, mul_566, add_352, permute_707, clone_143, view_1023, view_1024, permute_708, mm_337], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf320, (3072, 128), (1, 3072), 0), view_479, out=buf321)
            assert_size_stride(primals_167, (3072, 3072), (3072, 1), 'input')
            buf322 = reinterpret_tensor(buf312, (128, 3072), (3072, 1), 0); del buf312  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1014, mul_565, slice_155, slice_156, neg_124, add_351, mul_566, add_352, permute_707, clone_143, view_1023, view_1024, linear_126, permute_710, mm_338], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf320, (128, 3072), (3072, 1), 0), primals_167, out=buf322)
            del primals_167
            buf318 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_703, mm_335], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf317, (1024, 128), (1, 1024), 0), view_479, out=buf318)
            del view_479
            assert_size_stride(primals_168, (1024, 3072), (3072, 1), 'input')
            buf319 = reinterpret_tensor(buf320, (128, 3072), (3072, 1), 0); del buf320  # reuse
            # Topologically Sorted Source Nodes: [linear_127, permute_705, mm_336], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf317, primals_168, out=buf319)
            del primals_168
            assert_size_stride(primals_166, (3072, ), (1, ), 'input')
            assert_size_stride(add_144, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_36, (1, 128, 1), (128, 1, 1), 'input')
            buf325 = buf304; del buf304  # reuse
            # Topologically Sorted Source Nodes: [view_1019, view_1022, add_353, view_1025, add_354, mul_567, hidden_states_180, convert_element_type_1192, mul_569, mul_570, sum_104, pow_98, mul_571, mul_572, expand_193, div_78, pow_99, mul_573, mul_574, add_355, convert_element_type_1193, add_356], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf325, buf316, buf319, buf322, primals_166, add_144, rsqrt_36, 128, 3072, stream=raw_stream0)
            del primals_166
            buf323 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1019, view_1022, add_353, view_1025, add_354, hidden_states_180, hidden_states_181, to_96, mul_568, sum_103], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf316, buf319, buf322, add_144, rsqrt_36, buf323, 3072, 128, stream=raw_stream0)
            del add_144
            del rsqrt_36
            assert_size_stride(view_477, (128, 8192), (8192, 1), 'input')
            buf326 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1027, permute_712, mm_339], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf325, (3072, 128), (1, 3072), 0), view_477, out=buf326)
            del view_477
            assert_size_stride(primals_165, (3072, 8192), (8192, 1), 'input')
            buf327 = reinterpret_tensor(buf299, (128, 8192), (8192, 1), 0); del buf299  # reuse
            # Topologically Sorted Source Nodes: [view_1027, down_proj_17, permute_714, mm_340], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf325, (128, 3072), (3072, 1), 0), primals_165, out=buf327)
            del primals_165
            assert_size_stride(mm_123, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_124, (128, 8192), (8192, 1), 'input')
            buf328 = buf296; del buf296  # reuse
            buf331 = reinterpret_tensor(mm_124, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_124  # reuse
            # Topologically Sorted Source Nodes: [view_1028, linear_123, silu_17, mul_575, linear_124, mul_576, convert_element_type_1202, reciprocal_10, mul_577, mul_578, sub_43, mul_579, add_358, mul_580, convert_element_type_1204], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf331, buf327, mm_123, buf328, 1048576, stream=raw_stream0)
            del buf327
            del mm_123
            assert_size_stride(view_473, (128, 3072), (3072, 1), 'input')
            buf329 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1028, linear_123, silu_17, mul_575, view_1029, permute_716, mm_341], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf328, (8192, 128), (1, 8192), 0), view_473, out=buf329)
            assert_size_stride(primals_164, (8192, 3072), (3072, 1), 'input')
            buf330 = buf322; del buf322  # reuse
            # Topologically Sorted Source Nodes: [view_1028, linear_123, silu_17, mul_575, view_1029, linear_124, permute_718, mm_342], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf328, (128, 8192), (8192, 1), 0), primals_164, out=buf330)
            del primals_164
            buf332 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1028, linear_123, silu_17, linear_124, mul_576, convert_element_type_1202, reciprocal_10, mul_577, mul_578, sub_43, mul_579, add_358, mul_580, convert_element_type_1204, view_1031, permute_720, mm_343], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf331, (8192, 128), (1, 8192), 0), view_473, out=buf332)
            del view_473
            assert_size_stride(primals_163, (8192, 3072), (3072, 1), 'input')
            buf333 = buf319; del buf319  # reuse
            # Topologically Sorted Source Nodes: [view_1028, linear_123, silu_17, linear_124, mul_576, convert_element_type_1202, reciprocal_10, mul_577, mul_578, sub_43, mul_579, add_358, mul_580, convert_element_type_1204, view_1031, permute_722, mm_344], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf331, (128, 8192), (8192, 1), 0), primals_163, out=buf333)
            del primals_163
            assert_size_stride(primals_162, (3072, ), (1, ), 'input')
            assert_size_stride(add_141, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_35, (1, 128, 1), (128, 1, 1), 'input')
            buf336 = buf325; del buf325  # reuse
            # Topologically Sorted Source Nodes: [view_1030, view_1032, add_359, mul_581, hidden_states_176, convert_element_type_1209, mul_583, mul_584, sum_106, pow_100, mul_585, mul_586, expand_194, div_79, pow_101, mul_587, mul_588, add_360, convert_element_type_1210, add_361], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf336, buf330, buf333, primals_162, add_141, rsqrt_35, 128, 3072, stream=raw_stream0)
            del primals_162
            buf334 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1030, view_1032, add_359, hidden_states_176, hidden_states_177, to_94, mul_582, sum_105], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf330, buf333, add_141, rsqrt_35, buf334, 3072, 128, stream=raw_stream0)
            del add_141
            del rsqrt_35
            assert_size_stride(view_471, (128, 3072), (3072, 1), 'input')
            buf337 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1034, permute_724, mm_345], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf336, (3072, 128), (1, 3072), 0), view_471, out=buf337)
            del view_471
            assert_size_stride(primals_161, (3072, 3072), (3072, 1), 'input')
            buf338 = buf333; del buf333  # reuse
            # Topologically Sorted Source Nodes: [view_1034, attn_output_71, permute_726, mm_346], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf336, (128, 3072), (3072, 1), 0), primals_161, out=buf338)
            del primals_161
            assert_size_stride(permute_729, (24, 128, 128), (16384, 1, 128), 'input')
            buf339 = reinterpret_tensor(buf330, (24, 128, 128), (16384, 128, 1), 0); del buf330  # reuse
            # Topologically Sorted Source Nodes: [view_1035, view_1036, permute_728, view_1037, bmm_96], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_729, reinterpret_tensor(buf338, (24, 128, 128), (128, 3072, 1), 0), out=buf339)
            del permute_729
            assert_size_stride(permute_730, (24, 128, 128), (16384, 1, 128), 'input')
            buf340 = reinterpret_tensor(buf316, (24, 128, 128), (16384, 128, 1), 0); del buf316  # reuse
            # Topologically Sorted Source Nodes: [view_1035, view_1036, permute_728, view_1037, bmm_97], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf338, (24, 128, 128), (128, 3072, 1), 0), permute_730, out=buf340)
            del permute_730
            buf346 = reinterpret_tensor(buf317, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf317  # reuse
            # Topologically Sorted Source Nodes: [view_1038, view_1043, sum_108, squeeze_21, permute_734, clone_144], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf339, buf346, 131072, stream=raw_stream0)
            assert_size_stride(add_140, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_17, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_18, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf342 = add_140; del add_140  # reuse
            # Topologically Sorted Source Nodes: [view_1039, convert_element_type_1219, softmax_17, mul_589, sum_107, neg_126, fma_10, convert_element_type_1220, mul_590], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf342, buf340, amax_17, sum_18, 3072, 128, stream=raw_stream0)
            del amax_17
            del sum_18
            assert_size_stride(view_453, (128, 3072), (3072, 1), 'input')
            buf347 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1038, view_1043, sum_108, squeeze_21, permute_734, clone_144, view_1045, view_1046, permute_735, mm_347], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf346, (1024, 128), (1, 1024), 0), view_453, out=buf347)
            assert_size_stride(primals_160, (1024, 3072), (3072, 1), 'input')
            buf348 = reinterpret_tensor(buf340, (128, 3072), (3072, 1), 0); del buf340  # reuse
            # Topologically Sorted Source Nodes: [view_1038, view_1043, sum_108, squeeze_21, permute_734, clone_144, view_1045, view_1046, linear_121, permute_737, mm_348], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf346, (128, 1024), (1024, 1), 0), primals_160, out=buf348)
            del primals_160
            assert_size_stride(permute_731, (24, 128, 128), (128, 1, 3072), 'input')
            buf343 = buf339; del buf339  # reuse
            # Topologically Sorted Source Nodes: [view_1039, convert_element_type_1219, softmax_17, mul_589, neg_126, fma_10, convert_element_type_1220, mul_590, view_1040, bmm_98], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_731, reinterpret_tensor(buf342, (24, 128, 128), (16384, 128, 1), 0), out=buf343)
            del permute_731
            assert_size_stride(permute_732, (24, 128, 128), (16384, 128, 1), 'input')
            buf344 = reinterpret_tensor(buf338, (24, 128, 128), (16384, 128, 1), 0); del buf338  # reuse
            # Topologically Sorted Source Nodes: [view_1039, convert_element_type_1219, softmax_17, mul_589, neg_126, fma_10, convert_element_type_1220, mul_590, view_1040, bmm_99], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf342, (24, 128, 128), (16384, 128, 1), 0), permute_732, out=buf344)
            del buf342
            del permute_732
            buf345 = reinterpret_tensor(buf346, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1041, permute_733, view_1044, sum_109, squeeze_22, mul_591, slice_157, slice_158, neg_127, add_362, mul_592, add_363], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf343, primals_3, buf345, 131072, stream=raw_stream0)
            buf352 = reinterpret_tensor(buf343, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf343  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1042, mul_593, slice_159, slice_160, neg_128, add_364, mul_594, add_365, permute_744, clone_146], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf344, primals_3, buf352, 393216, stream=raw_stream0)
            buf349 = reinterpret_tensor(buf313, (128, 1024), (1024, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [permute_739, clone_145, view_1048, view_1049], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf345, buf349, 128, 1024, stream=raw_stream0)
            buf353 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1042, mul_593, slice_159, slice_160, neg_128, add_364, mul_594, add_365, permute_744, clone_146, view_1051, view_1052, permute_745, mm_351], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf352, (3072, 128), (1, 3072), 0), view_453, out=buf353)
            assert_size_stride(primals_158, (3072, 3072), (3072, 1), 'input')
            buf354 = reinterpret_tensor(buf344, (128, 3072), (3072, 1), 0); del buf344  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1042, mul_593, slice_159, slice_160, neg_128, add_364, mul_594, add_365, permute_744, clone_146, view_1051, view_1052, linear_119, permute_747, mm_352], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf352, (128, 3072), (3072, 1), 0), primals_158, out=buf354)
            del primals_158
            buf350 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_740, mm_349], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf349, (1024, 128), (1, 1024), 0), view_453, out=buf350)
            del view_453
            assert_size_stride(primals_159, (1024, 3072), (3072, 1), 'input')
            buf351 = reinterpret_tensor(buf352, (128, 3072), (3072, 1), 0); del buf352  # reuse
            # Topologically Sorted Source Nodes: [linear_120, permute_742, mm_350], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf349, primals_159, out=buf351)
            del primals_159
            assert_size_stride(primals_157, (3072, ), (1, ), 'input')
            assert_size_stride(add_136, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_34, (1, 128, 1), (128, 1, 1), 'input')
            buf357 = buf336; del buf336  # reuse
            # Topologically Sorted Source Nodes: [view_1047, view_1050, add_366, view_1053, add_367, mul_595, hidden_states_170, convert_element_type_1237, mul_597, mul_598, sum_111, pow_102, mul_599, mul_600, expand_195, div_80, pow_103, mul_601, mul_602, add_368, convert_element_type_1238, add_369], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf357, buf348, buf351, buf354, primals_157, add_136, rsqrt_34, 128, 3072, stream=raw_stream0)
            del primals_157
            buf355 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1047, view_1050, add_366, view_1053, add_367, hidden_states_170, hidden_states_171, to_91, mul_596, sum_110], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf348, buf351, buf354, add_136, rsqrt_34, buf355, 3072, 128, stream=raw_stream0)
            del add_136
            del rsqrt_34
            assert_size_stride(view_451, (128, 8192), (8192, 1), 'input')
            buf358 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1055, permute_749, mm_353], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf357, (3072, 128), (1, 3072), 0), view_451, out=buf358)
            del view_451
            assert_size_stride(primals_156, (3072, 8192), (8192, 1), 'input')
            buf359 = reinterpret_tensor(buf331, (128, 8192), (8192, 1), 0); del buf331  # reuse
            # Topologically Sorted Source Nodes: [view_1055, down_proj_16, permute_751, mm_354], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf357, (128, 3072), (3072, 1), 0), primals_156, out=buf359)
            del primals_156
            assert_size_stride(mm_116, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_117, (128, 8192), (8192, 1), 'input')
            buf360 = buf328; del buf328  # reuse
            buf363 = reinterpret_tensor(mm_117, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_117  # reuse
            # Topologically Sorted Source Nodes: [view_1056, linear_116, silu_16, mul_603, linear_117, mul_604, convert_element_type_1247, reciprocal_11, mul_605, mul_606, sub_44, mul_607, add_371, mul_608, convert_element_type_1249], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf363, buf359, mm_116, buf360, 1048576, stream=raw_stream0)
            del buf359
            del mm_116
            assert_size_stride(view_447, (128, 3072), (3072, 1), 'input')
            buf361 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1056, linear_116, silu_16, mul_603, view_1057, permute_753, mm_355], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf360, (8192, 128), (1, 8192), 0), view_447, out=buf361)
            assert_size_stride(primals_155, (8192, 3072), (3072, 1), 'input')
            buf362 = buf354; del buf354  # reuse
            # Topologically Sorted Source Nodes: [view_1056, linear_116, silu_16, mul_603, view_1057, linear_117, permute_755, mm_356], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf360, (128, 8192), (8192, 1), 0), primals_155, out=buf362)
            del primals_155
            buf364 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1056, linear_116, silu_16, linear_117, mul_604, convert_element_type_1247, reciprocal_11, mul_605, mul_606, sub_44, mul_607, add_371, mul_608, convert_element_type_1249, view_1059, permute_757, mm_357], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf363, (8192, 128), (1, 8192), 0), view_447, out=buf364)
            del view_447
            assert_size_stride(primals_154, (8192, 3072), (3072, 1), 'input')
            buf365 = buf351; del buf351  # reuse
            # Topologically Sorted Source Nodes: [view_1056, linear_116, silu_16, linear_117, mul_604, convert_element_type_1247, reciprocal_11, mul_605, mul_606, sub_44, mul_607, add_371, mul_608, convert_element_type_1249, view_1059, permute_759, mm_358], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf363, (128, 8192), (8192, 1), 0), primals_154, out=buf365)
            del primals_154
            assert_size_stride(primals_153, (3072, ), (1, ), 'input')
            assert_size_stride(add_133, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_33, (1, 128, 1), (128, 1, 1), 'input')
            buf368 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [view_1058, view_1060, add_372, mul_609, hidden_states_166, convert_element_type_1254, mul_611, mul_612, sum_113, pow_104, mul_613, mul_614, expand_196, div_81, pow_105, mul_615, mul_616, add_373, convert_element_type_1255, add_374], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf368, buf362, buf365, primals_153, add_133, rsqrt_33, 128, 3072, stream=raw_stream0)
            del primals_153
            buf366 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1058, view_1060, add_372, hidden_states_166, hidden_states_167, to_89, mul_610, sum_112], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf362, buf365, add_133, rsqrt_33, buf366, 3072, 128, stream=raw_stream0)
            del add_133
            del rsqrt_33
            assert_size_stride(view_445, (128, 3072), (3072, 1), 'input')
            buf369 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1062, permute_761, mm_359], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf368, (3072, 128), (1, 3072), 0), view_445, out=buf369)
            del view_445
            assert_size_stride(primals_152, (3072, 3072), (3072, 1), 'input')
            buf370 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [view_1062, attn_output_67, permute_763, mm_360], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf368, (128, 3072), (3072, 1), 0), primals_152, out=buf370)
            del primals_152
            assert_size_stride(permute_766, (24, 128, 128), (16384, 1, 128), 'input')
            buf371 = reinterpret_tensor(buf362, (24, 128, 128), (16384, 128, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [view_1063, view_1064, permute_765, view_1065, bmm_100], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_766, reinterpret_tensor(buf370, (24, 128, 128), (128, 3072, 1), 0), out=buf371)
            del permute_766
            assert_size_stride(permute_767, (24, 128, 128), (16384, 1, 128), 'input')
            buf372 = reinterpret_tensor(buf348, (24, 128, 128), (16384, 128, 1), 0); del buf348  # reuse
            # Topologically Sorted Source Nodes: [view_1063, view_1064, permute_765, view_1065, bmm_101], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf370, (24, 128, 128), (128, 3072, 1), 0), permute_767, out=buf372)
            del permute_767
            buf378 = reinterpret_tensor(buf349, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [view_1066, view_1071, sum_115, squeeze_23, permute_771, clone_147], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf371, buf378, 131072, stream=raw_stream0)
            assert_size_stride(add_132, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_16, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_17, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf374 = add_132; del add_132  # reuse
            # Topologically Sorted Source Nodes: [view_1067, convert_element_type_1264, softmax_16, mul_617, sum_114, neg_130, fma_11, convert_element_type_1265, mul_618], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf374, buf372, amax_16, sum_17, 3072, 128, stream=raw_stream0)
            del amax_16
            del sum_17
            assert_size_stride(view_427, (128, 3072), (3072, 1), 'input')
            buf379 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1066, view_1071, sum_115, squeeze_23, permute_771, clone_147, view_1073, view_1074, permute_772, mm_361], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf378, (1024, 128), (1, 1024), 0), view_427, out=buf379)
            assert_size_stride(primals_151, (1024, 3072), (3072, 1), 'input')
            buf380 = reinterpret_tensor(buf372, (128, 3072), (3072, 1), 0); del buf372  # reuse
            # Topologically Sorted Source Nodes: [view_1066, view_1071, sum_115, squeeze_23, permute_771, clone_147, view_1073, view_1074, linear_114, permute_774, mm_362], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf378, (128, 1024), (1024, 1), 0), primals_151, out=buf380)
            del primals_151
            assert_size_stride(permute_768, (24, 128, 128), (128, 1, 3072), 'input')
            buf375 = buf371; del buf371  # reuse
            # Topologically Sorted Source Nodes: [view_1067, convert_element_type_1264, softmax_16, mul_617, neg_130, fma_11, convert_element_type_1265, mul_618, view_1068, bmm_102], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_768, reinterpret_tensor(buf374, (24, 128, 128), (16384, 128, 1), 0), out=buf375)
            del permute_768
            assert_size_stride(permute_769, (24, 128, 128), (16384, 128, 1), 'input')
            buf376 = reinterpret_tensor(buf370, (24, 128, 128), (16384, 128, 1), 0); del buf370  # reuse
            # Topologically Sorted Source Nodes: [view_1067, convert_element_type_1264, softmax_16, mul_617, neg_130, fma_11, convert_element_type_1265, mul_618, view_1068, bmm_103], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf374, (24, 128, 128), (16384, 128, 1), 0), permute_769, out=buf376)
            del buf374
            del permute_769
            buf377 = reinterpret_tensor(buf378, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1069, permute_770, view_1072, sum_116, squeeze_24, mul_619, slice_161, slice_162, neg_131, add_375, mul_620, add_376], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf375, primals_3, buf377, 131072, stream=raw_stream0)
            buf384 = reinterpret_tensor(buf375, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf375  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1070, mul_621, slice_163, slice_164, neg_132, add_377, mul_622, add_378, permute_781, clone_149], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf376, primals_3, buf384, 393216, stream=raw_stream0)
            buf381 = reinterpret_tensor(buf345, (128, 1024), (1024, 1), 0); del buf345  # reuse
            # Topologically Sorted Source Nodes: [permute_776, clone_148, view_1076, view_1077], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf377, buf381, 128, 1024, stream=raw_stream0)
            buf385 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1070, mul_621, slice_163, slice_164, neg_132, add_377, mul_622, add_378, permute_781, clone_149, view_1079, view_1080, permute_782, mm_365], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf384, (3072, 128), (1, 3072), 0), view_427, out=buf385)
            assert_size_stride(primals_149, (3072, 3072), (3072, 1), 'input')
            buf386 = reinterpret_tensor(buf376, (128, 3072), (3072, 1), 0); del buf376  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1070, mul_621, slice_163, slice_164, neg_132, add_377, mul_622, add_378, permute_781, clone_149, view_1079, view_1080, linear_112, permute_784, mm_366], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf384, (128, 3072), (3072, 1), 0), primals_149, out=buf386)
            del primals_149
            buf382 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_777, mm_363], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf381, (1024, 128), (1, 1024), 0), view_427, out=buf382)
            del view_427
            assert_size_stride(primals_150, (1024, 3072), (3072, 1), 'input')
            buf383 = reinterpret_tensor(buf384, (128, 3072), (3072, 1), 0); del buf384  # reuse
            # Topologically Sorted Source Nodes: [linear_113, permute_779, mm_364], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf381, primals_150, out=buf383)
            del primals_150
            assert_size_stride(primals_148, (3072, ), (1, ), 'input')
            assert_size_stride(add_128, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_32, (1, 128, 1), (128, 1, 1), 'input')
            buf389 = buf368; del buf368  # reuse
            # Topologically Sorted Source Nodes: [view_1075, view_1078, add_379, view_1081, add_380, mul_623, hidden_states_160, convert_element_type_1282, mul_625, mul_626, sum_118, pow_106, mul_627, mul_628, expand_197, div_82, pow_107, mul_629, mul_630, add_381, convert_element_type_1283, add_382], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf389, buf380, buf383, buf386, primals_148, add_128, rsqrt_32, 128, 3072, stream=raw_stream0)
            del primals_148
            buf387 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1075, view_1078, add_379, view_1081, add_380, hidden_states_160, hidden_states_161, to_86, mul_624, sum_117], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf380, buf383, buf386, add_128, rsqrt_32, buf387, 3072, 128, stream=raw_stream0)
            del add_128
            del rsqrt_32
            assert_size_stride(view_425, (128, 8192), (8192, 1), 'input')
            buf390 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1083, permute_786, mm_367], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf389, (3072, 128), (1, 3072), 0), view_425, out=buf390)
            del view_425
            assert_size_stride(primals_147, (3072, 8192), (8192, 1), 'input')
            buf391 = reinterpret_tensor(buf363, (128, 8192), (8192, 1), 0); del buf363  # reuse
            # Topologically Sorted Source Nodes: [view_1083, down_proj_15, permute_788, mm_368], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf389, (128, 3072), (3072, 1), 0), primals_147, out=buf391)
            del primals_147
            assert_size_stride(mm_109, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_110, (128, 8192), (8192, 1), 'input')
            buf392 = buf360; del buf360  # reuse
            buf395 = reinterpret_tensor(mm_110, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_110  # reuse
            # Topologically Sorted Source Nodes: [view_1084, linear_109, silu_15, mul_631, linear_110, mul_632, convert_element_type_1292, reciprocal_12, mul_633, mul_634, sub_45, mul_635, add_384, mul_636, convert_element_type_1294], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf395, buf391, mm_109, buf392, 1048576, stream=raw_stream0)
            del buf391
            del mm_109
            assert_size_stride(view_421, (128, 3072), (3072, 1), 'input')
            buf393 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1084, linear_109, silu_15, mul_631, view_1085, permute_790, mm_369], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf392, (8192, 128), (1, 8192), 0), view_421, out=buf393)
            assert_size_stride(primals_146, (8192, 3072), (3072, 1), 'input')
            buf394 = buf386; del buf386  # reuse
            # Topologically Sorted Source Nodes: [view_1084, linear_109, silu_15, mul_631, view_1085, linear_110, permute_792, mm_370], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf392, (128, 8192), (8192, 1), 0), primals_146, out=buf394)
            del primals_146
            buf396 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1084, linear_109, silu_15, linear_110, mul_632, convert_element_type_1292, reciprocal_12, mul_633, mul_634, sub_45, mul_635, add_384, mul_636, convert_element_type_1294, view_1087, permute_794, mm_371], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf395, (8192, 128), (1, 8192), 0), view_421, out=buf396)
            del view_421
            assert_size_stride(primals_145, (8192, 3072), (3072, 1), 'input')
            buf397 = buf383; del buf383  # reuse
            # Topologically Sorted Source Nodes: [view_1084, linear_109, silu_15, linear_110, mul_632, convert_element_type_1292, reciprocal_12, mul_633, mul_634, sub_45, mul_635, add_384, mul_636, convert_element_type_1294, view_1087, permute_796, mm_372], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf395, (128, 8192), (8192, 1), 0), primals_145, out=buf397)
            del primals_145
            assert_size_stride(primals_144, (3072, ), (1, ), 'input')
            assert_size_stride(add_125, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_31, (1, 128, 1), (128, 1, 1), 'input')
            buf400 = buf389; del buf389  # reuse
            # Topologically Sorted Source Nodes: [view_1086, view_1088, add_385, mul_637, hidden_states_156, convert_element_type_1299, mul_639, mul_640, sum_120, pow_108, mul_641, mul_642, expand_198, div_83, pow_109, mul_643, mul_644, add_386, convert_element_type_1300, add_387], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf400, buf394, buf397, primals_144, add_125, rsqrt_31, 128, 3072, stream=raw_stream0)
            del primals_144
            buf398 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1086, view_1088, add_385, hidden_states_156, hidden_states_157, to_84, mul_638, sum_119], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf394, buf397, add_125, rsqrt_31, buf398, 3072, 128, stream=raw_stream0)
            del add_125
            del rsqrt_31
            assert_size_stride(view_419, (128, 3072), (3072, 1), 'input')
            buf401 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1090, permute_798, mm_373], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf400, (3072, 128), (1, 3072), 0), view_419, out=buf401)
            del view_419
            assert_size_stride(primals_143, (3072, 3072), (3072, 1), 'input')
            buf402 = buf397; del buf397  # reuse
            # Topologically Sorted Source Nodes: [view_1090, attn_output_63, permute_800, mm_374], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf400, (128, 3072), (3072, 1), 0), primals_143, out=buf402)
            del primals_143
            assert_size_stride(permute_803, (24, 128, 128), (16384, 1, 128), 'input')
            buf403 = reinterpret_tensor(buf394, (24, 128, 128), (16384, 128, 1), 0); del buf394  # reuse
            # Topologically Sorted Source Nodes: [view_1091, view_1092, permute_802, view_1093, bmm_104], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_803, reinterpret_tensor(buf402, (24, 128, 128), (128, 3072, 1), 0), out=buf403)
            del permute_803
            assert_size_stride(permute_804, (24, 128, 128), (16384, 1, 128), 'input')
            buf404 = reinterpret_tensor(buf380, (24, 128, 128), (16384, 128, 1), 0); del buf380  # reuse
            # Topologically Sorted Source Nodes: [view_1091, view_1092, permute_802, view_1093, bmm_105], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf402, (24, 128, 128), (128, 3072, 1), 0), permute_804, out=buf404)
            del permute_804
            buf410 = reinterpret_tensor(buf381, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf381  # reuse
            # Topologically Sorted Source Nodes: [view_1094, view_1099, sum_122, squeeze_25, permute_808, clone_150], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf403, buf410, 131072, stream=raw_stream0)
            assert_size_stride(add_124, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_15, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_16, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf406 = add_124; del add_124  # reuse
            # Topologically Sorted Source Nodes: [view_1095, convert_element_type_1309, softmax_15, mul_645, sum_121, neg_134, fma_12, convert_element_type_1310, mul_646], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf406, buf404, amax_15, sum_16, 3072, 128, stream=raw_stream0)
            del amax_15
            del sum_16
            assert_size_stride(view_401, (128, 3072), (3072, 1), 'input')
            buf411 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1094, view_1099, sum_122, squeeze_25, permute_808, clone_150, view_1101, view_1102, permute_809, mm_375], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf410, (1024, 128), (1, 1024), 0), view_401, out=buf411)
            assert_size_stride(primals_142, (1024, 3072), (3072, 1), 'input')
            buf412 = reinterpret_tensor(buf404, (128, 3072), (3072, 1), 0); del buf404  # reuse
            # Topologically Sorted Source Nodes: [view_1094, view_1099, sum_122, squeeze_25, permute_808, clone_150, view_1101, view_1102, linear_107, permute_811, mm_376], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf410, (128, 1024), (1024, 1), 0), primals_142, out=buf412)
            del primals_142
            assert_size_stride(permute_805, (24, 128, 128), (128, 1, 3072), 'input')
            buf407 = buf403; del buf403  # reuse
            # Topologically Sorted Source Nodes: [view_1095, convert_element_type_1309, softmax_15, mul_645, neg_134, fma_12, convert_element_type_1310, mul_646, view_1096, bmm_106], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_805, reinterpret_tensor(buf406, (24, 128, 128), (16384, 128, 1), 0), out=buf407)
            del permute_805
            assert_size_stride(permute_806, (24, 128, 128), (16384, 128, 1), 'input')
            buf408 = reinterpret_tensor(buf402, (24, 128, 128), (16384, 128, 1), 0); del buf402  # reuse
            # Topologically Sorted Source Nodes: [view_1095, convert_element_type_1309, softmax_15, mul_645, neg_134, fma_12, convert_element_type_1310, mul_646, view_1096, bmm_107], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf406, (24, 128, 128), (16384, 128, 1), 0), permute_806, out=buf408)
            del buf406
            del permute_806
            buf409 = reinterpret_tensor(buf410, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf410  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1097, permute_807, view_1100, sum_123, squeeze_26, mul_647, slice_165, slice_166, neg_135, add_388, mul_648, add_389], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf407, primals_3, buf409, 131072, stream=raw_stream0)
            buf416 = reinterpret_tensor(buf407, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf407  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1098, mul_649, slice_167, slice_168, neg_136, add_390, mul_650, add_391, permute_818, clone_152], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf408, primals_3, buf416, 393216, stream=raw_stream0)
            buf413 = reinterpret_tensor(buf377, (128, 1024), (1024, 1), 0); del buf377  # reuse
            # Topologically Sorted Source Nodes: [permute_813, clone_151, view_1104, view_1105], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf409, buf413, 128, 1024, stream=raw_stream0)
            buf417 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1098, mul_649, slice_167, slice_168, neg_136, add_390, mul_650, add_391, permute_818, clone_152, view_1107, view_1108, permute_819, mm_379], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf416, (3072, 128), (1, 3072), 0), view_401, out=buf417)
            assert_size_stride(primals_140, (3072, 3072), (3072, 1), 'input')
            buf418 = reinterpret_tensor(buf408, (128, 3072), (3072, 1), 0); del buf408  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1098, mul_649, slice_167, slice_168, neg_136, add_390, mul_650, add_391, permute_818, clone_152, view_1107, view_1108, linear_105, permute_821, mm_380], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf416, (128, 3072), (3072, 1), 0), primals_140, out=buf418)
            del primals_140
            buf414 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_814, mm_377], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf413, (1024, 128), (1, 1024), 0), view_401, out=buf414)
            del view_401
            assert_size_stride(primals_141, (1024, 3072), (3072, 1), 'input')
            buf415 = reinterpret_tensor(buf416, (128, 3072), (3072, 1), 0); del buf416  # reuse
            # Topologically Sorted Source Nodes: [linear_106, permute_816, mm_378], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf413, primals_141, out=buf415)
            del primals_141
            assert_size_stride(primals_139, (3072, ), (1, ), 'input')
            assert_size_stride(add_120, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_30, (1, 128, 1), (128, 1, 1), 'input')
            buf421 = buf400; del buf400  # reuse
            # Topologically Sorted Source Nodes: [view_1103, view_1106, add_392, view_1109, add_393, mul_651, hidden_states_150, convert_element_type_1327, mul_653, mul_654, sum_125, pow_110, mul_655, mul_656, expand_199, div_84, pow_111, mul_657, mul_658, add_394, convert_element_type_1328, add_395], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf421, buf412, buf415, buf418, primals_139, add_120, rsqrt_30, 128, 3072, stream=raw_stream0)
            del primals_139
            buf419 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1103, view_1106, add_392, view_1109, add_393, hidden_states_150, hidden_states_151, to_81, mul_652, sum_124], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf412, buf415, buf418, add_120, rsqrt_30, buf419, 3072, 128, stream=raw_stream0)
            del add_120
            del rsqrt_30
            assert_size_stride(view_399, (128, 8192), (8192, 1), 'input')
            buf422 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1111, permute_823, mm_381], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf421, (3072, 128), (1, 3072), 0), view_399, out=buf422)
            del view_399
            assert_size_stride(primals_138, (3072, 8192), (8192, 1), 'input')
            buf423 = reinterpret_tensor(buf395, (128, 8192), (8192, 1), 0); del buf395  # reuse
            # Topologically Sorted Source Nodes: [view_1111, down_proj_14, permute_825, mm_382], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf421, (128, 3072), (3072, 1), 0), primals_138, out=buf423)
            del primals_138
            assert_size_stride(mm_102, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_103, (128, 8192), (8192, 1), 'input')
            buf424 = buf392; del buf392  # reuse
            buf427 = reinterpret_tensor(mm_103, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_103  # reuse
            # Topologically Sorted Source Nodes: [view_1112, linear_102, silu_14, mul_659, linear_103, mul_660, convert_element_type_1337, reciprocal_13, mul_661, mul_662, sub_46, mul_663, add_397, mul_664, convert_element_type_1339], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf427, buf423, mm_102, buf424, 1048576, stream=raw_stream0)
            del buf423
            del mm_102
            assert_size_stride(view_395, (128, 3072), (3072, 1), 'input')
            buf425 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1112, linear_102, silu_14, mul_659, view_1113, permute_827, mm_383], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf424, (8192, 128), (1, 8192), 0), view_395, out=buf425)
            assert_size_stride(primals_137, (8192, 3072), (3072, 1), 'input')
            buf426 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [view_1112, linear_102, silu_14, mul_659, view_1113, linear_103, permute_829, mm_384], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf424, (128, 8192), (8192, 1), 0), primals_137, out=buf426)
            del primals_137
            buf428 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1112, linear_102, silu_14, linear_103, mul_660, convert_element_type_1337, reciprocal_13, mul_661, mul_662, sub_46, mul_663, add_397, mul_664, convert_element_type_1339, view_1115, permute_831, mm_385], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf427, (8192, 128), (1, 8192), 0), view_395, out=buf428)
            del view_395
            assert_size_stride(primals_136, (8192, 3072), (3072, 1), 'input')
            buf429 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [view_1112, linear_102, silu_14, linear_103, mul_660, convert_element_type_1337, reciprocal_13, mul_661, mul_662, sub_46, mul_663, add_397, mul_664, convert_element_type_1339, view_1115, permute_833, mm_386], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf427, (128, 8192), (8192, 1), 0), primals_136, out=buf429)
            del primals_136
            assert_size_stride(primals_135, (3072, ), (1, ), 'input')
            assert_size_stride(add_117, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_29, (1, 128, 1), (128, 1, 1), 'input')
            buf432 = buf421; del buf421  # reuse
            # Topologically Sorted Source Nodes: [view_1114, view_1116, add_398, mul_665, hidden_states_146, convert_element_type_1344, mul_667, mul_668, sum_127, pow_112, mul_669, mul_670, expand_200, div_85, pow_113, mul_671, mul_672, add_399, convert_element_type_1345, add_400], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf432, buf426, buf429, primals_135, add_117, rsqrt_29, 128, 3072, stream=raw_stream0)
            del primals_135
            buf430 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1114, view_1116, add_398, hidden_states_146, hidden_states_147, to_79, mul_666, sum_126], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf426, buf429, add_117, rsqrt_29, buf430, 3072, 128, stream=raw_stream0)
            del add_117
            del rsqrt_29
            assert_size_stride(view_393, (128, 3072), (3072, 1), 'input')
            buf433 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1118, permute_835, mm_387], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf432, (3072, 128), (1, 3072), 0), view_393, out=buf433)
            del view_393
            assert_size_stride(primals_134, (3072, 3072), (3072, 1), 'input')
            buf434 = buf429; del buf429  # reuse
            # Topologically Sorted Source Nodes: [view_1118, attn_output_59, permute_837, mm_388], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf432, (128, 3072), (3072, 1), 0), primals_134, out=buf434)
            del primals_134
            assert_size_stride(permute_840, (24, 128, 128), (16384, 1, 128), 'input')
            buf435 = reinterpret_tensor(buf426, (24, 128, 128), (16384, 128, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [view_1119, view_1120, permute_839, view_1121, bmm_108], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_840, reinterpret_tensor(buf434, (24, 128, 128), (128, 3072, 1), 0), out=buf435)
            del permute_840
            assert_size_stride(permute_841, (24, 128, 128), (16384, 1, 128), 'input')
            buf436 = reinterpret_tensor(buf412, (24, 128, 128), (16384, 128, 1), 0); del buf412  # reuse
            # Topologically Sorted Source Nodes: [view_1119, view_1120, permute_839, view_1121, bmm_109], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf434, (24, 128, 128), (128, 3072, 1), 0), permute_841, out=buf436)
            del permute_841
            buf442 = reinterpret_tensor(buf413, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf413  # reuse
            # Topologically Sorted Source Nodes: [view_1122, view_1127, sum_129, squeeze_27, permute_845, clone_153], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf435, buf442, 131072, stream=raw_stream0)
            assert_size_stride(add_116, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_14, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_15, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf438 = add_116; del add_116  # reuse
            # Topologically Sorted Source Nodes: [view_1123, convert_element_type_1354, softmax_14, mul_673, sum_128, neg_138, fma_13, convert_element_type_1355, mul_674], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf438, buf436, amax_14, sum_15, 3072, 128, stream=raw_stream0)
            del amax_14
            del sum_15
            assert_size_stride(view_375, (128, 3072), (3072, 1), 'input')
            buf443 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1122, view_1127, sum_129, squeeze_27, permute_845, clone_153, view_1129, view_1130, permute_846, mm_389], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf442, (1024, 128), (1, 1024), 0), view_375, out=buf443)
            assert_size_stride(primals_133, (1024, 3072), (3072, 1), 'input')
            buf444 = reinterpret_tensor(buf436, (128, 3072), (3072, 1), 0); del buf436  # reuse
            # Topologically Sorted Source Nodes: [view_1122, view_1127, sum_129, squeeze_27, permute_845, clone_153, view_1129, view_1130, linear_100, permute_848, mm_390], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf442, (128, 1024), (1024, 1), 0), primals_133, out=buf444)
            del primals_133
            assert_size_stride(permute_842, (24, 128, 128), (128, 1, 3072), 'input')
            buf439 = buf435; del buf435  # reuse
            # Topologically Sorted Source Nodes: [view_1123, convert_element_type_1354, softmax_14, mul_673, neg_138, fma_13, convert_element_type_1355, mul_674, view_1124, bmm_110], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_842, reinterpret_tensor(buf438, (24, 128, 128), (16384, 128, 1), 0), out=buf439)
            del permute_842
            assert_size_stride(permute_843, (24, 128, 128), (16384, 128, 1), 'input')
            buf440 = reinterpret_tensor(buf434, (24, 128, 128), (16384, 128, 1), 0); del buf434  # reuse
            # Topologically Sorted Source Nodes: [view_1123, convert_element_type_1354, softmax_14, mul_673, neg_138, fma_13, convert_element_type_1355, mul_674, view_1124, bmm_111], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf438, (24, 128, 128), (16384, 128, 1), 0), permute_843, out=buf440)
            del buf438
            del permute_843
            buf441 = reinterpret_tensor(buf442, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf442  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1125, permute_844, view_1128, sum_130, squeeze_28, mul_675, slice_169, slice_170, neg_139, add_401, mul_676, add_402], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf439, primals_3, buf441, 131072, stream=raw_stream0)
            buf448 = reinterpret_tensor(buf439, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf439  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1126, mul_677, slice_171, slice_172, neg_140, add_403, mul_678, add_404, permute_855, clone_155], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf440, primals_3, buf448, 393216, stream=raw_stream0)
            buf445 = reinterpret_tensor(buf409, (128, 1024), (1024, 1), 0); del buf409  # reuse
            # Topologically Sorted Source Nodes: [permute_850, clone_154, view_1132, view_1133], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf441, buf445, 128, 1024, stream=raw_stream0)
            buf449 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1126, mul_677, slice_171, slice_172, neg_140, add_403, mul_678, add_404, permute_855, clone_155, view_1135, view_1136, permute_856, mm_393], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf448, (3072, 128), (1, 3072), 0), view_375, out=buf449)
            assert_size_stride(primals_131, (3072, 3072), (3072, 1), 'input')
            buf450 = reinterpret_tensor(buf440, (128, 3072), (3072, 1), 0); del buf440  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1126, mul_677, slice_171, slice_172, neg_140, add_403, mul_678, add_404, permute_855, clone_155, view_1135, view_1136, linear_98, permute_858, mm_394], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf448, (128, 3072), (3072, 1), 0), primals_131, out=buf450)
            del primals_131
            buf446 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_851, mm_391], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf445, (1024, 128), (1, 1024), 0), view_375, out=buf446)
            del view_375
            assert_size_stride(primals_132, (1024, 3072), (3072, 1), 'input')
            buf447 = reinterpret_tensor(buf448, (128, 3072), (3072, 1), 0); del buf448  # reuse
            # Topologically Sorted Source Nodes: [linear_99, permute_853, mm_392], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf445, primals_132, out=buf447)
            del primals_132
            assert_size_stride(primals_130, (3072, ), (1, ), 'input')
            assert_size_stride(add_112, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_28, (1, 128, 1), (128, 1, 1), 'input')
            buf453 = buf432; del buf432  # reuse
            # Topologically Sorted Source Nodes: [view_1131, view_1134, add_405, view_1137, add_406, mul_679, hidden_states_140, convert_element_type_1372, mul_681, mul_682, sum_132, pow_114, mul_683, mul_684, expand_201, div_86, pow_115, mul_685, mul_686, add_407, convert_element_type_1373, add_408], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf453, buf444, buf447, buf450, primals_130, add_112, rsqrt_28, 128, 3072, stream=raw_stream0)
            del primals_130
            buf451 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1131, view_1134, add_405, view_1137, add_406, hidden_states_140, hidden_states_141, to_76, mul_680, sum_131], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf444, buf447, buf450, add_112, rsqrt_28, buf451, 3072, 128, stream=raw_stream0)
            del add_112
            del rsqrt_28
            assert_size_stride(view_373, (128, 8192), (8192, 1), 'input')
            buf454 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1139, permute_860, mm_395], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf453, (3072, 128), (1, 3072), 0), view_373, out=buf454)
            del view_373
            assert_size_stride(primals_129, (3072, 8192), (8192, 1), 'input')
            buf455 = reinterpret_tensor(buf427, (128, 8192), (8192, 1), 0); del buf427  # reuse
            # Topologically Sorted Source Nodes: [view_1139, down_proj_13, permute_862, mm_396], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf453, (128, 3072), (3072, 1), 0), primals_129, out=buf455)
            del primals_129
            assert_size_stride(mm_95, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_96, (128, 8192), (8192, 1), 'input')
            buf456 = buf424; del buf424  # reuse
            buf459 = reinterpret_tensor(mm_96, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_96  # reuse
            # Topologically Sorted Source Nodes: [view_1140, linear_95, silu_13, mul_687, linear_96, mul_688, convert_element_type_1382, reciprocal_14, mul_689, mul_690, sub_47, mul_691, add_410, mul_692, convert_element_type_1384], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf459, buf455, mm_95, buf456, 1048576, stream=raw_stream0)
            del buf455
            del mm_95
            assert_size_stride(view_369, (128, 3072), (3072, 1), 'input')
            buf457 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1140, linear_95, silu_13, mul_687, view_1141, permute_864, mm_397], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf456, (8192, 128), (1, 8192), 0), view_369, out=buf457)
            assert_size_stride(primals_128, (8192, 3072), (3072, 1), 'input')
            buf458 = buf450; del buf450  # reuse
            # Topologically Sorted Source Nodes: [view_1140, linear_95, silu_13, mul_687, view_1141, linear_96, permute_866, mm_398], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf456, (128, 8192), (8192, 1), 0), primals_128, out=buf458)
            del primals_128
            buf460 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1140, linear_95, silu_13, linear_96, mul_688, convert_element_type_1382, reciprocal_14, mul_689, mul_690, sub_47, mul_691, add_410, mul_692, convert_element_type_1384, view_1143, permute_868, mm_399], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf459, (8192, 128), (1, 8192), 0), view_369, out=buf460)
            del view_369
            assert_size_stride(primals_127, (8192, 3072), (3072, 1), 'input')
            buf461 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [view_1140, linear_95, silu_13, linear_96, mul_688, convert_element_type_1382, reciprocal_14, mul_689, mul_690, sub_47, mul_691, add_410, mul_692, convert_element_type_1384, view_1143, permute_870, mm_400], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf459, (128, 8192), (8192, 1), 0), primals_127, out=buf461)
            del primals_127
            assert_size_stride(primals_126, (3072, ), (1, ), 'input')
            assert_size_stride(add_109, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_27, (1, 128, 1), (128, 1, 1), 'input')
            buf464 = buf453; del buf453  # reuse
            # Topologically Sorted Source Nodes: [view_1142, view_1144, add_411, mul_693, hidden_states_136, convert_element_type_1389, mul_695, mul_696, sum_134, pow_116, mul_697, mul_698, expand_202, div_87, pow_117, mul_699, mul_700, add_412, convert_element_type_1390, add_413], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf464, buf458, buf461, primals_126, add_109, rsqrt_27, 128, 3072, stream=raw_stream0)
            del primals_126
            buf462 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1142, view_1144, add_411, hidden_states_136, hidden_states_137, to_74, mul_694, sum_133], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf458, buf461, add_109, rsqrt_27, buf462, 3072, 128, stream=raw_stream0)
            del add_109
            del rsqrt_27
            assert_size_stride(view_367, (128, 3072), (3072, 1), 'input')
            buf465 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1146, permute_872, mm_401], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf464, (3072, 128), (1, 3072), 0), view_367, out=buf465)
            del view_367
            assert_size_stride(primals_125, (3072, 3072), (3072, 1), 'input')
            buf466 = buf461; del buf461  # reuse
            # Topologically Sorted Source Nodes: [view_1146, attn_output_55, permute_874, mm_402], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf464, (128, 3072), (3072, 1), 0), primals_125, out=buf466)
            del primals_125
            assert_size_stride(permute_877, (24, 128, 128), (16384, 1, 128), 'input')
            buf467 = reinterpret_tensor(buf458, (24, 128, 128), (16384, 128, 1), 0); del buf458  # reuse
            # Topologically Sorted Source Nodes: [view_1147, view_1148, permute_876, view_1149, bmm_112], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_877, reinterpret_tensor(buf466, (24, 128, 128), (128, 3072, 1), 0), out=buf467)
            del permute_877
            assert_size_stride(permute_878, (24, 128, 128), (16384, 1, 128), 'input')
            buf468 = reinterpret_tensor(buf444, (24, 128, 128), (16384, 128, 1), 0); del buf444  # reuse
            # Topologically Sorted Source Nodes: [view_1147, view_1148, permute_876, view_1149, bmm_113], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf466, (24, 128, 128), (128, 3072, 1), 0), permute_878, out=buf468)
            del permute_878
            buf474 = reinterpret_tensor(buf445, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf445  # reuse
            # Topologically Sorted Source Nodes: [view_1150, view_1155, sum_136, squeeze_29, permute_882, clone_156], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf467, buf474, 131072, stream=raw_stream0)
            assert_size_stride(add_108, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_13, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_14, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf470 = add_108; del add_108  # reuse
            # Topologically Sorted Source Nodes: [view_1151, convert_element_type_1399, softmax_13, mul_701, sum_135, neg_142, fma_14, convert_element_type_1400, mul_702], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf470, buf468, amax_13, sum_14, 3072, 128, stream=raw_stream0)
            del amax_13
            del sum_14
            assert_size_stride(view_349, (128, 3072), (3072, 1), 'input')
            buf475 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1150, view_1155, sum_136, squeeze_29, permute_882, clone_156, view_1157, view_1158, permute_883, mm_403], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf474, (1024, 128), (1, 1024), 0), view_349, out=buf475)
            assert_size_stride(primals_124, (1024, 3072), (3072, 1), 'input')
            buf476 = reinterpret_tensor(buf468, (128, 3072), (3072, 1), 0); del buf468  # reuse
            # Topologically Sorted Source Nodes: [view_1150, view_1155, sum_136, squeeze_29, permute_882, clone_156, view_1157, view_1158, linear_93, permute_885, mm_404], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf474, (128, 1024), (1024, 1), 0), primals_124, out=buf476)
            del primals_124
            assert_size_stride(permute_879, (24, 128, 128), (128, 1, 3072), 'input')
            buf471 = buf467; del buf467  # reuse
            # Topologically Sorted Source Nodes: [view_1151, convert_element_type_1399, softmax_13, mul_701, neg_142, fma_14, convert_element_type_1400, mul_702, view_1152, bmm_114], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_879, reinterpret_tensor(buf470, (24, 128, 128), (16384, 128, 1), 0), out=buf471)
            del permute_879
            assert_size_stride(permute_880, (24, 128, 128), (16384, 128, 1), 'input')
            buf472 = reinterpret_tensor(buf466, (24, 128, 128), (16384, 128, 1), 0); del buf466  # reuse
            # Topologically Sorted Source Nodes: [view_1151, convert_element_type_1399, softmax_13, mul_701, neg_142, fma_14, convert_element_type_1400, mul_702, view_1152, bmm_115], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf470, (24, 128, 128), (16384, 128, 1), 0), permute_880, out=buf472)
            del buf470
            del permute_880
            buf473 = reinterpret_tensor(buf474, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf474  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1153, permute_881, view_1156, sum_137, squeeze_30, mul_703, slice_173, slice_174, neg_143, add_414, mul_704, add_415], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf471, primals_3, buf473, 131072, stream=raw_stream0)
            buf480 = reinterpret_tensor(buf471, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf471  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1154, mul_705, slice_175, slice_176, neg_144, add_416, mul_706, add_417, permute_892, clone_158], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf472, primals_3, buf480, 393216, stream=raw_stream0)
            buf477 = reinterpret_tensor(buf441, (128, 1024), (1024, 1), 0); del buf441  # reuse
            # Topologically Sorted Source Nodes: [permute_887, clone_157, view_1160, view_1161], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf473, buf477, 128, 1024, stream=raw_stream0)
            buf481 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1154, mul_705, slice_175, slice_176, neg_144, add_416, mul_706, add_417, permute_892, clone_158, view_1163, view_1164, permute_893, mm_407], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf480, (3072, 128), (1, 3072), 0), view_349, out=buf481)
            assert_size_stride(primals_122, (3072, 3072), (3072, 1), 'input')
            buf482 = reinterpret_tensor(buf472, (128, 3072), (3072, 1), 0); del buf472  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1154, mul_705, slice_175, slice_176, neg_144, add_416, mul_706, add_417, permute_892, clone_158, view_1163, view_1164, linear_91, permute_895, mm_408], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf480, (128, 3072), (3072, 1), 0), primals_122, out=buf482)
            del primals_122
            buf478 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_888, mm_405], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf477, (1024, 128), (1, 1024), 0), view_349, out=buf478)
            del view_349
            assert_size_stride(primals_123, (1024, 3072), (3072, 1), 'input')
            buf479 = reinterpret_tensor(buf480, (128, 3072), (3072, 1), 0); del buf480  # reuse
            # Topologically Sorted Source Nodes: [linear_92, permute_890, mm_406], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf477, primals_123, out=buf479)
            del primals_123
            assert_size_stride(primals_121, (3072, ), (1, ), 'input')
            assert_size_stride(add_104, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_26, (1, 128, 1), (128, 1, 1), 'input')
            buf485 = buf464; del buf464  # reuse
            # Topologically Sorted Source Nodes: [view_1159, view_1162, add_418, view_1165, add_419, mul_707, hidden_states_130, convert_element_type_1417, mul_709, mul_710, sum_139, pow_118, mul_711, mul_712, expand_203, div_88, pow_119, mul_713, mul_714, add_420, convert_element_type_1418, add_421], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf485, buf476, buf479, buf482, primals_121, add_104, rsqrt_26, 128, 3072, stream=raw_stream0)
            del primals_121
            buf483 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1159, view_1162, add_418, view_1165, add_419, hidden_states_130, hidden_states_131, to_71, mul_708, sum_138], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf476, buf479, buf482, add_104, rsqrt_26, buf483, 3072, 128, stream=raw_stream0)
            del add_104
            del rsqrt_26
            assert_size_stride(view_347, (128, 8192), (8192, 1), 'input')
            buf486 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1167, permute_897, mm_409], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf485, (3072, 128), (1, 3072), 0), view_347, out=buf486)
            del view_347
            assert_size_stride(primals_120, (3072, 8192), (8192, 1), 'input')
            buf487 = reinterpret_tensor(buf459, (128, 8192), (8192, 1), 0); del buf459  # reuse
            # Topologically Sorted Source Nodes: [view_1167, down_proj_12, permute_899, mm_410], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf485, (128, 3072), (3072, 1), 0), primals_120, out=buf487)
            del primals_120
            assert_size_stride(mm_88, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_89, (128, 8192), (8192, 1), 'input')
            buf488 = buf456; del buf456  # reuse
            buf491 = reinterpret_tensor(mm_89, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_89  # reuse
            # Topologically Sorted Source Nodes: [view_1168, linear_88, silu_12, mul_715, linear_89, mul_716, convert_element_type_1427, reciprocal_15, mul_717, mul_718, sub_48, mul_719, add_423, mul_720, convert_element_type_1429], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf491, buf487, mm_88, buf488, 1048576, stream=raw_stream0)
            del buf487
            del mm_88
            assert_size_stride(view_343, (128, 3072), (3072, 1), 'input')
            buf489 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, linear_88, silu_12, mul_715, view_1169, permute_901, mm_411], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf488, (8192, 128), (1, 8192), 0), view_343, out=buf489)
            assert_size_stride(primals_119, (8192, 3072), (3072, 1), 'input')
            buf490 = buf482; del buf482  # reuse
            # Topologically Sorted Source Nodes: [view_1168, linear_88, silu_12, mul_715, view_1169, linear_89, permute_903, mm_412], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf488, (128, 8192), (8192, 1), 0), primals_119, out=buf490)
            del primals_119
            buf492 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, linear_88, silu_12, linear_89, mul_716, convert_element_type_1427, reciprocal_15, mul_717, mul_718, sub_48, mul_719, add_423, mul_720, convert_element_type_1429, view_1171, permute_905, mm_413], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf491, (8192, 128), (1, 8192), 0), view_343, out=buf492)
            del view_343
            assert_size_stride(primals_118, (8192, 3072), (3072, 1), 'input')
            buf493 = buf479; del buf479  # reuse
            # Topologically Sorted Source Nodes: [view_1168, linear_88, silu_12, linear_89, mul_716, convert_element_type_1427, reciprocal_15, mul_717, mul_718, sub_48, mul_719, add_423, mul_720, convert_element_type_1429, view_1171, permute_907, mm_414], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf491, (128, 8192), (8192, 1), 0), primals_118, out=buf493)
            del primals_118
            assert_size_stride(primals_117, (3072, ), (1, ), 'input')
            assert_size_stride(add_101, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_25, (1, 128, 1), (128, 1, 1), 'input')
            buf496 = buf485; del buf485  # reuse
            # Topologically Sorted Source Nodes: [view_1170, view_1172, add_424, mul_721, hidden_states_126, convert_element_type_1434, mul_723, mul_724, sum_141, pow_120, mul_725, mul_726, expand_204, div_89, pow_121, mul_727, mul_728, add_425, convert_element_type_1435, add_426], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf496, buf490, buf493, primals_117, add_101, rsqrt_25, 128, 3072, stream=raw_stream0)
            del primals_117
            buf494 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1170, view_1172, add_424, hidden_states_126, hidden_states_127, to_69, mul_722, sum_140], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf490, buf493, add_101, rsqrt_25, buf494, 3072, 128, stream=raw_stream0)
            del add_101
            del rsqrt_25
            assert_size_stride(view_341, (128, 3072), (3072, 1), 'input')
            buf497 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1174, permute_909, mm_415], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf496, (3072, 128), (1, 3072), 0), view_341, out=buf497)
            del view_341
            assert_size_stride(primals_116, (3072, 3072), (3072, 1), 'input')
            buf498 = buf493; del buf493  # reuse
            # Topologically Sorted Source Nodes: [view_1174, attn_output_51, permute_911, mm_416], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf496, (128, 3072), (3072, 1), 0), primals_116, out=buf498)
            del primals_116
            assert_size_stride(permute_914, (24, 128, 128), (16384, 1, 128), 'input')
            buf499 = reinterpret_tensor(buf490, (24, 128, 128), (16384, 128, 1), 0); del buf490  # reuse
            # Topologically Sorted Source Nodes: [view_1175, view_1176, permute_913, view_1177, bmm_116], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_914, reinterpret_tensor(buf498, (24, 128, 128), (128, 3072, 1), 0), out=buf499)
            del permute_914
            assert_size_stride(permute_915, (24, 128, 128), (16384, 1, 128), 'input')
            buf500 = reinterpret_tensor(buf476, (24, 128, 128), (16384, 128, 1), 0); del buf476  # reuse
            # Topologically Sorted Source Nodes: [view_1175, view_1176, permute_913, view_1177, bmm_117], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf498, (24, 128, 128), (128, 3072, 1), 0), permute_915, out=buf500)
            del permute_915
            buf506 = reinterpret_tensor(buf477, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf477  # reuse
            # Topologically Sorted Source Nodes: [view_1178, view_1183, sum_143, squeeze_31, permute_919, clone_159], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf499, buf506, 131072, stream=raw_stream0)
            assert_size_stride(add_100, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_12, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_13, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf502 = add_100; del add_100  # reuse
            # Topologically Sorted Source Nodes: [view_1179, convert_element_type_1444, softmax_12, mul_729, sum_142, neg_146, fma_15, convert_element_type_1445, mul_730], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf502, buf500, amax_12, sum_13, 3072, 128, stream=raw_stream0)
            del amax_12
            del sum_13
            assert_size_stride(view_323, (128, 3072), (3072, 1), 'input')
            buf507 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1178, view_1183, sum_143, squeeze_31, permute_919, clone_159, view_1185, view_1186, permute_920, mm_417], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf506, (1024, 128), (1, 1024), 0), view_323, out=buf507)
            assert_size_stride(primals_115, (1024, 3072), (3072, 1), 'input')
            buf508 = reinterpret_tensor(buf500, (128, 3072), (3072, 1), 0); del buf500  # reuse
            # Topologically Sorted Source Nodes: [view_1178, view_1183, sum_143, squeeze_31, permute_919, clone_159, view_1185, view_1186, linear_86, permute_922, mm_418], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf506, (128, 1024), (1024, 1), 0), primals_115, out=buf508)
            del primals_115
            assert_size_stride(permute_916, (24, 128, 128), (128, 1, 3072), 'input')
            buf503 = buf499; del buf499  # reuse
            # Topologically Sorted Source Nodes: [view_1179, convert_element_type_1444, softmax_12, mul_729, neg_146, fma_15, convert_element_type_1445, mul_730, view_1180, bmm_118], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_916, reinterpret_tensor(buf502, (24, 128, 128), (16384, 128, 1), 0), out=buf503)
            del permute_916
            assert_size_stride(permute_917, (24, 128, 128), (16384, 128, 1), 'input')
            buf504 = reinterpret_tensor(buf498, (24, 128, 128), (16384, 128, 1), 0); del buf498  # reuse
            # Topologically Sorted Source Nodes: [view_1179, convert_element_type_1444, softmax_12, mul_729, neg_146, fma_15, convert_element_type_1445, mul_730, view_1180, bmm_119], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf502, (24, 128, 128), (16384, 128, 1), 0), permute_917, out=buf504)
            del buf502
            del permute_917
            buf505 = reinterpret_tensor(buf506, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf506  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1181, permute_918, view_1184, sum_144, squeeze_32, mul_731, slice_177, slice_178, neg_147, add_427, mul_732, add_428], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf503, primals_3, buf505, 131072, stream=raw_stream0)
            buf512 = reinterpret_tensor(buf503, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf503  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1182, mul_733, slice_179, slice_180, neg_148, add_429, mul_734, add_430, permute_929, clone_161], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf504, primals_3, buf512, 393216, stream=raw_stream0)
            buf509 = reinterpret_tensor(buf473, (128, 1024), (1024, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [permute_924, clone_160, view_1188, view_1189], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf505, buf509, 128, 1024, stream=raw_stream0)
            buf513 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1182, mul_733, slice_179, slice_180, neg_148, add_429, mul_734, add_430, permute_929, clone_161, view_1191, view_1192, permute_930, mm_421], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf512, (3072, 128), (1, 3072), 0), view_323, out=buf513)
            assert_size_stride(primals_113, (3072, 3072), (3072, 1), 'input')
            buf514 = reinterpret_tensor(buf504, (128, 3072), (3072, 1), 0); del buf504  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1182, mul_733, slice_179, slice_180, neg_148, add_429, mul_734, add_430, permute_929, clone_161, view_1191, view_1192, linear_84, permute_932, mm_422], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf512, (128, 3072), (3072, 1), 0), primals_113, out=buf514)
            del primals_113
            buf510 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_925, mm_419], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf509, (1024, 128), (1, 1024), 0), view_323, out=buf510)
            del view_323
            assert_size_stride(primals_114, (1024, 3072), (3072, 1), 'input')
            buf511 = reinterpret_tensor(buf512, (128, 3072), (3072, 1), 0); del buf512  # reuse
            # Topologically Sorted Source Nodes: [linear_85, permute_927, mm_420], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf509, primals_114, out=buf511)
            del primals_114
            assert_size_stride(primals_112, (3072, ), (1, ), 'input')
            assert_size_stride(add_96, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_24, (1, 128, 1), (128, 1, 1), 'input')
            buf517 = buf496; del buf496  # reuse
            # Topologically Sorted Source Nodes: [view_1187, view_1190, add_431, view_1193, add_432, mul_735, hidden_states_120, convert_element_type_1462, mul_737, mul_738, sum_146, pow_122, mul_739, mul_740, expand_205, div_90, pow_123, mul_741, mul_742, add_433, convert_element_type_1463, add_434], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf517, buf508, buf511, buf514, primals_112, add_96, rsqrt_24, 128, 3072, stream=raw_stream0)
            del primals_112
            buf515 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1187, view_1190, add_431, view_1193, add_432, hidden_states_120, hidden_states_121, to_66, mul_736, sum_145], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf508, buf511, buf514, add_96, rsqrt_24, buf515, 3072, 128, stream=raw_stream0)
            del add_96
            del rsqrt_24
            assert_size_stride(view_321, (128, 8192), (8192, 1), 'input')
            buf518 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1195, permute_934, mm_423], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf517, (3072, 128), (1, 3072), 0), view_321, out=buf518)
            del view_321
            assert_size_stride(primals_111, (3072, 8192), (8192, 1), 'input')
            buf519 = reinterpret_tensor(buf491, (128, 8192), (8192, 1), 0); del buf491  # reuse
            # Topologically Sorted Source Nodes: [view_1195, down_proj_11, permute_936, mm_424], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf517, (128, 3072), (3072, 1), 0), primals_111, out=buf519)
            del primals_111
            assert_size_stride(mm_81, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_82, (128, 8192), (8192, 1), 'input')
            buf520 = buf488; del buf488  # reuse
            buf523 = reinterpret_tensor(mm_82, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_82  # reuse
            # Topologically Sorted Source Nodes: [view_1196, linear_81, silu_11, mul_743, linear_82, mul_744, convert_element_type_1472, reciprocal_16, mul_745, mul_746, sub_49, mul_747, add_436, mul_748, convert_element_type_1474], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf523, buf519, mm_81, buf520, 1048576, stream=raw_stream0)
            del buf519
            del mm_81
            assert_size_stride(view_317, (128, 3072), (3072, 1), 'input')
            buf521 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1196, linear_81, silu_11, mul_743, view_1197, permute_938, mm_425], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf520, (8192, 128), (1, 8192), 0), view_317, out=buf521)
            assert_size_stride(primals_110, (8192, 3072), (3072, 1), 'input')
            buf522 = buf514; del buf514  # reuse
            # Topologically Sorted Source Nodes: [view_1196, linear_81, silu_11, mul_743, view_1197, linear_82, permute_940, mm_426], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf520, (128, 8192), (8192, 1), 0), primals_110, out=buf522)
            del primals_110
            buf524 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1196, linear_81, silu_11, linear_82, mul_744, convert_element_type_1472, reciprocal_16, mul_745, mul_746, sub_49, mul_747, add_436, mul_748, convert_element_type_1474, view_1199, permute_942, mm_427], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf523, (8192, 128), (1, 8192), 0), view_317, out=buf524)
            del view_317
            assert_size_stride(primals_109, (8192, 3072), (3072, 1), 'input')
            buf525 = buf511; del buf511  # reuse
            # Topologically Sorted Source Nodes: [view_1196, linear_81, silu_11, linear_82, mul_744, convert_element_type_1472, reciprocal_16, mul_745, mul_746, sub_49, mul_747, add_436, mul_748, convert_element_type_1474, view_1199, permute_944, mm_428], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf523, (128, 8192), (8192, 1), 0), primals_109, out=buf525)
            del primals_109
            assert_size_stride(primals_108, (3072, ), (1, ), 'input')
            assert_size_stride(add_93, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_23, (1, 128, 1), (128, 1, 1), 'input')
            buf528 = buf517; del buf517  # reuse
            # Topologically Sorted Source Nodes: [view_1198, view_1200, add_437, mul_749, hidden_states_116, convert_element_type_1479, mul_751, mul_752, sum_148, pow_124, mul_753, mul_754, expand_206, div_91, pow_125, mul_755, mul_756, add_438, convert_element_type_1480, add_439], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf528, buf522, buf525, primals_108, add_93, rsqrt_23, 128, 3072, stream=raw_stream0)
            del primals_108
            buf526 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1198, view_1200, add_437, hidden_states_116, hidden_states_117, to_64, mul_750, sum_147], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf522, buf525, add_93, rsqrt_23, buf526, 3072, 128, stream=raw_stream0)
            del add_93
            del rsqrt_23
            assert_size_stride(view_315, (128, 3072), (3072, 1), 'input')
            buf529 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1202, permute_946, mm_429], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf528, (3072, 128), (1, 3072), 0), view_315, out=buf529)
            del view_315
            assert_size_stride(primals_107, (3072, 3072), (3072, 1), 'input')
            buf530 = buf525; del buf525  # reuse
            # Topologically Sorted Source Nodes: [view_1202, attn_output_47, permute_948, mm_430], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf528, (128, 3072), (3072, 1), 0), primals_107, out=buf530)
            del primals_107
            assert_size_stride(permute_951, (24, 128, 128), (16384, 1, 128), 'input')
            buf531 = reinterpret_tensor(buf522, (24, 128, 128), (16384, 128, 1), 0); del buf522  # reuse
            # Topologically Sorted Source Nodes: [view_1203, view_1204, permute_950, view_1205, bmm_120], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_951, reinterpret_tensor(buf530, (24, 128, 128), (128, 3072, 1), 0), out=buf531)
            del permute_951
            assert_size_stride(permute_952, (24, 128, 128), (16384, 1, 128), 'input')
            buf532 = reinterpret_tensor(buf508, (24, 128, 128), (16384, 128, 1), 0); del buf508  # reuse
            # Topologically Sorted Source Nodes: [view_1203, view_1204, permute_950, view_1205, bmm_121], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf530, (24, 128, 128), (128, 3072, 1), 0), permute_952, out=buf532)
            del permute_952
            buf538 = reinterpret_tensor(buf509, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf509  # reuse
            # Topologically Sorted Source Nodes: [view_1206, view_1211, sum_150, squeeze_33, permute_956, clone_162], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf531, buf538, 131072, stream=raw_stream0)
            assert_size_stride(add_92, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_11, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_12, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf534 = add_92; del add_92  # reuse
            # Topologically Sorted Source Nodes: [view_1207, convert_element_type_1489, softmax_11, mul_757, sum_149, neg_150, fma_16, convert_element_type_1490, mul_758], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf534, buf532, amax_11, sum_12, 3072, 128, stream=raw_stream0)
            del amax_11
            del sum_12
            assert_size_stride(view_297, (128, 3072), (3072, 1), 'input')
            buf539 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1206, view_1211, sum_150, squeeze_33, permute_956, clone_162, view_1213, view_1214, permute_957, mm_431], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf538, (1024, 128), (1, 1024), 0), view_297, out=buf539)
            assert_size_stride(primals_106, (1024, 3072), (3072, 1), 'input')
            buf540 = reinterpret_tensor(buf532, (128, 3072), (3072, 1), 0); del buf532  # reuse
            # Topologically Sorted Source Nodes: [view_1206, view_1211, sum_150, squeeze_33, permute_956, clone_162, view_1213, view_1214, linear_79, permute_959, mm_432], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf538, (128, 1024), (1024, 1), 0), primals_106, out=buf540)
            del primals_106
            assert_size_stride(permute_953, (24, 128, 128), (128, 1, 3072), 'input')
            buf535 = buf531; del buf531  # reuse
            # Topologically Sorted Source Nodes: [view_1207, convert_element_type_1489, softmax_11, mul_757, neg_150, fma_16, convert_element_type_1490, mul_758, view_1208, bmm_122], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_953, reinterpret_tensor(buf534, (24, 128, 128), (16384, 128, 1), 0), out=buf535)
            del permute_953
            assert_size_stride(permute_954, (24, 128, 128), (16384, 128, 1), 'input')
            buf536 = reinterpret_tensor(buf530, (24, 128, 128), (16384, 128, 1), 0); del buf530  # reuse
            # Topologically Sorted Source Nodes: [view_1207, convert_element_type_1489, softmax_11, mul_757, neg_150, fma_16, convert_element_type_1490, mul_758, view_1208, bmm_123], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf534, (24, 128, 128), (16384, 128, 1), 0), permute_954, out=buf536)
            del buf534
            del permute_954
            buf537 = reinterpret_tensor(buf538, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf538  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1209, permute_955, view_1212, sum_151, squeeze_34, mul_759, slice_181, slice_182, neg_151, add_440, mul_760, add_441], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf535, primals_3, buf537, 131072, stream=raw_stream0)
            buf544 = reinterpret_tensor(buf535, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf535  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1210, mul_761, slice_183, slice_184, neg_152, add_442, mul_762, add_443, permute_966, clone_164], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf536, primals_3, buf544, 393216, stream=raw_stream0)
            buf541 = reinterpret_tensor(buf505, (128, 1024), (1024, 1), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [permute_961, clone_163, view_1216, view_1217], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf537, buf541, 128, 1024, stream=raw_stream0)
            buf545 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1210, mul_761, slice_183, slice_184, neg_152, add_442, mul_762, add_443, permute_966, clone_164, view_1219, view_1220, permute_967, mm_435], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf544, (3072, 128), (1, 3072), 0), view_297, out=buf545)
            assert_size_stride(primals_104, (3072, 3072), (3072, 1), 'input')
            buf546 = reinterpret_tensor(buf536, (128, 3072), (3072, 1), 0); del buf536  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1210, mul_761, slice_183, slice_184, neg_152, add_442, mul_762, add_443, permute_966, clone_164, view_1219, view_1220, linear_77, permute_969, mm_436], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf544, (128, 3072), (3072, 1), 0), primals_104, out=buf546)
            del primals_104
            buf542 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_962, mm_433], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf541, (1024, 128), (1, 1024), 0), view_297, out=buf542)
            del view_297
            assert_size_stride(primals_105, (1024, 3072), (3072, 1), 'input')
            buf543 = reinterpret_tensor(buf544, (128, 3072), (3072, 1), 0); del buf544  # reuse
            # Topologically Sorted Source Nodes: [linear_78, permute_964, mm_434], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf541, primals_105, out=buf543)
            del primals_105
            assert_size_stride(primals_103, (3072, ), (1, ), 'input')
            assert_size_stride(add_88, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_22, (1, 128, 1), (128, 1, 1), 'input')
            buf549 = buf528; del buf528  # reuse
            # Topologically Sorted Source Nodes: [view_1215, view_1218, add_444, view_1221, add_445, mul_763, hidden_states_110, convert_element_type_1507, mul_765, mul_766, sum_153, pow_126, mul_767, mul_768, expand_207, div_92, pow_127, mul_769, mul_770, add_446, convert_element_type_1508, add_447], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf549, buf540, buf543, buf546, primals_103, add_88, rsqrt_22, 128, 3072, stream=raw_stream0)
            del primals_103
            buf547 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1215, view_1218, add_444, view_1221, add_445, hidden_states_110, hidden_states_111, to_61, mul_764, sum_152], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf540, buf543, buf546, add_88, rsqrt_22, buf547, 3072, 128, stream=raw_stream0)
            del add_88
            del rsqrt_22
            assert_size_stride(view_295, (128, 8192), (8192, 1), 'input')
            buf550 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1223, permute_971, mm_437], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf549, (3072, 128), (1, 3072), 0), view_295, out=buf550)
            del view_295
            assert_size_stride(primals_102, (3072, 8192), (8192, 1), 'input')
            buf551 = reinterpret_tensor(buf523, (128, 8192), (8192, 1), 0); del buf523  # reuse
            # Topologically Sorted Source Nodes: [view_1223, down_proj_10, permute_973, mm_438], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf549, (128, 3072), (3072, 1), 0), primals_102, out=buf551)
            del primals_102
            assert_size_stride(mm_74, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_75, (128, 8192), (8192, 1), 'input')
            buf552 = buf520; del buf520  # reuse
            buf555 = reinterpret_tensor(mm_75, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_75  # reuse
            # Topologically Sorted Source Nodes: [view_1224, linear_74, silu_10, mul_771, linear_75, mul_772, convert_element_type_1517, reciprocal_17, mul_773, mul_774, sub_50, mul_775, add_449, mul_776, convert_element_type_1519], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf555, buf551, mm_74, buf552, 1048576, stream=raw_stream0)
            del buf551
            del mm_74
            assert_size_stride(view_291, (128, 3072), (3072, 1), 'input')
            buf553 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1224, linear_74, silu_10, mul_771, view_1225, permute_975, mm_439], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf552, (8192, 128), (1, 8192), 0), view_291, out=buf553)
            assert_size_stride(primals_101, (8192, 3072), (3072, 1), 'input')
            buf554 = buf546; del buf546  # reuse
            # Topologically Sorted Source Nodes: [view_1224, linear_74, silu_10, mul_771, view_1225, linear_75, permute_977, mm_440], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf552, (128, 8192), (8192, 1), 0), primals_101, out=buf554)
            del primals_101
            buf556 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1224, linear_74, silu_10, linear_75, mul_772, convert_element_type_1517, reciprocal_17, mul_773, mul_774, sub_50, mul_775, add_449, mul_776, convert_element_type_1519, view_1227, permute_979, mm_441], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf555, (8192, 128), (1, 8192), 0), view_291, out=buf556)
            del view_291
            assert_size_stride(primals_100, (8192, 3072), (3072, 1), 'input')
            buf557 = buf543; del buf543  # reuse
            # Topologically Sorted Source Nodes: [view_1224, linear_74, silu_10, linear_75, mul_772, convert_element_type_1517, reciprocal_17, mul_773, mul_774, sub_50, mul_775, add_449, mul_776, convert_element_type_1519, view_1227, permute_981, mm_442], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf555, (128, 8192), (8192, 1), 0), primals_100, out=buf557)
            del primals_100
            assert_size_stride(primals_99, (3072, ), (1, ), 'input')
            assert_size_stride(add_85, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_21, (1, 128, 1), (128, 1, 1), 'input')
            buf560 = buf549; del buf549  # reuse
            # Topologically Sorted Source Nodes: [view_1226, view_1228, add_450, mul_777, hidden_states_106, convert_element_type_1524, mul_779, mul_780, sum_155, pow_128, mul_781, mul_782, expand_208, div_93, pow_129, mul_783, mul_784, add_451, convert_element_type_1525, add_452], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf560, buf554, buf557, primals_99, add_85, rsqrt_21, 128, 3072, stream=raw_stream0)
            del primals_99
            buf558 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1226, view_1228, add_450, hidden_states_106, hidden_states_107, to_59, mul_778, sum_154], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf554, buf557, add_85, rsqrt_21, buf558, 3072, 128, stream=raw_stream0)
            del add_85
            del rsqrt_21
            assert_size_stride(view_289, (128, 3072), (3072, 1), 'input')
            buf561 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1230, permute_983, mm_443], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf560, (3072, 128), (1, 3072), 0), view_289, out=buf561)
            del view_289
            assert_size_stride(primals_98, (3072, 3072), (3072, 1), 'input')
            buf562 = buf557; del buf557  # reuse
            # Topologically Sorted Source Nodes: [view_1230, attn_output_43, permute_985, mm_444], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf560, (128, 3072), (3072, 1), 0), primals_98, out=buf562)
            del primals_98
            assert_size_stride(permute_988, (24, 128, 128), (16384, 1, 128), 'input')
            buf563 = reinterpret_tensor(buf554, (24, 128, 128), (16384, 128, 1), 0); del buf554  # reuse
            # Topologically Sorted Source Nodes: [view_1231, view_1232, permute_987, view_1233, bmm_124], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_988, reinterpret_tensor(buf562, (24, 128, 128), (128, 3072, 1), 0), out=buf563)
            del permute_988
            assert_size_stride(permute_989, (24, 128, 128), (16384, 1, 128), 'input')
            buf564 = reinterpret_tensor(buf540, (24, 128, 128), (16384, 128, 1), 0); del buf540  # reuse
            # Topologically Sorted Source Nodes: [view_1231, view_1232, permute_987, view_1233, bmm_125], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf562, (24, 128, 128), (128, 3072, 1), 0), permute_989, out=buf564)
            del permute_989
            buf570 = reinterpret_tensor(buf541, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf541  # reuse
            # Topologically Sorted Source Nodes: [view_1234, view_1239, sum_157, squeeze_35, permute_993, clone_165], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf563, buf570, 131072, stream=raw_stream0)
            assert_size_stride(add_84, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_10, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_11, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf566 = add_84; del add_84  # reuse
            # Topologically Sorted Source Nodes: [view_1235, convert_element_type_1534, softmax_10, mul_785, sum_156, neg_154, fma_17, convert_element_type_1535, mul_786], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf566, buf564, amax_10, sum_11, 3072, 128, stream=raw_stream0)
            del amax_10
            del sum_11
            assert_size_stride(view_271, (128, 3072), (3072, 1), 'input')
            buf571 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1234, view_1239, sum_157, squeeze_35, permute_993, clone_165, view_1241, view_1242, permute_994, mm_445], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf570, (1024, 128), (1, 1024), 0), view_271, out=buf571)
            assert_size_stride(primals_97, (1024, 3072), (3072, 1), 'input')
            buf572 = reinterpret_tensor(buf564, (128, 3072), (3072, 1), 0); del buf564  # reuse
            # Topologically Sorted Source Nodes: [view_1234, view_1239, sum_157, squeeze_35, permute_993, clone_165, view_1241, view_1242, linear_72, permute_996, mm_446], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf570, (128, 1024), (1024, 1), 0), primals_97, out=buf572)
            del primals_97
            assert_size_stride(permute_990, (24, 128, 128), (128, 1, 3072), 'input')
            buf567 = buf563; del buf563  # reuse
            # Topologically Sorted Source Nodes: [view_1235, convert_element_type_1534, softmax_10, mul_785, neg_154, fma_17, convert_element_type_1535, mul_786, view_1236, bmm_126], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_990, reinterpret_tensor(buf566, (24, 128, 128), (16384, 128, 1), 0), out=buf567)
            del permute_990
            assert_size_stride(permute_991, (24, 128, 128), (16384, 128, 1), 'input')
            buf568 = reinterpret_tensor(buf562, (24, 128, 128), (16384, 128, 1), 0); del buf562  # reuse
            # Topologically Sorted Source Nodes: [view_1235, convert_element_type_1534, softmax_10, mul_785, neg_154, fma_17, convert_element_type_1535, mul_786, view_1236, bmm_127], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf566, (24, 128, 128), (16384, 128, 1), 0), permute_991, out=buf568)
            del buf566
            del permute_991
            buf569 = reinterpret_tensor(buf570, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf570  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1237, permute_992, view_1240, sum_158, squeeze_36, mul_787, slice_185, slice_186, neg_155, add_453, mul_788, add_454], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf567, primals_3, buf569, 131072, stream=raw_stream0)
            buf576 = reinterpret_tensor(buf567, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf567  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1238, mul_789, slice_187, slice_188, neg_156, add_455, mul_790, add_456, permute_1003, clone_167], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf568, primals_3, buf576, 393216, stream=raw_stream0)
            buf573 = reinterpret_tensor(buf537, (128, 1024), (1024, 1), 0); del buf537  # reuse
            # Topologically Sorted Source Nodes: [permute_998, clone_166, view_1244, view_1245], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf569, buf573, 128, 1024, stream=raw_stream0)
            buf577 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1238, mul_789, slice_187, slice_188, neg_156, add_455, mul_790, add_456, permute_1003, clone_167, view_1247, view_1248, permute_1004, mm_449], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf576, (3072, 128), (1, 3072), 0), view_271, out=buf577)
            assert_size_stride(primals_95, (3072, 3072), (3072, 1), 'input')
            buf578 = reinterpret_tensor(buf568, (128, 3072), (3072, 1), 0); del buf568  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1238, mul_789, slice_187, slice_188, neg_156, add_455, mul_790, add_456, permute_1003, clone_167, view_1247, view_1248, linear_70, permute_1006, mm_450], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf576, (128, 3072), (3072, 1), 0), primals_95, out=buf578)
            del primals_95
            buf574 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_999, mm_447], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf573, (1024, 128), (1, 1024), 0), view_271, out=buf574)
            del view_271
            assert_size_stride(primals_96, (1024, 3072), (3072, 1), 'input')
            buf575 = reinterpret_tensor(buf576, (128, 3072), (3072, 1), 0); del buf576  # reuse
            # Topologically Sorted Source Nodes: [linear_71, permute_1001, mm_448], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf573, primals_96, out=buf575)
            del primals_96
            assert_size_stride(primals_94, (3072, ), (1, ), 'input')
            assert_size_stride(add_80, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_20, (1, 128, 1), (128, 1, 1), 'input')
            buf581 = buf560; del buf560  # reuse
            # Topologically Sorted Source Nodes: [view_1243, view_1246, add_457, view_1249, add_458, mul_791, hidden_states_100, convert_element_type_1552, mul_793, mul_794, sum_160, pow_130, mul_795, mul_796, expand_209, div_94, pow_131, mul_797, mul_798, add_459, convert_element_type_1553, add_460], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf581, buf572, buf575, buf578, primals_94, add_80, rsqrt_20, 128, 3072, stream=raw_stream0)
            del primals_94
            buf579 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1243, view_1246, add_457, view_1249, add_458, hidden_states_100, hidden_states_101, to_56, mul_792, sum_159], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf572, buf575, buf578, add_80, rsqrt_20, buf579, 3072, 128, stream=raw_stream0)
            del add_80
            del rsqrt_20
            assert_size_stride(view_269, (128, 8192), (8192, 1), 'input')
            buf582 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1251, permute_1008, mm_451], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf581, (3072, 128), (1, 3072), 0), view_269, out=buf582)
            del view_269
            assert_size_stride(primals_93, (3072, 8192), (8192, 1), 'input')
            buf583 = reinterpret_tensor(buf555, (128, 8192), (8192, 1), 0); del buf555  # reuse
            # Topologically Sorted Source Nodes: [view_1251, down_proj_9, permute_1010, mm_452], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf581, (128, 3072), (3072, 1), 0), primals_93, out=buf583)
            del primals_93
            assert_size_stride(mm_67, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_68, (128, 8192), (8192, 1), 'input')
            buf584 = buf552; del buf552  # reuse
            buf587 = reinterpret_tensor(mm_68, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_68  # reuse
            # Topologically Sorted Source Nodes: [view_1252, linear_67, silu_9, mul_799, linear_68, mul_800, convert_element_type_1562, reciprocal_18, mul_801, mul_802, sub_51, mul_803, add_462, mul_804, convert_element_type_1564], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf587, buf583, mm_67, buf584, 1048576, stream=raw_stream0)
            del buf583
            del mm_67
            assert_size_stride(view_265, (128, 3072), (3072, 1), 'input')
            buf585 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1252, linear_67, silu_9, mul_799, view_1253, permute_1012, mm_453], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf584, (8192, 128), (1, 8192), 0), view_265, out=buf585)
            assert_size_stride(primals_92, (8192, 3072), (3072, 1), 'input')
            buf586 = buf578; del buf578  # reuse
            # Topologically Sorted Source Nodes: [view_1252, linear_67, silu_9, mul_799, view_1253, linear_68, permute_1014, mm_454], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf584, (128, 8192), (8192, 1), 0), primals_92, out=buf586)
            del primals_92
            buf588 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1252, linear_67, silu_9, linear_68, mul_800, convert_element_type_1562, reciprocal_18, mul_801, mul_802, sub_51, mul_803, add_462, mul_804, convert_element_type_1564, view_1255, permute_1016, mm_455], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf587, (8192, 128), (1, 8192), 0), view_265, out=buf588)
            del view_265
            assert_size_stride(primals_91, (8192, 3072), (3072, 1), 'input')
            buf589 = buf575; del buf575  # reuse
            # Topologically Sorted Source Nodes: [view_1252, linear_67, silu_9, linear_68, mul_800, convert_element_type_1562, reciprocal_18, mul_801, mul_802, sub_51, mul_803, add_462, mul_804, convert_element_type_1564, view_1255, permute_1018, mm_456], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf587, (128, 8192), (8192, 1), 0), primals_91, out=buf589)
            del primals_91
            assert_size_stride(primals_90, (3072, ), (1, ), 'input')
            assert_size_stride(add_77, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_19, (1, 128, 1), (128, 1, 1), 'input')
            buf592 = buf581; del buf581  # reuse
            # Topologically Sorted Source Nodes: [view_1254, view_1256, add_463, mul_805, hidden_states_96, convert_element_type_1569, mul_807, mul_808, sum_162, pow_132, mul_809, mul_810, expand_210, div_95, pow_133, mul_811, mul_812, add_464, convert_element_type_1570, add_465], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf592, buf586, buf589, primals_90, add_77, rsqrt_19, 128, 3072, stream=raw_stream0)
            del primals_90
            buf590 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1254, view_1256, add_463, hidden_states_96, hidden_states_97, to_54, mul_806, sum_161], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf586, buf589, add_77, rsqrt_19, buf590, 3072, 128, stream=raw_stream0)
            del add_77
            del rsqrt_19
            assert_size_stride(view_263, (128, 3072), (3072, 1), 'input')
            buf593 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1258, permute_1020, mm_457], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf592, (3072, 128), (1, 3072), 0), view_263, out=buf593)
            del view_263
            assert_size_stride(primals_89, (3072, 3072), (3072, 1), 'input')
            buf594 = buf589; del buf589  # reuse
            # Topologically Sorted Source Nodes: [view_1258, attn_output_39, permute_1022, mm_458], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf592, (128, 3072), (3072, 1), 0), primals_89, out=buf594)
            del primals_89
            assert_size_stride(permute_1025, (24, 128, 128), (16384, 1, 128), 'input')
            buf595 = reinterpret_tensor(buf586, (24, 128, 128), (16384, 128, 1), 0); del buf586  # reuse
            # Topologically Sorted Source Nodes: [view_1259, view_1260, permute_1024, view_1261, bmm_128], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1025, reinterpret_tensor(buf594, (24, 128, 128), (128, 3072, 1), 0), out=buf595)
            del permute_1025
            assert_size_stride(permute_1026, (24, 128, 128), (16384, 1, 128), 'input')
            buf596 = reinterpret_tensor(buf572, (24, 128, 128), (16384, 128, 1), 0); del buf572  # reuse
            # Topologically Sorted Source Nodes: [view_1259, view_1260, permute_1024, view_1261, bmm_129], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf594, (24, 128, 128), (128, 3072, 1), 0), permute_1026, out=buf596)
            del permute_1026
            buf602 = reinterpret_tensor(buf573, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf573  # reuse
            # Topologically Sorted Source Nodes: [view_1262, view_1267, sum_164, squeeze_37, permute_1030, clone_168], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf595, buf602, 131072, stream=raw_stream0)
            assert_size_stride(add_76, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_9, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_10, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf598 = add_76; del add_76  # reuse
            # Topologically Sorted Source Nodes: [view_1263, convert_element_type_1579, softmax_9, mul_813, sum_163, neg_158, fma_18, convert_element_type_1580, mul_814], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf598, buf596, amax_9, sum_10, 3072, 128, stream=raw_stream0)
            del amax_9
            del sum_10
            assert_size_stride(view_245, (128, 3072), (3072, 1), 'input')
            buf603 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1262, view_1267, sum_164, squeeze_37, permute_1030, clone_168, view_1269, view_1270, permute_1031, mm_459], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf602, (1024, 128), (1, 1024), 0), view_245, out=buf603)
            assert_size_stride(primals_88, (1024, 3072), (3072, 1), 'input')
            buf604 = reinterpret_tensor(buf596, (128, 3072), (3072, 1), 0); del buf596  # reuse
            # Topologically Sorted Source Nodes: [view_1262, view_1267, sum_164, squeeze_37, permute_1030, clone_168, view_1269, view_1270, linear_65, permute_1033, mm_460], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf602, (128, 1024), (1024, 1), 0), primals_88, out=buf604)
            del primals_88
            assert_size_stride(permute_1027, (24, 128, 128), (128, 1, 3072), 'input')
            buf599 = buf595; del buf595  # reuse
            # Topologically Sorted Source Nodes: [view_1263, convert_element_type_1579, softmax_9, mul_813, neg_158, fma_18, convert_element_type_1580, mul_814, view_1264, bmm_130], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1027, reinterpret_tensor(buf598, (24, 128, 128), (16384, 128, 1), 0), out=buf599)
            del permute_1027
            assert_size_stride(permute_1028, (24, 128, 128), (16384, 128, 1), 'input')
            buf600 = reinterpret_tensor(buf594, (24, 128, 128), (16384, 128, 1), 0); del buf594  # reuse
            # Topologically Sorted Source Nodes: [view_1263, convert_element_type_1579, softmax_9, mul_813, neg_158, fma_18, convert_element_type_1580, mul_814, view_1264, bmm_131], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf598, (24, 128, 128), (16384, 128, 1), 0), permute_1028, out=buf600)
            del buf598
            del permute_1028
            buf601 = reinterpret_tensor(buf602, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf602  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1265, permute_1029, view_1268, sum_165, squeeze_38, mul_815, slice_189, slice_190, neg_159, add_466, mul_816, add_467], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf599, primals_3, buf601, 131072, stream=raw_stream0)
            buf608 = reinterpret_tensor(buf599, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf599  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1266, mul_817, slice_191, slice_192, neg_160, add_468, mul_818, add_469, permute_1040, clone_170], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf600, primals_3, buf608, 393216, stream=raw_stream0)
            buf605 = reinterpret_tensor(buf569, (128, 1024), (1024, 1), 0); del buf569  # reuse
            # Topologically Sorted Source Nodes: [permute_1035, clone_169, view_1272, view_1273], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf601, buf605, 128, 1024, stream=raw_stream0)
            buf609 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1266, mul_817, slice_191, slice_192, neg_160, add_468, mul_818, add_469, permute_1040, clone_170, view_1275, view_1276, permute_1041, mm_463], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf608, (3072, 128), (1, 3072), 0), view_245, out=buf609)
            assert_size_stride(primals_86, (3072, 3072), (3072, 1), 'input')
            buf610 = reinterpret_tensor(buf600, (128, 3072), (3072, 1), 0); del buf600  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1266, mul_817, slice_191, slice_192, neg_160, add_468, mul_818, add_469, permute_1040, clone_170, view_1275, view_1276, linear_63, permute_1043, mm_464], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf608, (128, 3072), (3072, 1), 0), primals_86, out=buf610)
            del primals_86
            buf606 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1036, mm_461], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf605, (1024, 128), (1, 1024), 0), view_245, out=buf606)
            del view_245
            assert_size_stride(primals_87, (1024, 3072), (3072, 1), 'input')
            buf607 = reinterpret_tensor(buf608, (128, 3072), (3072, 1), 0); del buf608  # reuse
            # Topologically Sorted Source Nodes: [linear_64, permute_1038, mm_462], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf605, primals_87, out=buf607)
            del primals_87
            assert_size_stride(primals_85, (3072, ), (1, ), 'input')
            assert_size_stride(add_72, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_18, (1, 128, 1), (128, 1, 1), 'input')
            buf613 = buf592; del buf592  # reuse
            # Topologically Sorted Source Nodes: [view_1271, view_1274, add_470, view_1277, add_471, mul_819, hidden_states_90, convert_element_type_1597, mul_821, mul_822, sum_167, pow_134, mul_823, mul_824, expand_211, div_96, pow_135, mul_825, mul_826, add_472, convert_element_type_1598, add_473], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf613, buf604, buf607, buf610, primals_85, add_72, rsqrt_18, 128, 3072, stream=raw_stream0)
            del primals_85
            buf611 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1271, view_1274, add_470, view_1277, add_471, hidden_states_90, hidden_states_91, to_51, mul_820, sum_166], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf604, buf607, buf610, add_72, rsqrt_18, buf611, 3072, 128, stream=raw_stream0)
            del add_72
            del rsqrt_18
            assert_size_stride(view_243, (128, 8192), (8192, 1), 'input')
            buf614 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1279, permute_1045, mm_465], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf613, (3072, 128), (1, 3072), 0), view_243, out=buf614)
            del view_243
            assert_size_stride(primals_84, (3072, 8192), (8192, 1), 'input')
            buf615 = reinterpret_tensor(buf587, (128, 8192), (8192, 1), 0); del buf587  # reuse
            # Topologically Sorted Source Nodes: [view_1279, down_proj_8, permute_1047, mm_466], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf613, (128, 3072), (3072, 1), 0), primals_84, out=buf615)
            del primals_84
            assert_size_stride(mm_60, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_61, (128, 8192), (8192, 1), 'input')
            buf616 = buf584; del buf584  # reuse
            buf619 = reinterpret_tensor(mm_61, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_61  # reuse
            # Topologically Sorted Source Nodes: [view_1280, linear_60, silu_8, mul_827, linear_61, mul_828, convert_element_type_1607, reciprocal_19, mul_829, mul_830, sub_52, mul_831, add_475, mul_832, convert_element_type_1609], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf619, buf615, mm_60, buf616, 1048576, stream=raw_stream0)
            del buf615
            del mm_60
            assert_size_stride(view_239, (128, 3072), (3072, 1), 'input')
            buf617 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1280, linear_60, silu_8, mul_827, view_1281, permute_1049, mm_467], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf616, (8192, 128), (1, 8192), 0), view_239, out=buf617)
            assert_size_stride(primals_83, (8192, 3072), (3072, 1), 'input')
            buf618 = buf610; del buf610  # reuse
            # Topologically Sorted Source Nodes: [view_1280, linear_60, silu_8, mul_827, view_1281, linear_61, permute_1051, mm_468], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf616, (128, 8192), (8192, 1), 0), primals_83, out=buf618)
            del primals_83
            buf620 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1280, linear_60, silu_8, linear_61, mul_828, convert_element_type_1607, reciprocal_19, mul_829, mul_830, sub_52, mul_831, add_475, mul_832, convert_element_type_1609, view_1283, permute_1053, mm_469], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf619, (8192, 128), (1, 8192), 0), view_239, out=buf620)
            del view_239
            assert_size_stride(primals_82, (8192, 3072), (3072, 1), 'input')
            buf621 = buf607; del buf607  # reuse
            # Topologically Sorted Source Nodes: [view_1280, linear_60, silu_8, linear_61, mul_828, convert_element_type_1607, reciprocal_19, mul_829, mul_830, sub_52, mul_831, add_475, mul_832, convert_element_type_1609, view_1283, permute_1055, mm_470], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf619, (128, 8192), (8192, 1), 0), primals_82, out=buf621)
            del primals_82
            assert_size_stride(primals_81, (3072, ), (1, ), 'input')
            assert_size_stride(add_69, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_17, (1, 128, 1), (128, 1, 1), 'input')
            buf624 = buf613; del buf613  # reuse
            # Topologically Sorted Source Nodes: [view_1282, view_1284, add_476, mul_833, hidden_states_86, convert_element_type_1614, mul_835, mul_836, sum_169, pow_136, mul_837, mul_838, expand_212, div_97, pow_137, mul_839, mul_840, add_477, convert_element_type_1615, add_478], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf624, buf618, buf621, primals_81, add_69, rsqrt_17, 128, 3072, stream=raw_stream0)
            del primals_81
            buf622 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1282, view_1284, add_476, hidden_states_86, hidden_states_87, to_49, mul_834, sum_168], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf618, buf621, add_69, rsqrt_17, buf622, 3072, 128, stream=raw_stream0)
            del add_69
            del rsqrt_17
            assert_size_stride(view_237, (128, 3072), (3072, 1), 'input')
            buf625 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1286, permute_1057, mm_471], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf624, (3072, 128), (1, 3072), 0), view_237, out=buf625)
            del view_237
            assert_size_stride(primals_80, (3072, 3072), (3072, 1), 'input')
            buf626 = buf621; del buf621  # reuse
            # Topologically Sorted Source Nodes: [view_1286, attn_output_35, permute_1059, mm_472], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf624, (128, 3072), (3072, 1), 0), primals_80, out=buf626)
            del primals_80
            assert_size_stride(permute_1062, (24, 128, 128), (16384, 1, 128), 'input')
            buf627 = reinterpret_tensor(buf618, (24, 128, 128), (16384, 128, 1), 0); del buf618  # reuse
            # Topologically Sorted Source Nodes: [view_1287, view_1288, permute_1061, view_1289, bmm_132], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1062, reinterpret_tensor(buf626, (24, 128, 128), (128, 3072, 1), 0), out=buf627)
            del permute_1062
            assert_size_stride(permute_1063, (24, 128, 128), (16384, 1, 128), 'input')
            buf628 = reinterpret_tensor(buf604, (24, 128, 128), (16384, 128, 1), 0); del buf604  # reuse
            # Topologically Sorted Source Nodes: [view_1287, view_1288, permute_1061, view_1289, bmm_133], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf626, (24, 128, 128), (128, 3072, 1), 0), permute_1063, out=buf628)
            del permute_1063
            buf634 = reinterpret_tensor(buf605, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf605  # reuse
            # Topologically Sorted Source Nodes: [view_1290, view_1295, sum_171, squeeze_39, permute_1067, clone_171], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf627, buf634, 131072, stream=raw_stream0)
            assert_size_stride(add_68, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_8, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_9, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf630 = add_68; del add_68  # reuse
            # Topologically Sorted Source Nodes: [view_1291, convert_element_type_1624, softmax_8, mul_841, sum_170, neg_162, fma_19, convert_element_type_1625, mul_842], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf630, buf628, amax_8, sum_9, 3072, 128, stream=raw_stream0)
            del amax_8
            del sum_9
            assert_size_stride(view_219, (128, 3072), (3072, 1), 'input')
            buf635 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1290, view_1295, sum_171, squeeze_39, permute_1067, clone_171, view_1297, view_1298, permute_1068, mm_473], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf634, (1024, 128), (1, 1024), 0), view_219, out=buf635)
            assert_size_stride(primals_79, (1024, 3072), (3072, 1), 'input')
            buf636 = reinterpret_tensor(buf628, (128, 3072), (3072, 1), 0); del buf628  # reuse
            # Topologically Sorted Source Nodes: [view_1290, view_1295, sum_171, squeeze_39, permute_1067, clone_171, view_1297, view_1298, linear_58, permute_1070, mm_474], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf634, (128, 1024), (1024, 1), 0), primals_79, out=buf636)
            del primals_79
            assert_size_stride(permute_1064, (24, 128, 128), (128, 1, 3072), 'input')
            buf631 = buf627; del buf627  # reuse
            # Topologically Sorted Source Nodes: [view_1291, convert_element_type_1624, softmax_8, mul_841, neg_162, fma_19, convert_element_type_1625, mul_842, view_1292, bmm_134], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1064, reinterpret_tensor(buf630, (24, 128, 128), (16384, 128, 1), 0), out=buf631)
            del permute_1064
            assert_size_stride(permute_1065, (24, 128, 128), (16384, 128, 1), 'input')
            buf632 = reinterpret_tensor(buf626, (24, 128, 128), (16384, 128, 1), 0); del buf626  # reuse
            # Topologically Sorted Source Nodes: [view_1291, convert_element_type_1624, softmax_8, mul_841, neg_162, fma_19, convert_element_type_1625, mul_842, view_1292, bmm_135], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf630, (24, 128, 128), (16384, 128, 1), 0), permute_1065, out=buf632)
            del buf630
            del permute_1065
            buf633 = reinterpret_tensor(buf634, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf634  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1293, permute_1066, view_1296, sum_172, squeeze_40, mul_843, slice_193, slice_194, neg_163, add_479, mul_844, add_480], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf631, primals_3, buf633, 131072, stream=raw_stream0)
            buf640 = reinterpret_tensor(buf631, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf631  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1294, mul_845, slice_195, slice_196, neg_164, add_481, mul_846, add_482, permute_1077, clone_173], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf632, primals_3, buf640, 393216, stream=raw_stream0)
            buf637 = reinterpret_tensor(buf601, (128, 1024), (1024, 1), 0); del buf601  # reuse
            # Topologically Sorted Source Nodes: [permute_1072, clone_172, view_1300, view_1301], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf633, buf637, 128, 1024, stream=raw_stream0)
            buf641 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1294, mul_845, slice_195, slice_196, neg_164, add_481, mul_846, add_482, permute_1077, clone_173, view_1303, view_1304, permute_1078, mm_477], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf640, (3072, 128), (1, 3072), 0), view_219, out=buf641)
            assert_size_stride(primals_77, (3072, 3072), (3072, 1), 'input')
            buf642 = reinterpret_tensor(buf632, (128, 3072), (3072, 1), 0); del buf632  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1294, mul_845, slice_195, slice_196, neg_164, add_481, mul_846, add_482, permute_1077, clone_173, view_1303, view_1304, linear_56, permute_1080, mm_478], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf640, (128, 3072), (3072, 1), 0), primals_77, out=buf642)
            del primals_77
            buf638 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1073, mm_475], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf637, (1024, 128), (1, 1024), 0), view_219, out=buf638)
            del view_219
            assert_size_stride(primals_78, (1024, 3072), (3072, 1), 'input')
            buf639 = reinterpret_tensor(buf640, (128, 3072), (3072, 1), 0); del buf640  # reuse
            # Topologically Sorted Source Nodes: [linear_57, permute_1075, mm_476], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf637, primals_78, out=buf639)
            del primals_78
            assert_size_stride(primals_76, (3072, ), (1, ), 'input')
            assert_size_stride(add_64, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_16, (1, 128, 1), (128, 1, 1), 'input')
            buf645 = buf624; del buf624  # reuse
            # Topologically Sorted Source Nodes: [view_1299, view_1302, add_483, view_1305, add_484, mul_847, hidden_states_80, convert_element_type_1642, mul_849, mul_850, sum_174, pow_138, mul_851, mul_852, expand_213, div_98, pow_139, mul_853, mul_854, add_485, convert_element_type_1643, add_486], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf645, buf636, buf639, buf642, primals_76, add_64, rsqrt_16, 128, 3072, stream=raw_stream0)
            del primals_76
            buf643 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1299, view_1302, add_483, view_1305, add_484, hidden_states_80, hidden_states_81, to_46, mul_848, sum_173], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf636, buf639, buf642, add_64, rsqrt_16, buf643, 3072, 128, stream=raw_stream0)
            del add_64
            del rsqrt_16
            assert_size_stride(view_217, (128, 8192), (8192, 1), 'input')
            buf646 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1307, permute_1082, mm_479], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf645, (3072, 128), (1, 3072), 0), view_217, out=buf646)
            del view_217
            assert_size_stride(primals_75, (3072, 8192), (8192, 1), 'input')
            buf647 = reinterpret_tensor(buf619, (128, 8192), (8192, 1), 0); del buf619  # reuse
            # Topologically Sorted Source Nodes: [view_1307, down_proj_7, permute_1084, mm_480], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf645, (128, 3072), (3072, 1), 0), primals_75, out=buf647)
            del primals_75
            assert_size_stride(mm_53, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_54, (128, 8192), (8192, 1), 'input')
            buf648 = buf616; del buf616  # reuse
            buf651 = reinterpret_tensor(mm_54, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_54  # reuse
            # Topologically Sorted Source Nodes: [view_1308, linear_53, silu_7, mul_855, linear_54, mul_856, convert_element_type_1652, reciprocal_20, mul_857, mul_858, sub_53, mul_859, add_488, mul_860, convert_element_type_1654], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf651, buf647, mm_53, buf648, 1048576, stream=raw_stream0)
            del buf647
            del mm_53
            assert_size_stride(view_213, (128, 3072), (3072, 1), 'input')
            buf649 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1308, linear_53, silu_7, mul_855, view_1309, permute_1086, mm_481], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf648, (8192, 128), (1, 8192), 0), view_213, out=buf649)
            assert_size_stride(primals_74, (8192, 3072), (3072, 1), 'input')
            buf650 = buf642; del buf642  # reuse
            # Topologically Sorted Source Nodes: [view_1308, linear_53, silu_7, mul_855, view_1309, linear_54, permute_1088, mm_482], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf648, (128, 8192), (8192, 1), 0), primals_74, out=buf650)
            del primals_74
            buf652 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1308, linear_53, silu_7, linear_54, mul_856, convert_element_type_1652, reciprocal_20, mul_857, mul_858, sub_53, mul_859, add_488, mul_860, convert_element_type_1654, view_1311, permute_1090, mm_483], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf651, (8192, 128), (1, 8192), 0), view_213, out=buf652)
            del view_213
            assert_size_stride(primals_73, (8192, 3072), (3072, 1), 'input')
            buf653 = buf639; del buf639  # reuse
            # Topologically Sorted Source Nodes: [view_1308, linear_53, silu_7, linear_54, mul_856, convert_element_type_1652, reciprocal_20, mul_857, mul_858, sub_53, mul_859, add_488, mul_860, convert_element_type_1654, view_1311, permute_1092, mm_484], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf651, (128, 8192), (8192, 1), 0), primals_73, out=buf653)
            del primals_73
            assert_size_stride(primals_72, (3072, ), (1, ), 'input')
            assert_size_stride(add_61, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_15, (1, 128, 1), (128, 1, 1), 'input')
            buf656 = buf645; del buf645  # reuse
            # Topologically Sorted Source Nodes: [view_1310, view_1312, add_489, mul_861, hidden_states_76, convert_element_type_1659, mul_863, mul_864, sum_176, pow_140, mul_865, mul_866, expand_214, div_99, pow_141, mul_867, mul_868, add_490, convert_element_type_1660, add_491], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf656, buf650, buf653, primals_72, add_61, rsqrt_15, 128, 3072, stream=raw_stream0)
            del primals_72
            buf654 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1310, view_1312, add_489, hidden_states_76, hidden_states_77, to_44, mul_862, sum_175], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf650, buf653, add_61, rsqrt_15, buf654, 3072, 128, stream=raw_stream0)
            del add_61
            del rsqrt_15
            assert_size_stride(view_211, (128, 3072), (3072, 1), 'input')
            buf657 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1314, permute_1094, mm_485], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf656, (3072, 128), (1, 3072), 0), view_211, out=buf657)
            del view_211
            assert_size_stride(primals_71, (3072, 3072), (3072, 1), 'input')
            buf658 = buf653; del buf653  # reuse
            # Topologically Sorted Source Nodes: [view_1314, attn_output_31, permute_1096, mm_486], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf656, (128, 3072), (3072, 1), 0), primals_71, out=buf658)
            del primals_71
            assert_size_stride(permute_1099, (24, 128, 128), (16384, 1, 128), 'input')
            buf659 = reinterpret_tensor(buf650, (24, 128, 128), (16384, 128, 1), 0); del buf650  # reuse
            # Topologically Sorted Source Nodes: [view_1315, view_1316, permute_1098, view_1317, bmm_136], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1099, reinterpret_tensor(buf658, (24, 128, 128), (128, 3072, 1), 0), out=buf659)
            del permute_1099
            assert_size_stride(permute_1100, (24, 128, 128), (16384, 1, 128), 'input')
            buf660 = reinterpret_tensor(buf636, (24, 128, 128), (16384, 128, 1), 0); del buf636  # reuse
            # Topologically Sorted Source Nodes: [view_1315, view_1316, permute_1098, view_1317, bmm_137], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf658, (24, 128, 128), (128, 3072, 1), 0), permute_1100, out=buf660)
            del permute_1100
            buf666 = reinterpret_tensor(buf637, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf637  # reuse
            # Topologically Sorted Source Nodes: [view_1318, view_1323, sum_178, squeeze_41, permute_1104, clone_174], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf659, buf666, 131072, stream=raw_stream0)
            assert_size_stride(add_60, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_7, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_8, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf662 = add_60; del add_60  # reuse
            # Topologically Sorted Source Nodes: [view_1319, convert_element_type_1669, softmax_7, mul_869, sum_177, neg_166, fma_20, convert_element_type_1670, mul_870], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf662, buf660, amax_7, sum_8, 3072, 128, stream=raw_stream0)
            del amax_7
            del sum_8
            assert_size_stride(view_193, (128, 3072), (3072, 1), 'input')
            buf667 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1318, view_1323, sum_178, squeeze_41, permute_1104, clone_174, view_1325, view_1326, permute_1105, mm_487], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf666, (1024, 128), (1, 1024), 0), view_193, out=buf667)
            assert_size_stride(primals_70, (1024, 3072), (3072, 1), 'input')
            buf668 = reinterpret_tensor(buf660, (128, 3072), (3072, 1), 0); del buf660  # reuse
            # Topologically Sorted Source Nodes: [view_1318, view_1323, sum_178, squeeze_41, permute_1104, clone_174, view_1325, view_1326, linear_51, permute_1107, mm_488], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf666, (128, 1024), (1024, 1), 0), primals_70, out=buf668)
            del primals_70
            assert_size_stride(permute_1101, (24, 128, 128), (128, 1, 3072), 'input')
            buf663 = buf659; del buf659  # reuse
            # Topologically Sorted Source Nodes: [view_1319, convert_element_type_1669, softmax_7, mul_869, neg_166, fma_20, convert_element_type_1670, mul_870, view_1320, bmm_138], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1101, reinterpret_tensor(buf662, (24, 128, 128), (16384, 128, 1), 0), out=buf663)
            del permute_1101
            assert_size_stride(permute_1102, (24, 128, 128), (16384, 128, 1), 'input')
            buf664 = reinterpret_tensor(buf658, (24, 128, 128), (16384, 128, 1), 0); del buf658  # reuse
            # Topologically Sorted Source Nodes: [view_1319, convert_element_type_1669, softmax_7, mul_869, neg_166, fma_20, convert_element_type_1670, mul_870, view_1320, bmm_139], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf662, (24, 128, 128), (16384, 128, 1), 0), permute_1102, out=buf664)
            del buf662
            del permute_1102
            buf665 = reinterpret_tensor(buf666, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf666  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1321, permute_1103, view_1324, sum_179, squeeze_42, mul_871, slice_197, slice_198, neg_167, add_492, mul_872, add_493], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf663, primals_3, buf665, 131072, stream=raw_stream0)
            buf672 = reinterpret_tensor(buf663, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf663  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1322, mul_873, slice_199, slice_200, neg_168, add_494, mul_874, add_495, permute_1114, clone_176], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf664, primals_3, buf672, 393216, stream=raw_stream0)
            buf669 = reinterpret_tensor(buf633, (128, 1024), (1024, 1), 0); del buf633  # reuse
            # Topologically Sorted Source Nodes: [permute_1109, clone_175, view_1328, view_1329], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf665, buf669, 128, 1024, stream=raw_stream0)
            buf673 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1322, mul_873, slice_199, slice_200, neg_168, add_494, mul_874, add_495, permute_1114, clone_176, view_1331, view_1332, permute_1115, mm_491], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf672, (3072, 128), (1, 3072), 0), view_193, out=buf673)
            assert_size_stride(primals_68, (3072, 3072), (3072, 1), 'input')
            buf674 = reinterpret_tensor(buf664, (128, 3072), (3072, 1), 0); del buf664  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1322, mul_873, slice_199, slice_200, neg_168, add_494, mul_874, add_495, permute_1114, clone_176, view_1331, view_1332, linear_49, permute_1117, mm_492], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf672, (128, 3072), (3072, 1), 0), primals_68, out=buf674)
            del primals_68
            buf670 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1110, mm_489], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf669, (1024, 128), (1, 1024), 0), view_193, out=buf670)
            del view_193
            assert_size_stride(primals_69, (1024, 3072), (3072, 1), 'input')
            buf671 = reinterpret_tensor(buf672, (128, 3072), (3072, 1), 0); del buf672  # reuse
            # Topologically Sorted Source Nodes: [linear_50, permute_1112, mm_490], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf669, primals_69, out=buf671)
            del primals_69
            assert_size_stride(primals_67, (3072, ), (1, ), 'input')
            assert_size_stride(add_56, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_14, (1, 128, 1), (128, 1, 1), 'input')
            buf677 = buf656; del buf656  # reuse
            # Topologically Sorted Source Nodes: [view_1327, view_1330, add_496, view_1333, add_497, mul_875, hidden_states_70, convert_element_type_1687, mul_877, mul_878, sum_181, pow_142, mul_879, mul_880, expand_215, div_100, pow_143, mul_881, mul_882, add_498, convert_element_type_1688, add_499], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf677, buf668, buf671, buf674, primals_67, add_56, rsqrt_14, 128, 3072, stream=raw_stream0)
            del primals_67
            buf675 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1327, view_1330, add_496, view_1333, add_497, hidden_states_70, hidden_states_71, to_41, mul_876, sum_180], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf668, buf671, buf674, add_56, rsqrt_14, buf675, 3072, 128, stream=raw_stream0)
            del add_56
            del rsqrt_14
            assert_size_stride(view_191, (128, 8192), (8192, 1), 'input')
            buf678 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1335, permute_1119, mm_493], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf677, (3072, 128), (1, 3072), 0), view_191, out=buf678)
            del view_191
            assert_size_stride(primals_66, (3072, 8192), (8192, 1), 'input')
            buf679 = reinterpret_tensor(buf651, (128, 8192), (8192, 1), 0); del buf651  # reuse
            # Topologically Sorted Source Nodes: [view_1335, down_proj_6, permute_1121, mm_494], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf677, (128, 3072), (3072, 1), 0), primals_66, out=buf679)
            del primals_66
            assert_size_stride(mm_46, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_47, (128, 8192), (8192, 1), 'input')
            buf680 = buf648; del buf648  # reuse
            buf683 = reinterpret_tensor(mm_47, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_47  # reuse
            # Topologically Sorted Source Nodes: [view_1336, linear_46, silu_6, mul_883, linear_47, mul_884, convert_element_type_1697, reciprocal_21, mul_885, mul_886, sub_54, mul_887, add_501, mul_888, convert_element_type_1699], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf683, buf679, mm_46, buf680, 1048576, stream=raw_stream0)
            del buf679
            del mm_46
            assert_size_stride(view_187, (128, 3072), (3072, 1), 'input')
            buf681 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1336, linear_46, silu_6, mul_883, view_1337, permute_1123, mm_495], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf680, (8192, 128), (1, 8192), 0), view_187, out=buf681)
            assert_size_stride(primals_65, (8192, 3072), (3072, 1), 'input')
            buf682 = buf674; del buf674  # reuse
            # Topologically Sorted Source Nodes: [view_1336, linear_46, silu_6, mul_883, view_1337, linear_47, permute_1125, mm_496], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf680, (128, 8192), (8192, 1), 0), primals_65, out=buf682)
            del primals_65
            buf684 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1336, linear_46, silu_6, linear_47, mul_884, convert_element_type_1697, reciprocal_21, mul_885, mul_886, sub_54, mul_887, add_501, mul_888, convert_element_type_1699, view_1339, permute_1127, mm_497], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf683, (8192, 128), (1, 8192), 0), view_187, out=buf684)
            del view_187
            assert_size_stride(primals_64, (8192, 3072), (3072, 1), 'input')
            buf685 = buf671; del buf671  # reuse
            # Topologically Sorted Source Nodes: [view_1336, linear_46, silu_6, linear_47, mul_884, convert_element_type_1697, reciprocal_21, mul_885, mul_886, sub_54, mul_887, add_501, mul_888, convert_element_type_1699, view_1339, permute_1129, mm_498], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf683, (128, 8192), (8192, 1), 0), primals_64, out=buf685)
            del primals_64
            assert_size_stride(primals_63, (3072, ), (1, ), 'input')
            assert_size_stride(add_53, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_13, (1, 128, 1), (128, 1, 1), 'input')
            buf688 = buf677; del buf677  # reuse
            # Topologically Sorted Source Nodes: [view_1338, view_1340, add_502, mul_889, hidden_states_66, convert_element_type_1704, mul_891, mul_892, sum_183, pow_144, mul_893, mul_894, expand_216, div_101, pow_145, mul_895, mul_896, add_503, convert_element_type_1705, add_504], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf688, buf682, buf685, primals_63, add_53, rsqrt_13, 128, 3072, stream=raw_stream0)
            del primals_63
            buf686 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1338, view_1340, add_502, hidden_states_66, hidden_states_67, to_39, mul_890, sum_182], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf682, buf685, add_53, rsqrt_13, buf686, 3072, 128, stream=raw_stream0)
            del add_53
            del rsqrt_13
            assert_size_stride(view_185, (128, 3072), (3072, 1), 'input')
            buf689 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1342, permute_1131, mm_499], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf688, (3072, 128), (1, 3072), 0), view_185, out=buf689)
            del view_185
            assert_size_stride(primals_62, (3072, 3072), (3072, 1), 'input')
            buf690 = buf685; del buf685  # reuse
            # Topologically Sorted Source Nodes: [view_1342, attn_output_27, permute_1133, mm_500], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf688, (128, 3072), (3072, 1), 0), primals_62, out=buf690)
            del primals_62
            assert_size_stride(permute_1136, (24, 128, 128), (16384, 1, 128), 'input')
            buf691 = reinterpret_tensor(buf682, (24, 128, 128), (16384, 128, 1), 0); del buf682  # reuse
            # Topologically Sorted Source Nodes: [view_1343, view_1344, permute_1135, view_1345, bmm_140], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1136, reinterpret_tensor(buf690, (24, 128, 128), (128, 3072, 1), 0), out=buf691)
            del permute_1136
            assert_size_stride(permute_1137, (24, 128, 128), (16384, 1, 128), 'input')
            buf692 = reinterpret_tensor(buf668, (24, 128, 128), (16384, 128, 1), 0); del buf668  # reuse
            # Topologically Sorted Source Nodes: [view_1343, view_1344, permute_1135, view_1345, bmm_141], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf690, (24, 128, 128), (128, 3072, 1), 0), permute_1137, out=buf692)
            del permute_1137
            buf698 = reinterpret_tensor(buf669, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf669  # reuse
            # Topologically Sorted Source Nodes: [view_1346, view_1351, sum_185, squeeze_43, permute_1141, clone_177], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf691, buf698, 131072, stream=raw_stream0)
            assert_size_stride(add_52, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_6, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_7, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf694 = add_52; del add_52  # reuse
            # Topologically Sorted Source Nodes: [view_1347, convert_element_type_1714, softmax_6, mul_897, sum_184, neg_170, fma_21, convert_element_type_1715, mul_898], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf694, buf692, amax_6, sum_7, 3072, 128, stream=raw_stream0)
            del amax_6
            del sum_7
            assert_size_stride(view_167, (128, 3072), (3072, 1), 'input')
            buf699 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1346, view_1351, sum_185, squeeze_43, permute_1141, clone_177, view_1353, view_1354, permute_1142, mm_501], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf698, (1024, 128), (1, 1024), 0), view_167, out=buf699)
            assert_size_stride(primals_61, (1024, 3072), (3072, 1), 'input')
            buf700 = reinterpret_tensor(buf692, (128, 3072), (3072, 1), 0); del buf692  # reuse
            # Topologically Sorted Source Nodes: [view_1346, view_1351, sum_185, squeeze_43, permute_1141, clone_177, view_1353, view_1354, linear_44, permute_1144, mm_502], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf698, (128, 1024), (1024, 1), 0), primals_61, out=buf700)
            del primals_61
            assert_size_stride(permute_1138, (24, 128, 128), (128, 1, 3072), 'input')
            buf695 = buf691; del buf691  # reuse
            # Topologically Sorted Source Nodes: [view_1347, convert_element_type_1714, softmax_6, mul_897, neg_170, fma_21, convert_element_type_1715, mul_898, view_1348, bmm_142], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1138, reinterpret_tensor(buf694, (24, 128, 128), (16384, 128, 1), 0), out=buf695)
            del permute_1138
            assert_size_stride(permute_1139, (24, 128, 128), (16384, 128, 1), 'input')
            buf696 = reinterpret_tensor(buf690, (24, 128, 128), (16384, 128, 1), 0); del buf690  # reuse
            # Topologically Sorted Source Nodes: [view_1347, convert_element_type_1714, softmax_6, mul_897, neg_170, fma_21, convert_element_type_1715, mul_898, view_1348, bmm_143], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf694, (24, 128, 128), (16384, 128, 1), 0), permute_1139, out=buf696)
            del buf694
            del permute_1139
            buf697 = reinterpret_tensor(buf698, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf698  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1349, permute_1140, view_1352, sum_186, squeeze_44, mul_899, slice_201, slice_202, neg_171, add_505, mul_900, add_506], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf695, primals_3, buf697, 131072, stream=raw_stream0)
            buf704 = reinterpret_tensor(buf695, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf695  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1350, mul_901, slice_203, slice_204, neg_172, add_507, mul_902, add_508, permute_1151, clone_179], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf696, primals_3, buf704, 393216, stream=raw_stream0)
            buf701 = reinterpret_tensor(buf665, (128, 1024), (1024, 1), 0); del buf665  # reuse
            # Topologically Sorted Source Nodes: [permute_1146, clone_178, view_1356, view_1357], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf697, buf701, 128, 1024, stream=raw_stream0)
            buf705 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1350, mul_901, slice_203, slice_204, neg_172, add_507, mul_902, add_508, permute_1151, clone_179, view_1359, view_1360, permute_1152, mm_505], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf704, (3072, 128), (1, 3072), 0), view_167, out=buf705)
            assert_size_stride(primals_59, (3072, 3072), (3072, 1), 'input')
            buf706 = reinterpret_tensor(buf696, (128, 3072), (3072, 1), 0); del buf696  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1350, mul_901, slice_203, slice_204, neg_172, add_507, mul_902, add_508, permute_1151, clone_179, view_1359, view_1360, linear_42, permute_1154, mm_506], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf704, (128, 3072), (3072, 1), 0), primals_59, out=buf706)
            del primals_59
            buf702 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1147, mm_503], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf701, (1024, 128), (1, 1024), 0), view_167, out=buf702)
            del view_167
            assert_size_stride(primals_60, (1024, 3072), (3072, 1), 'input')
            buf703 = reinterpret_tensor(buf704, (128, 3072), (3072, 1), 0); del buf704  # reuse
            # Topologically Sorted Source Nodes: [linear_43, permute_1149, mm_504], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf701, primals_60, out=buf703)
            del primals_60
            assert_size_stride(primals_58, (3072, ), (1, ), 'input')
            assert_size_stride(add_48, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_12, (1, 128, 1), (128, 1, 1), 'input')
            buf709 = buf688; del buf688  # reuse
            # Topologically Sorted Source Nodes: [view_1355, view_1358, add_509, view_1361, add_510, mul_903, hidden_states_60, convert_element_type_1732, mul_905, mul_906, sum_188, pow_146, mul_907, mul_908, expand_217, div_102, pow_147, mul_909, mul_910, add_511, convert_element_type_1733, add_512], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf709, buf700, buf703, buf706, primals_58, add_48, rsqrt_12, 128, 3072, stream=raw_stream0)
            del primals_58
            buf707 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1355, view_1358, add_509, view_1361, add_510, hidden_states_60, hidden_states_61, to_36, mul_904, sum_187], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf700, buf703, buf706, add_48, rsqrt_12, buf707, 3072, 128, stream=raw_stream0)
            del add_48
            del rsqrt_12
            assert_size_stride(view_165, (128, 8192), (8192, 1), 'input')
            buf710 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1363, permute_1156, mm_507], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf709, (3072, 128), (1, 3072), 0), view_165, out=buf710)
            del view_165
            assert_size_stride(primals_57, (3072, 8192), (8192, 1), 'input')
            buf711 = reinterpret_tensor(buf683, (128, 8192), (8192, 1), 0); del buf683  # reuse
            # Topologically Sorted Source Nodes: [view_1363, down_proj_5, permute_1158, mm_508], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf709, (128, 3072), (3072, 1), 0), primals_57, out=buf711)
            del primals_57
            assert_size_stride(mm_39, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_40, (128, 8192), (8192, 1), 'input')
            buf712 = buf680; del buf680  # reuse
            buf715 = reinterpret_tensor(mm_40, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_40  # reuse
            # Topologically Sorted Source Nodes: [view_1364, linear_39, silu_5, mul_911, linear_40, mul_912, convert_element_type_1742, reciprocal_22, mul_913, mul_914, sub_55, mul_915, add_514, mul_916, convert_element_type_1744], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf715, buf711, mm_39, buf712, 1048576, stream=raw_stream0)
            del buf711
            del mm_39
            assert_size_stride(view_161, (128, 3072), (3072, 1), 'input')
            buf713 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1364, linear_39, silu_5, mul_911, view_1365, permute_1160, mm_509], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf712, (8192, 128), (1, 8192), 0), view_161, out=buf713)
            assert_size_stride(primals_56, (8192, 3072), (3072, 1), 'input')
            buf714 = buf706; del buf706  # reuse
            # Topologically Sorted Source Nodes: [view_1364, linear_39, silu_5, mul_911, view_1365, linear_40, permute_1162, mm_510], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf712, (128, 8192), (8192, 1), 0), primals_56, out=buf714)
            del primals_56
            buf716 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1364, linear_39, silu_5, linear_40, mul_912, convert_element_type_1742, reciprocal_22, mul_913, mul_914, sub_55, mul_915, add_514, mul_916, convert_element_type_1744, view_1367, permute_1164, mm_511], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf715, (8192, 128), (1, 8192), 0), view_161, out=buf716)
            del view_161
            assert_size_stride(primals_55, (8192, 3072), (3072, 1), 'input')
            buf717 = buf703; del buf703  # reuse
            # Topologically Sorted Source Nodes: [view_1364, linear_39, silu_5, linear_40, mul_912, convert_element_type_1742, reciprocal_22, mul_913, mul_914, sub_55, mul_915, add_514, mul_916, convert_element_type_1744, view_1367, permute_1166, mm_512], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf715, (128, 8192), (8192, 1), 0), primals_55, out=buf717)
            del primals_55
            assert_size_stride(primals_54, (3072, ), (1, ), 'input')
            assert_size_stride(add_45, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_11, (1, 128, 1), (128, 1, 1), 'input')
            buf720 = buf709; del buf709  # reuse
            # Topologically Sorted Source Nodes: [view_1366, view_1368, add_515, mul_917, hidden_states_56, convert_element_type_1749, mul_919, mul_920, sum_190, pow_148, mul_921, mul_922, expand_218, div_103, pow_149, mul_923, mul_924, add_516, convert_element_type_1750, add_517], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf720, buf714, buf717, primals_54, add_45, rsqrt_11, 128, 3072, stream=raw_stream0)
            del primals_54
            buf718 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1366, view_1368, add_515, hidden_states_56, hidden_states_57, to_34, mul_918, sum_189], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf714, buf717, add_45, rsqrt_11, buf718, 3072, 128, stream=raw_stream0)
            del add_45
            del rsqrt_11
            assert_size_stride(view_159, (128, 3072), (3072, 1), 'input')
            buf721 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1370, permute_1168, mm_513], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf720, (3072, 128), (1, 3072), 0), view_159, out=buf721)
            del view_159
            assert_size_stride(primals_53, (3072, 3072), (3072, 1), 'input')
            buf722 = buf717; del buf717  # reuse
            # Topologically Sorted Source Nodes: [view_1370, attn_output_23, permute_1170, mm_514], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf720, (128, 3072), (3072, 1), 0), primals_53, out=buf722)
            del primals_53
            assert_size_stride(permute_1173, (24, 128, 128), (16384, 1, 128), 'input')
            buf723 = reinterpret_tensor(buf714, (24, 128, 128), (16384, 128, 1), 0); del buf714  # reuse
            # Topologically Sorted Source Nodes: [view_1371, view_1372, permute_1172, view_1373, bmm_144], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1173, reinterpret_tensor(buf722, (24, 128, 128), (128, 3072, 1), 0), out=buf723)
            del permute_1173
            assert_size_stride(permute_1174, (24, 128, 128), (16384, 1, 128), 'input')
            buf724 = reinterpret_tensor(buf700, (24, 128, 128), (16384, 128, 1), 0); del buf700  # reuse
            # Topologically Sorted Source Nodes: [view_1371, view_1372, permute_1172, view_1373, bmm_145], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf722, (24, 128, 128), (128, 3072, 1), 0), permute_1174, out=buf724)
            del permute_1174
            buf730 = reinterpret_tensor(buf701, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf701  # reuse
            # Topologically Sorted Source Nodes: [view_1374, view_1379, sum_192, squeeze_45, permute_1178, clone_180], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf723, buf730, 131072, stream=raw_stream0)
            assert_size_stride(add_44, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_5, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_6, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf726 = add_44; del add_44  # reuse
            # Topologically Sorted Source Nodes: [view_1375, convert_element_type_1759, softmax_5, mul_925, sum_191, neg_174, fma_22, convert_element_type_1760, mul_926], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf726, buf724, amax_5, sum_6, 3072, 128, stream=raw_stream0)
            del amax_5
            del sum_6
            assert_size_stride(view_141, (128, 3072), (3072, 1), 'input')
            buf731 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1374, view_1379, sum_192, squeeze_45, permute_1178, clone_180, view_1381, view_1382, permute_1179, mm_515], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf730, (1024, 128), (1, 1024), 0), view_141, out=buf731)
            assert_size_stride(primals_52, (1024, 3072), (3072, 1), 'input')
            buf732 = reinterpret_tensor(buf724, (128, 3072), (3072, 1), 0); del buf724  # reuse
            # Topologically Sorted Source Nodes: [view_1374, view_1379, sum_192, squeeze_45, permute_1178, clone_180, view_1381, view_1382, linear_37, permute_1181, mm_516], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf730, (128, 1024), (1024, 1), 0), primals_52, out=buf732)
            del primals_52
            assert_size_stride(permute_1175, (24, 128, 128), (128, 1, 3072), 'input')
            buf727 = buf723; del buf723  # reuse
            # Topologically Sorted Source Nodes: [view_1375, convert_element_type_1759, softmax_5, mul_925, neg_174, fma_22, convert_element_type_1760, mul_926, view_1376, bmm_146], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1175, reinterpret_tensor(buf726, (24, 128, 128), (16384, 128, 1), 0), out=buf727)
            del permute_1175
            assert_size_stride(permute_1176, (24, 128, 128), (16384, 128, 1), 'input')
            buf728 = reinterpret_tensor(buf722, (24, 128, 128), (16384, 128, 1), 0); del buf722  # reuse
            # Topologically Sorted Source Nodes: [view_1375, convert_element_type_1759, softmax_5, mul_925, neg_174, fma_22, convert_element_type_1760, mul_926, view_1376, bmm_147], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf726, (24, 128, 128), (16384, 128, 1), 0), permute_1176, out=buf728)
            del buf726
            del permute_1176
            buf729 = reinterpret_tensor(buf730, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf730  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1377, permute_1177, view_1380, sum_193, squeeze_46, mul_927, slice_205, slice_206, neg_175, add_518, mul_928, add_519], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf727, primals_3, buf729, 131072, stream=raw_stream0)
            buf736 = reinterpret_tensor(buf727, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf727  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1378, mul_929, slice_207, slice_208, neg_176, add_520, mul_930, add_521, permute_1188, clone_182], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf728, primals_3, buf736, 393216, stream=raw_stream0)
            buf733 = reinterpret_tensor(buf697, (128, 1024), (1024, 1), 0); del buf697  # reuse
            # Topologically Sorted Source Nodes: [permute_1183, clone_181, view_1384, view_1385], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf729, buf733, 128, 1024, stream=raw_stream0)
            buf737 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1378, mul_929, slice_207, slice_208, neg_176, add_520, mul_930, add_521, permute_1188, clone_182, view_1387, view_1388, permute_1189, mm_519], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf736, (3072, 128), (1, 3072), 0), view_141, out=buf737)
            assert_size_stride(primals_50, (3072, 3072), (3072, 1), 'input')
            buf738 = reinterpret_tensor(buf728, (128, 3072), (3072, 1), 0); del buf728  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1378, mul_929, slice_207, slice_208, neg_176, add_520, mul_930, add_521, permute_1188, clone_182, view_1387, view_1388, linear_35, permute_1191, mm_520], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf736, (128, 3072), (3072, 1), 0), primals_50, out=buf738)
            del primals_50
            buf734 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1184, mm_517], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf733, (1024, 128), (1, 1024), 0), view_141, out=buf734)
            del view_141
            assert_size_stride(primals_51, (1024, 3072), (3072, 1), 'input')
            buf735 = reinterpret_tensor(buf736, (128, 3072), (3072, 1), 0); del buf736  # reuse
            # Topologically Sorted Source Nodes: [linear_36, permute_1186, mm_518], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf733, primals_51, out=buf735)
            del primals_51
            assert_size_stride(primals_49, (3072, ), (1, ), 'input')
            assert_size_stride(add_40, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_10, (1, 128, 1), (128, 1, 1), 'input')
            buf741 = buf720; del buf720  # reuse
            # Topologically Sorted Source Nodes: [view_1383, view_1386, add_522, view_1389, add_523, mul_931, hidden_states_50, convert_element_type_1777, mul_933, mul_934, sum_195, pow_150, mul_935, mul_936, expand_219, div_104, pow_151, mul_937, mul_938, add_524, convert_element_type_1778, add_525], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf741, buf732, buf735, buf738, primals_49, add_40, rsqrt_10, 128, 3072, stream=raw_stream0)
            del primals_49
            buf739 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1383, view_1386, add_522, view_1389, add_523, hidden_states_50, hidden_states_51, to_31, mul_932, sum_194], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf732, buf735, buf738, add_40, rsqrt_10, buf739, 3072, 128, stream=raw_stream0)
            del add_40
            del rsqrt_10
            assert_size_stride(view_139, (128, 8192), (8192, 1), 'input')
            buf742 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1391, permute_1193, mm_521], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf741, (3072, 128), (1, 3072), 0), view_139, out=buf742)
            del view_139
            assert_size_stride(primals_48, (3072, 8192), (8192, 1), 'input')
            buf743 = reinterpret_tensor(buf715, (128, 8192), (8192, 1), 0); del buf715  # reuse
            # Topologically Sorted Source Nodes: [view_1391, down_proj_4, permute_1195, mm_522], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf741, (128, 3072), (3072, 1), 0), primals_48, out=buf743)
            del primals_48
            assert_size_stride(mm_32, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_33, (128, 8192), (8192, 1), 'input')
            buf744 = buf712; del buf712  # reuse
            buf747 = reinterpret_tensor(mm_33, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_33  # reuse
            # Topologically Sorted Source Nodes: [view_1392, linear_32, silu_4, mul_939, linear_33, mul_940, convert_element_type_1787, reciprocal_23, mul_941, mul_942, sub_56, mul_943, add_527, mul_944, convert_element_type_1789], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf747, buf743, mm_32, buf744, 1048576, stream=raw_stream0)
            del buf743
            del mm_32
            assert_size_stride(view_135, (128, 3072), (3072, 1), 'input')
            buf745 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1392, linear_32, silu_4, mul_939, view_1393, permute_1197, mm_523], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf744, (8192, 128), (1, 8192), 0), view_135, out=buf745)
            assert_size_stride(primals_47, (8192, 3072), (3072, 1), 'input')
            buf746 = buf738; del buf738  # reuse
            # Topologically Sorted Source Nodes: [view_1392, linear_32, silu_4, mul_939, view_1393, linear_33, permute_1199, mm_524], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf744, (128, 8192), (8192, 1), 0), primals_47, out=buf746)
            del primals_47
            buf748 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1392, linear_32, silu_4, linear_33, mul_940, convert_element_type_1787, reciprocal_23, mul_941, mul_942, sub_56, mul_943, add_527, mul_944, convert_element_type_1789, view_1395, permute_1201, mm_525], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf747, (8192, 128), (1, 8192), 0), view_135, out=buf748)
            del view_135
            assert_size_stride(primals_46, (8192, 3072), (3072, 1), 'input')
            buf749 = buf735; del buf735  # reuse
            # Topologically Sorted Source Nodes: [view_1392, linear_32, silu_4, linear_33, mul_940, convert_element_type_1787, reciprocal_23, mul_941, mul_942, sub_56, mul_943, add_527, mul_944, convert_element_type_1789, view_1395, permute_1203, mm_526], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf747, (128, 8192), (8192, 1), 0), primals_46, out=buf749)
            del primals_46
            assert_size_stride(primals_45, (3072, ), (1, ), 'input')
            assert_size_stride(add_37, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_9, (1, 128, 1), (128, 1, 1), 'input')
            buf752 = buf741; del buf741  # reuse
            # Topologically Sorted Source Nodes: [view_1394, view_1396, add_528, mul_945, hidden_states_46, convert_element_type_1794, mul_947, mul_948, sum_197, pow_152, mul_949, mul_950, expand_220, div_105, pow_153, mul_951, mul_952, add_529, convert_element_type_1795, add_530], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf752, buf746, buf749, primals_45, add_37, rsqrt_9, 128, 3072, stream=raw_stream0)
            del primals_45
            buf750 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1394, view_1396, add_528, hidden_states_46, hidden_states_47, to_29, mul_946, sum_196], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf746, buf749, add_37, rsqrt_9, buf750, 3072, 128, stream=raw_stream0)
            del add_37
            del rsqrt_9
            assert_size_stride(view_133, (128, 3072), (3072, 1), 'input')
            buf753 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1398, permute_1205, mm_527], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf752, (3072, 128), (1, 3072), 0), view_133, out=buf753)
            del view_133
            assert_size_stride(primals_44, (3072, 3072), (3072, 1), 'input')
            buf754 = buf749; del buf749  # reuse
            # Topologically Sorted Source Nodes: [view_1398, attn_output_19, permute_1207, mm_528], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf752, (128, 3072), (3072, 1), 0), primals_44, out=buf754)
            del primals_44
            assert_size_stride(permute_1210, (24, 128, 128), (16384, 1, 128), 'input')
            buf755 = reinterpret_tensor(buf746, (24, 128, 128), (16384, 128, 1), 0); del buf746  # reuse
            # Topologically Sorted Source Nodes: [view_1399, view_1400, permute_1209, view_1401, bmm_148], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1210, reinterpret_tensor(buf754, (24, 128, 128), (128, 3072, 1), 0), out=buf755)
            del permute_1210
            assert_size_stride(permute_1211, (24, 128, 128), (16384, 1, 128), 'input')
            buf756 = reinterpret_tensor(buf732, (24, 128, 128), (16384, 128, 1), 0); del buf732  # reuse
            # Topologically Sorted Source Nodes: [view_1399, view_1400, permute_1209, view_1401, bmm_149], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf754, (24, 128, 128), (128, 3072, 1), 0), permute_1211, out=buf756)
            del permute_1211
            buf762 = reinterpret_tensor(buf733, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf733  # reuse
            # Topologically Sorted Source Nodes: [view_1402, view_1407, sum_199, squeeze_47, permute_1215, clone_183], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf755, buf762, 131072, stream=raw_stream0)
            assert_size_stride(add_36, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_4, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_5, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf758 = add_36; del add_36  # reuse
            # Topologically Sorted Source Nodes: [view_1403, convert_element_type_1804, softmax_4, mul_953, sum_198, neg_178, fma_23, convert_element_type_1805, mul_954], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf758, buf756, amax_4, sum_5, 3072, 128, stream=raw_stream0)
            del amax_4
            del sum_5
            assert_size_stride(view_115, (128, 3072), (3072, 1), 'input')
            buf763 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1402, view_1407, sum_199, squeeze_47, permute_1215, clone_183, view_1409, view_1410, permute_1216, mm_529], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf762, (1024, 128), (1, 1024), 0), view_115, out=buf763)
            assert_size_stride(primals_43, (1024, 3072), (3072, 1), 'input')
            buf764 = reinterpret_tensor(buf756, (128, 3072), (3072, 1), 0); del buf756  # reuse
            # Topologically Sorted Source Nodes: [view_1402, view_1407, sum_199, squeeze_47, permute_1215, clone_183, view_1409, view_1410, linear_30, permute_1218, mm_530], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf762, (128, 1024), (1024, 1), 0), primals_43, out=buf764)
            del primals_43
            assert_size_stride(permute_1212, (24, 128, 128), (128, 1, 3072), 'input')
            buf759 = buf755; del buf755  # reuse
            # Topologically Sorted Source Nodes: [view_1403, convert_element_type_1804, softmax_4, mul_953, neg_178, fma_23, convert_element_type_1805, mul_954, view_1404, bmm_150], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1212, reinterpret_tensor(buf758, (24, 128, 128), (16384, 128, 1), 0), out=buf759)
            del permute_1212
            assert_size_stride(permute_1213, (24, 128, 128), (16384, 128, 1), 'input')
            buf760 = reinterpret_tensor(buf754, (24, 128, 128), (16384, 128, 1), 0); del buf754  # reuse
            # Topologically Sorted Source Nodes: [view_1403, convert_element_type_1804, softmax_4, mul_953, neg_178, fma_23, convert_element_type_1805, mul_954, view_1404, bmm_151], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf758, (24, 128, 128), (16384, 128, 1), 0), permute_1213, out=buf760)
            del buf758
            del permute_1213
            buf761 = reinterpret_tensor(buf762, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf762  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1405, permute_1214, view_1408, sum_200, squeeze_48, mul_955, slice_209, slice_210, neg_179, add_531, mul_956, add_532], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf759, primals_3, buf761, 131072, stream=raw_stream0)
            buf768 = reinterpret_tensor(buf759, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf759  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1406, mul_957, slice_211, slice_212, neg_180, add_533, mul_958, add_534, permute_1225, clone_185], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf760, primals_3, buf768, 393216, stream=raw_stream0)
            buf765 = reinterpret_tensor(buf729, (128, 1024), (1024, 1), 0); del buf729  # reuse
            # Topologically Sorted Source Nodes: [permute_1220, clone_184, view_1412, view_1413], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf761, buf765, 128, 1024, stream=raw_stream0)
            buf769 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1406, mul_957, slice_211, slice_212, neg_180, add_533, mul_958, add_534, permute_1225, clone_185, view_1415, view_1416, permute_1226, mm_533], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf768, (3072, 128), (1, 3072), 0), view_115, out=buf769)
            assert_size_stride(primals_41, (3072, 3072), (3072, 1), 'input')
            buf770 = reinterpret_tensor(buf760, (128, 3072), (3072, 1), 0); del buf760  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1406, mul_957, slice_211, slice_212, neg_180, add_533, mul_958, add_534, permute_1225, clone_185, view_1415, view_1416, linear_28, permute_1228, mm_534], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf768, (128, 3072), (3072, 1), 0), primals_41, out=buf770)
            del primals_41
            buf766 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1221, mm_531], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf765, (1024, 128), (1, 1024), 0), view_115, out=buf766)
            del view_115
            assert_size_stride(primals_42, (1024, 3072), (3072, 1), 'input')
            buf767 = reinterpret_tensor(buf768, (128, 3072), (3072, 1), 0); del buf768  # reuse
            # Topologically Sorted Source Nodes: [linear_29, permute_1223, mm_532], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf765, primals_42, out=buf767)
            del primals_42
            assert_size_stride(primals_40, (3072, ), (1, ), 'input')
            assert_size_stride(add_32, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_8, (1, 128, 1), (128, 1, 1), 'input')
            buf773 = buf752; del buf752  # reuse
            # Topologically Sorted Source Nodes: [view_1411, view_1414, add_535, view_1417, add_536, mul_959, hidden_states_40, convert_element_type_1822, mul_961, mul_962, sum_202, pow_154, mul_963, mul_964, expand_221, div_106, pow_155, mul_965, mul_966, add_537, convert_element_type_1823, add_538], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf773, buf764, buf767, buf770, primals_40, add_32, rsqrt_8, 128, 3072, stream=raw_stream0)
            del primals_40
            buf771 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1411, view_1414, add_535, view_1417, add_536, hidden_states_40, hidden_states_41, to_26, mul_960, sum_201], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf764, buf767, buf770, add_32, rsqrt_8, buf771, 3072, 128, stream=raw_stream0)
            del add_32
            del rsqrt_8
            assert_size_stride(view_113, (128, 8192), (8192, 1), 'input')
            buf774 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1419, permute_1230, mm_535], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf773, (3072, 128), (1, 3072), 0), view_113, out=buf774)
            del view_113
            assert_size_stride(primals_39, (3072, 8192), (8192, 1), 'input')
            buf775 = reinterpret_tensor(buf747, (128, 8192), (8192, 1), 0); del buf747  # reuse
            # Topologically Sorted Source Nodes: [view_1419, down_proj_3, permute_1232, mm_536], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf773, (128, 3072), (3072, 1), 0), primals_39, out=buf775)
            del primals_39
            assert_size_stride(mm_25, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_26, (128, 8192), (8192, 1), 'input')
            buf776 = buf744; del buf744  # reuse
            buf779 = reinterpret_tensor(mm_26, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_26  # reuse
            # Topologically Sorted Source Nodes: [view_1420, linear_25, silu_3, mul_967, linear_26, mul_968, convert_element_type_1832, reciprocal_24, mul_969, mul_970, sub_57, mul_971, add_540, mul_972, convert_element_type_1834], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf779, buf775, mm_25, buf776, 1048576, stream=raw_stream0)
            del buf775
            del mm_25
            assert_size_stride(view_109, (128, 3072), (3072, 1), 'input')
            buf777 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1420, linear_25, silu_3, mul_967, view_1421, permute_1234, mm_537], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf776, (8192, 128), (1, 8192), 0), view_109, out=buf777)
            assert_size_stride(primals_38, (8192, 3072), (3072, 1), 'input')
            buf778 = buf770; del buf770  # reuse
            # Topologically Sorted Source Nodes: [view_1420, linear_25, silu_3, mul_967, view_1421, linear_26, permute_1236, mm_538], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf776, (128, 8192), (8192, 1), 0), primals_38, out=buf778)
            del primals_38
            buf780 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1420, linear_25, silu_3, linear_26, mul_968, convert_element_type_1832, reciprocal_24, mul_969, mul_970, sub_57, mul_971, add_540, mul_972, convert_element_type_1834, view_1423, permute_1238, mm_539], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf779, (8192, 128), (1, 8192), 0), view_109, out=buf780)
            del view_109
            assert_size_stride(primals_37, (8192, 3072), (3072, 1), 'input')
            buf781 = buf767; del buf767  # reuse
            # Topologically Sorted Source Nodes: [view_1420, linear_25, silu_3, linear_26, mul_968, convert_element_type_1832, reciprocal_24, mul_969, mul_970, sub_57, mul_971, add_540, mul_972, convert_element_type_1834, view_1423, permute_1240, mm_540], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf779, (128, 8192), (8192, 1), 0), primals_37, out=buf781)
            del primals_37
            assert_size_stride(primals_36, (3072, ), (1, ), 'input')
            assert_size_stride(add_29, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_7, (1, 128, 1), (128, 1, 1), 'input')
            buf784 = buf773; del buf773  # reuse
            # Topologically Sorted Source Nodes: [view_1422, view_1424, add_541, mul_973, hidden_states_36, convert_element_type_1839, mul_975, mul_976, sum_204, pow_156, mul_977, mul_978, expand_222, div_107, pow_157, mul_979, mul_980, add_542, convert_element_type_1840, add_543], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf784, buf778, buf781, primals_36, add_29, rsqrt_7, 128, 3072, stream=raw_stream0)
            del primals_36
            buf782 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1422, view_1424, add_541, hidden_states_36, hidden_states_37, to_24, mul_974, sum_203], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf778, buf781, add_29, rsqrt_7, buf782, 3072, 128, stream=raw_stream0)
            del add_29
            del rsqrt_7
            assert_size_stride(view_107, (128, 3072), (3072, 1), 'input')
            buf785 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1426, permute_1242, mm_541], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf784, (3072, 128), (1, 3072), 0), view_107, out=buf785)
            del view_107
            assert_size_stride(primals_35, (3072, 3072), (3072, 1), 'input')
            buf786 = buf781; del buf781  # reuse
            # Topologically Sorted Source Nodes: [view_1426, attn_output_15, permute_1244, mm_542], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf784, (128, 3072), (3072, 1), 0), primals_35, out=buf786)
            del primals_35
            assert_size_stride(permute_1247, (24, 128, 128), (16384, 1, 128), 'input')
            buf787 = reinterpret_tensor(buf778, (24, 128, 128), (16384, 128, 1), 0); del buf778  # reuse
            # Topologically Sorted Source Nodes: [view_1427, view_1428, permute_1246, view_1429, bmm_152], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1247, reinterpret_tensor(buf786, (24, 128, 128), (128, 3072, 1), 0), out=buf787)
            del permute_1247
            assert_size_stride(permute_1248, (24, 128, 128), (16384, 1, 128), 'input')
            buf788 = reinterpret_tensor(buf764, (24, 128, 128), (16384, 128, 1), 0); del buf764  # reuse
            # Topologically Sorted Source Nodes: [view_1427, view_1428, permute_1246, view_1429, bmm_153], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf786, (24, 128, 128), (128, 3072, 1), 0), permute_1248, out=buf788)
            del permute_1248
            buf794 = reinterpret_tensor(buf765, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf765  # reuse
            # Topologically Sorted Source Nodes: [view_1430, view_1435, sum_206, squeeze_49, permute_1252, clone_186], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf787, buf794, 131072, stream=raw_stream0)
            assert_size_stride(add_28, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_3, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_4, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf790 = add_28; del add_28  # reuse
            # Topologically Sorted Source Nodes: [view_1431, convert_element_type_1849, softmax_3, mul_981, sum_205, neg_182, fma_24, convert_element_type_1850, mul_982], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf790, buf788, amax_3, sum_4, 3072, 128, stream=raw_stream0)
            del amax_3
            del sum_4
            assert_size_stride(view_89, (128, 3072), (3072, 1), 'input')
            buf795 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1430, view_1435, sum_206, squeeze_49, permute_1252, clone_186, view_1437, view_1438, permute_1253, mm_543], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf794, (1024, 128), (1, 1024), 0), view_89, out=buf795)
            assert_size_stride(primals_34, (1024, 3072), (3072, 1), 'input')
            buf796 = reinterpret_tensor(buf788, (128, 3072), (3072, 1), 0); del buf788  # reuse
            # Topologically Sorted Source Nodes: [view_1430, view_1435, sum_206, squeeze_49, permute_1252, clone_186, view_1437, view_1438, linear_23, permute_1255, mm_544], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf794, (128, 1024), (1024, 1), 0), primals_34, out=buf796)
            del primals_34
            assert_size_stride(permute_1249, (24, 128, 128), (128, 1, 3072), 'input')
            buf791 = buf787; del buf787  # reuse
            # Topologically Sorted Source Nodes: [view_1431, convert_element_type_1849, softmax_3, mul_981, neg_182, fma_24, convert_element_type_1850, mul_982, view_1432, bmm_154], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1249, reinterpret_tensor(buf790, (24, 128, 128), (16384, 128, 1), 0), out=buf791)
            del permute_1249
            assert_size_stride(permute_1250, (24, 128, 128), (16384, 128, 1), 'input')
            buf792 = reinterpret_tensor(buf786, (24, 128, 128), (16384, 128, 1), 0); del buf786  # reuse
            # Topologically Sorted Source Nodes: [view_1431, convert_element_type_1849, softmax_3, mul_981, neg_182, fma_24, convert_element_type_1850, mul_982, view_1432, bmm_155], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf790, (24, 128, 128), (16384, 128, 1), 0), permute_1250, out=buf792)
            del buf790
            del permute_1250
            buf793 = reinterpret_tensor(buf794, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf794  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1433, permute_1251, view_1436, sum_207, squeeze_50, mul_983, slice_213, slice_214, neg_183, add_544, mul_984, add_545], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf791, primals_3, buf793, 131072, stream=raw_stream0)
            buf800 = reinterpret_tensor(buf791, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf791  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1434, mul_985, slice_215, slice_216, neg_184, add_546, mul_986, add_547, permute_1262, clone_188], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf792, primals_3, buf800, 393216, stream=raw_stream0)
            buf797 = reinterpret_tensor(buf761, (128, 1024), (1024, 1), 0); del buf761  # reuse
            # Topologically Sorted Source Nodes: [permute_1257, clone_187, view_1440, view_1441], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf793, buf797, 128, 1024, stream=raw_stream0)
            buf801 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1434, mul_985, slice_215, slice_216, neg_184, add_546, mul_986, add_547, permute_1262, clone_188, view_1443, view_1444, permute_1263, mm_547], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf800, (3072, 128), (1, 3072), 0), view_89, out=buf801)
            assert_size_stride(primals_32, (3072, 3072), (3072, 1), 'input')
            buf802 = reinterpret_tensor(buf792, (128, 3072), (3072, 1), 0); del buf792  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1434, mul_985, slice_215, slice_216, neg_184, add_546, mul_986, add_547, permute_1262, clone_188, view_1443, view_1444, linear_21, permute_1265, mm_548], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf800, (128, 3072), (3072, 1), 0), primals_32, out=buf802)
            del primals_32
            buf798 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1258, mm_545], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf797, (1024, 128), (1, 1024), 0), view_89, out=buf798)
            del view_89
            assert_size_stride(primals_33, (1024, 3072), (3072, 1), 'input')
            buf799 = reinterpret_tensor(buf800, (128, 3072), (3072, 1), 0); del buf800  # reuse
            # Topologically Sorted Source Nodes: [linear_22, permute_1260, mm_546], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf797, primals_33, out=buf799)
            del primals_33
            assert_size_stride(primals_31, (3072, ), (1, ), 'input')
            assert_size_stride(add_24, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_6, (1, 128, 1), (128, 1, 1), 'input')
            buf805 = buf784; del buf784  # reuse
            # Topologically Sorted Source Nodes: [view_1439, view_1442, add_548, view_1445, add_549, mul_987, hidden_states_30, convert_element_type_1867, mul_989, mul_990, sum_209, pow_158, mul_991, mul_992, expand_223, div_108, pow_159, mul_993, mul_994, add_550, convert_element_type_1868, add_551], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf805, buf796, buf799, buf802, primals_31, add_24, rsqrt_6, 128, 3072, stream=raw_stream0)
            del primals_31
            buf803 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1439, view_1442, add_548, view_1445, add_549, hidden_states_30, hidden_states_31, to_21, mul_988, sum_208], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf796, buf799, buf802, add_24, rsqrt_6, buf803, 3072, 128, stream=raw_stream0)
            del add_24
            del rsqrt_6
            assert_size_stride(view_87, (128, 8192), (8192, 1), 'input')
            buf806 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1447, permute_1267, mm_549], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf805, (3072, 128), (1, 3072), 0), view_87, out=buf806)
            del view_87
            assert_size_stride(primals_30, (3072, 8192), (8192, 1), 'input')
            buf807 = reinterpret_tensor(buf779, (128, 8192), (8192, 1), 0); del buf779  # reuse
            # Topologically Sorted Source Nodes: [view_1447, down_proj_2, permute_1269, mm_550], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf805, (128, 3072), (3072, 1), 0), primals_30, out=buf807)
            del primals_30
            assert_size_stride(mm_18, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_19, (128, 8192), (8192, 1), 'input')
            buf808 = buf776; del buf776  # reuse
            buf811 = reinterpret_tensor(mm_19, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_19  # reuse
            # Topologically Sorted Source Nodes: [view_1448, linear_18, silu_2, mul_995, linear_19, mul_996, convert_element_type_1877, reciprocal_25, mul_997, mul_998, sub_58, mul_999, add_553, mul_1000, convert_element_type_1879], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf811, buf807, mm_18, buf808, 1048576, stream=raw_stream0)
            del buf807
            del mm_18
            assert_size_stride(view_83, (128, 3072), (3072, 1), 'input')
            buf809 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1448, linear_18, silu_2, mul_995, view_1449, permute_1271, mm_551], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf808, (8192, 128), (1, 8192), 0), view_83, out=buf809)
            assert_size_stride(primals_29, (8192, 3072), (3072, 1), 'input')
            buf810 = buf802; del buf802  # reuse
            # Topologically Sorted Source Nodes: [view_1448, linear_18, silu_2, mul_995, view_1449, linear_19, permute_1273, mm_552], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf808, (128, 8192), (8192, 1), 0), primals_29, out=buf810)
            del primals_29
            buf812 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1448, linear_18, silu_2, linear_19, mul_996, convert_element_type_1877, reciprocal_25, mul_997, mul_998, sub_58, mul_999, add_553, mul_1000, convert_element_type_1879, view_1451, permute_1275, mm_553], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf811, (8192, 128), (1, 8192), 0), view_83, out=buf812)
            del view_83
            assert_size_stride(primals_28, (8192, 3072), (3072, 1), 'input')
            buf813 = buf799; del buf799  # reuse
            # Topologically Sorted Source Nodes: [view_1448, linear_18, silu_2, linear_19, mul_996, convert_element_type_1877, reciprocal_25, mul_997, mul_998, sub_58, mul_999, add_553, mul_1000, convert_element_type_1879, view_1451, permute_1277, mm_554], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf811, (128, 8192), (8192, 1), 0), primals_28, out=buf813)
            del primals_28
            assert_size_stride(primals_27, (3072, ), (1, ), 'input')
            assert_size_stride(add_21, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_5, (1, 128, 1), (128, 1, 1), 'input')
            buf816 = buf805; del buf805  # reuse
            # Topologically Sorted Source Nodes: [view_1450, view_1452, add_554, mul_1001, hidden_states_26, convert_element_type_1884, mul_1003, mul_1004, sum_211, pow_160, mul_1005, mul_1006, expand_224, div_109, pow_161, mul_1007, mul_1008, add_555, convert_element_type_1885, add_556], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf816, buf810, buf813, primals_27, add_21, rsqrt_5, 128, 3072, stream=raw_stream0)
            del primals_27
            buf814 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1450, view_1452, add_554, hidden_states_26, hidden_states_27, to_19, mul_1002, sum_210], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf810, buf813, add_21, rsqrt_5, buf814, 3072, 128, stream=raw_stream0)
            del add_21
            del rsqrt_5
            assert_size_stride(view_81, (128, 3072), (3072, 1), 'input')
            buf817 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1454, permute_1279, mm_555], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf816, (3072, 128), (1, 3072), 0), view_81, out=buf817)
            del view_81
            assert_size_stride(primals_26, (3072, 3072), (3072, 1), 'input')
            buf818 = buf813; del buf813  # reuse
            # Topologically Sorted Source Nodes: [view_1454, attn_output_11, permute_1281, mm_556], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf816, (128, 3072), (3072, 1), 0), primals_26, out=buf818)
            del primals_26
            assert_size_stride(permute_1284, (24, 128, 128), (16384, 1, 128), 'input')
            buf819 = reinterpret_tensor(buf810, (24, 128, 128), (16384, 128, 1), 0); del buf810  # reuse
            # Topologically Sorted Source Nodes: [view_1455, view_1456, permute_1283, view_1457, bmm_156], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1284, reinterpret_tensor(buf818, (24, 128, 128), (128, 3072, 1), 0), out=buf819)
            del permute_1284
            assert_size_stride(permute_1285, (24, 128, 128), (16384, 1, 128), 'input')
            buf820 = reinterpret_tensor(buf796, (24, 128, 128), (16384, 128, 1), 0); del buf796  # reuse
            # Topologically Sorted Source Nodes: [view_1455, view_1456, permute_1283, view_1457, bmm_157], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf818, (24, 128, 128), (128, 3072, 1), 0), permute_1285, out=buf820)
            del permute_1285
            buf826 = reinterpret_tensor(buf797, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf797  # reuse
            # Topologically Sorted Source Nodes: [view_1458, view_1463, sum_213, squeeze_51, permute_1289, clone_189], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf819, buf826, 131072, stream=raw_stream0)
            assert_size_stride(add_20, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_2, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_3, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf822 = add_20; del add_20  # reuse
            # Topologically Sorted Source Nodes: [view_1459, convert_element_type_1894, softmax_2, mul_1009, sum_212, neg_186, fma_25, convert_element_type_1895, mul_1010], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf822, buf820, amax_2, sum_3, 3072, 128, stream=raw_stream0)
            del amax_2
            del sum_3
            assert_size_stride(view_63, (128, 3072), (3072, 1), 'input')
            buf827 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1458, view_1463, sum_213, squeeze_51, permute_1289, clone_189, view_1465, view_1466, permute_1290, mm_557], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf826, (1024, 128), (1, 1024), 0), view_63, out=buf827)
            assert_size_stride(primals_25, (1024, 3072), (3072, 1), 'input')
            buf828 = reinterpret_tensor(buf820, (128, 3072), (3072, 1), 0); del buf820  # reuse
            # Topologically Sorted Source Nodes: [view_1458, view_1463, sum_213, squeeze_51, permute_1289, clone_189, view_1465, view_1466, linear_16, permute_1292, mm_558], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf826, (128, 1024), (1024, 1), 0), primals_25, out=buf828)
            del primals_25
            assert_size_stride(permute_1286, (24, 128, 128), (128, 1, 3072), 'input')
            buf823 = buf819; del buf819  # reuse
            # Topologically Sorted Source Nodes: [view_1459, convert_element_type_1894, softmax_2, mul_1009, neg_186, fma_25, convert_element_type_1895, mul_1010, view_1460, bmm_158], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1286, reinterpret_tensor(buf822, (24, 128, 128), (16384, 128, 1), 0), out=buf823)
            del permute_1286
            assert_size_stride(permute_1287, (24, 128, 128), (16384, 128, 1), 'input')
            buf824 = reinterpret_tensor(buf818, (24, 128, 128), (16384, 128, 1), 0); del buf818  # reuse
            # Topologically Sorted Source Nodes: [view_1459, convert_element_type_1894, softmax_2, mul_1009, neg_186, fma_25, convert_element_type_1895, mul_1010, view_1460, bmm_159], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf822, (24, 128, 128), (16384, 128, 1), 0), permute_1287, out=buf824)
            del permute_1287
            buf825 = reinterpret_tensor(buf826, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf826  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1461, permute_1288, view_1464, sum_214, squeeze_52, mul_1011, slice_217, slice_218, neg_187, add_557, mul_1012, add_558], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf823, primals_3, buf825, 131072, stream=raw_stream0)
            buf832 = reinterpret_tensor(buf823, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf823  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1462, mul_1013, slice_219, slice_220, neg_188, add_559, mul_1014, add_560, permute_1299, clone_191], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf824, primals_3, buf832, 393216, stream=raw_stream0)
            buf829 = reinterpret_tensor(buf793, (128, 1024), (1024, 1), 0); del buf793  # reuse
            # Topologically Sorted Source Nodes: [permute_1294, clone_190, view_1468, view_1469], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf825, buf829, 128, 1024, stream=raw_stream0)
            buf833 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1462, mul_1013, slice_219, slice_220, neg_188, add_559, mul_1014, add_560, permute_1299, clone_191, view_1471, view_1472, permute_1300, mm_561], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf832, (3072, 128), (1, 3072), 0), view_63, out=buf833)
            assert_size_stride(primals_23, (3072, 3072), (3072, 1), 'input')
            buf834 = reinterpret_tensor(buf824, (128, 3072), (3072, 1), 0); del buf824  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1462, mul_1013, slice_219, slice_220, neg_188, add_559, mul_1014, add_560, permute_1299, clone_191, view_1471, view_1472, linear_14, permute_1302, mm_562], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf832, (128, 3072), (3072, 1), 0), primals_23, out=buf834)
            del primals_23
            buf830 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1295, mm_559], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf829, (1024, 128), (1, 1024), 0), view_63, out=buf830)
            del view_63
            assert_size_stride(primals_24, (1024, 3072), (3072, 1), 'input')
            buf831 = reinterpret_tensor(buf832, (128, 3072), (3072, 1), 0); del buf832  # reuse
            # Topologically Sorted Source Nodes: [linear_15, permute_1297, mm_560], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf829, primals_24, out=buf831)
            del primals_24
            assert_size_stride(primals_22, (3072, ), (1, ), 'input')
            assert_size_stride(add_16, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_4, (1, 128, 1), (128, 1, 1), 'input')
            buf837 = buf816; del buf816  # reuse
            # Topologically Sorted Source Nodes: [view_1467, view_1470, add_561, view_1473, add_562, mul_1015, hidden_states_20, convert_element_type_1912, mul_1017, mul_1018, sum_216, pow_162, mul_1019, mul_1020, expand_225, div_110, pow_163, mul_1021, mul_1022, add_563, convert_element_type_1913, add_564], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf837, buf828, buf831, buf834, primals_22, add_16, rsqrt_4, 128, 3072, stream=raw_stream0)
            del primals_22
            buf835 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1467, view_1470, add_561, view_1473, add_562, hidden_states_20, hidden_states_21, to_16, mul_1016, sum_215], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf828, buf831, buf834, add_16, rsqrt_4, buf835, 3072, 128, stream=raw_stream0)
            del add_16
            del rsqrt_4
            assert_size_stride(view_61, (128, 8192), (8192, 1), 'input')
            buf838 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1475, permute_1304, mm_563], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf837, (3072, 128), (1, 3072), 0), view_61, out=buf838)
            del view_61
            assert_size_stride(primals_21, (3072, 8192), (8192, 1), 'input')
            buf839 = reinterpret_tensor(buf811, (128, 8192), (8192, 1), 0); del buf811  # reuse
            # Topologically Sorted Source Nodes: [view_1475, down_proj_1, permute_1306, mm_564], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf837, (128, 3072), (3072, 1), 0), primals_21, out=buf839)
            del primals_21
            assert_size_stride(mm_11, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_12, (128, 8192), (8192, 1), 'input')
            buf840 = buf808; del buf808  # reuse
            buf843 = reinterpret_tensor(mm_12, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_12  # reuse
            # Topologically Sorted Source Nodes: [view_1476, linear_11, silu_1, mul_1023, linear_12, mul_1024, convert_element_type_1922, reciprocal_26, mul_1025, mul_1026, sub_59, mul_1027, add_566, mul_1028, convert_element_type_1924], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf843, buf839, mm_11, buf840, 1048576, stream=raw_stream0)
            del buf839
            del mm_11
            assert_size_stride(view_57, (128, 3072), (3072, 1), 'input')
            buf841 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1476, linear_11, silu_1, mul_1023, view_1477, permute_1308, mm_565], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf840, (8192, 128), (1, 8192), 0), view_57, out=buf841)
            assert_size_stride(primals_20, (8192, 3072), (3072, 1), 'input')
            buf842 = buf834; del buf834  # reuse
            # Topologically Sorted Source Nodes: [view_1476, linear_11, silu_1, mul_1023, view_1477, linear_12, permute_1310, mm_566], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf840, (128, 8192), (8192, 1), 0), primals_20, out=buf842)
            del primals_20
            buf844 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1476, linear_11, silu_1, linear_12, mul_1024, convert_element_type_1922, reciprocal_26, mul_1025, mul_1026, sub_59, mul_1027, add_566, mul_1028, convert_element_type_1924, view_1479, permute_1312, mm_567], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf843, (8192, 128), (1, 8192), 0), view_57, out=buf844)
            del view_57
            assert_size_stride(primals_19, (8192, 3072), (3072, 1), 'input')
            buf845 = buf831; del buf831  # reuse
            # Topologically Sorted Source Nodes: [view_1476, linear_11, silu_1, linear_12, mul_1024, convert_element_type_1922, reciprocal_26, mul_1025, mul_1026, sub_59, mul_1027, add_566, mul_1028, convert_element_type_1924, view_1479, permute_1314, mm_568], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf843, (128, 8192), (8192, 1), 0), primals_19, out=buf845)
            del primals_19
            assert_size_stride(primals_18, (3072, ), (1, ), 'input')
            assert_size_stride(add_13, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_3, (1, 128, 1), (128, 1, 1), 'input')
            buf848 = buf837; del buf837  # reuse
            # Topologically Sorted Source Nodes: [view_1478, view_1480, add_567, mul_1029, hidden_states_16, convert_element_type_1929, mul_1031, mul_1032, sum_218, pow_164, mul_1033, mul_1034, expand_226, div_111, pow_165, mul_1035, mul_1036, add_568, convert_element_type_1930, add_569], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf848, buf842, buf845, primals_18, add_13, rsqrt_3, 128, 3072, stream=raw_stream0)
            del primals_18
            buf846 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1478, view_1480, add_567, hidden_states_16, hidden_states_17, to_14, mul_1030, sum_217], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_6.run(buf842, buf845, add_13, rsqrt_3, buf846, 3072, 128, stream=raw_stream0)
            del add_13
            del rsqrt_3
            assert_size_stride(view_55, (128, 3072), (3072, 1), 'input')
            buf849 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1482, permute_1316, mm_569], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf848, (3072, 128), (1, 3072), 0), view_55, out=buf849)
            del view_55
            assert_size_stride(primals_17, (3072, 3072), (3072, 1), 'input')
            buf850 = buf845; del buf845  # reuse
            # Topologically Sorted Source Nodes: [view_1482, attn_output_7, permute_1318, mm_570], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf848, (128, 3072), (3072, 1), 0), primals_17, out=buf850)
            del primals_17
            assert_size_stride(permute_1321, (24, 128, 128), (16384, 1, 128), 'input')
            buf851 = reinterpret_tensor(buf842, (24, 128, 128), (16384, 128, 1), 0); del buf842  # reuse
            # Topologically Sorted Source Nodes: [view_1483, view_1484, permute_1320, view_1485, bmm_160], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1321, reinterpret_tensor(buf850, (24, 128, 128), (128, 3072, 1), 0), out=buf851)
            del permute_1321
            assert_size_stride(permute_1322, (24, 128, 128), (16384, 1, 128), 'input')
            buf852 = reinterpret_tensor(buf828, (24, 128, 128), (16384, 128, 1), 0); del buf828  # reuse
            # Topologically Sorted Source Nodes: [view_1483, view_1484, permute_1320, view_1485, bmm_161], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf850, (24, 128, 128), (128, 3072, 1), 0), permute_1322, out=buf852)
            del permute_1322
            buf858 = reinterpret_tensor(buf829, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf829  # reuse
            # Topologically Sorted Source Nodes: [view_1486, view_1491, sum_220, squeeze_53, permute_1326, clone_192], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf851, buf858, 131072, stream=raw_stream0)
            assert_size_stride(add_12, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax_1, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_2, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf854 = add_12; del add_12  # reuse
            # Topologically Sorted Source Nodes: [view_1487, convert_element_type_1939, softmax_1, mul_1037, sum_219, neg_190, fma_26, convert_element_type_1940, mul_1038], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf854, buf852, amax_1, sum_2, 3072, 128, stream=raw_stream0)
            del amax_1
            del sum_2
            assert_size_stride(view_37, (128, 3072), (3072, 1), 'input')
            buf859 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1486, view_1491, sum_220, squeeze_53, permute_1326, clone_192, view_1493, view_1494, permute_1327, mm_571], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf858, (1024, 128), (1, 1024), 0), view_37, out=buf859)
            assert_size_stride(primals_16, (1024, 3072), (3072, 1), 'input')
            buf860 = reinterpret_tensor(buf852, (128, 3072), (3072, 1), 0); del buf852  # reuse
            # Topologically Sorted Source Nodes: [view_1486, view_1491, sum_220, squeeze_53, permute_1326, clone_192, view_1493, view_1494, linear_9, permute_1329, mm_572], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf858, (128, 1024), (1024, 1), 0), primals_16, out=buf860)
            del primals_16
            assert_size_stride(permute_1323, (24, 128, 128), (128, 1, 3072), 'input')
            buf855 = buf851; del buf851  # reuse
            # Topologically Sorted Source Nodes: [view_1487, convert_element_type_1939, softmax_1, mul_1037, neg_190, fma_26, convert_element_type_1940, mul_1038, view_1488, bmm_162], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1323, reinterpret_tensor(buf854, (24, 128, 128), (16384, 128, 1), 0), out=buf855)
            del permute_1323
            assert_size_stride(permute_1324, (24, 128, 128), (16384, 128, 1), 'input')
            buf856 = reinterpret_tensor(buf850, (24, 128, 128), (16384, 128, 1), 0); del buf850  # reuse
            # Topologically Sorted Source Nodes: [view_1487, convert_element_type_1939, softmax_1, mul_1037, neg_190, fma_26, convert_element_type_1940, mul_1038, view_1488, bmm_163], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf854, (24, 128, 128), (16384, 128, 1), 0), permute_1324, out=buf856)
            del permute_1324
            buf857 = reinterpret_tensor(buf858, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf858  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1489, permute_1325, view_1492, sum_221, squeeze_54, mul_1039, slice_221, slice_222, neg_191, add_570, mul_1040, add_571], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf855, primals_3, buf857, 131072, stream=raw_stream0)
            buf864 = reinterpret_tensor(buf855, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf855  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1490, mul_1041, slice_223, slice_224, neg_192, add_572, mul_1042, add_573, permute_1336, clone_194], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf856, primals_3, buf864, 393216, stream=raw_stream0)
            buf861 = reinterpret_tensor(buf825, (128, 1024), (1024, 1), 0); del buf825  # reuse
            # Topologically Sorted Source Nodes: [permute_1331, clone_193, view_1496, view_1497], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf857, buf861, 128, 1024, stream=raw_stream0)
            buf865 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1490, mul_1041, slice_223, slice_224, neg_192, add_572, mul_1042, add_573, permute_1336, clone_194, view_1499, view_1500, permute_1337, mm_575], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf864, (3072, 128), (1, 3072), 0), view_37, out=buf865)
            assert_size_stride(primals_14, (3072, 3072), (3072, 1), 'input')
            buf866 = reinterpret_tensor(buf856, (128, 3072), (3072, 1), 0); del buf856  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1490, mul_1041, slice_223, slice_224, neg_192, add_572, mul_1042, add_573, permute_1336, clone_194, view_1499, view_1500, linear_7, permute_1339, mm_576], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf864, (128, 3072), (3072, 1), 0), primals_14, out=buf866)
            del primals_14
            buf862 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1332, mm_573], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf861, (1024, 128), (1, 1024), 0), view_37, out=buf862)
            del view_37
            assert_size_stride(primals_15, (1024, 3072), (3072, 1), 'input')
            buf863 = reinterpret_tensor(buf864, (128, 3072), (3072, 1), 0); del buf864  # reuse
            # Topologically Sorted Source Nodes: [linear_8, permute_1334, mm_574], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf861, primals_15, out=buf863)
            del primals_15
            assert_size_stride(primals_13, (3072, ), (1, ), 'input')
            assert_size_stride(add_8, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(rsqrt_2, (1, 128, 1), (128, 1, 1), 'input')
            buf869 = buf848; del buf848  # reuse
            # Topologically Sorted Source Nodes: [view_1495, view_1498, add_574, view_1501, add_575, mul_1043, hidden_states_10, convert_element_type_1957, mul_1045, mul_1046, sum_223, pow_166, mul_1047, mul_1048, expand_227, div_112, pow_167, mul_1049, mul_1050, add_576, convert_element_type_1958, add_577], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf869, buf860, buf863, buf866, primals_13, add_8, rsqrt_2, 128, 3072, stream=raw_stream0)
            del primals_13
            buf867 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1495, view_1498, add_574, view_1501, add_575, hidden_states_10, hidden_states_11, to_11, mul_1044, sum_222], Original ATen: [aten.view, aten.add, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_mul_sum_view_13.run(buf860, buf863, buf866, add_8, rsqrt_2, buf867, 3072, 128, stream=raw_stream0)
            del add_8
            del rsqrt_2
            assert_size_stride(view_35, (128, 8192), (8192, 1), 'input')
            buf870 = empty_strided_cuda((3072, 8192), (8192, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1503, permute_1341, mm_577], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf869, (3072, 128), (1, 3072), 0), view_35, out=buf870)
            del view_35
            assert_size_stride(primals_12, (3072, 8192), (8192, 1), 'input')
            buf871 = reinterpret_tensor(buf843, (128, 8192), (8192, 1), 0); del buf843  # reuse
            # Topologically Sorted Source Nodes: [view_1503, down_proj, permute_1343, mm_578], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf869, (128, 3072), (3072, 1), 0), primals_12, out=buf871)
            del primals_12
            assert_size_stride(mm_4, (128, 8192), (8192, 1), 'input')
            assert_size_stride(mm_5, (128, 8192), (8192, 1), 'input')
            buf872 = buf840; del buf840  # reuse
            buf875 = reinterpret_tensor(mm_5, (1, 128, 8192), (1048576, 8192, 1), 0); del mm_5  # reuse
            # Topologically Sorted Source Nodes: [view_1504, linear_4, silu, mul_1051, linear_5, mul_1052, convert_element_type_1967, reciprocal_27, mul_1053, mul_1054, sub_60, mul_1055, add_579, mul_1056, convert_element_type_1969], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_silu_backward_view_4.run(buf875, buf871, mm_4, buf872, 1048576, stream=raw_stream0)
            del buf871
            del mm_4
            assert_size_stride(view_31, (128, 3072), (3072, 1), 'input')
            buf873 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1504, linear_4, silu, mul_1051, view_1505, permute_1345, mm_579], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf872, (8192, 128), (1, 8192), 0), view_31, out=buf873)
            assert_size_stride(primals_11, (8192, 3072), (3072, 1), 'input')
            buf874 = buf866; del buf866  # reuse
            # Topologically Sorted Source Nodes: [view_1504, linear_4, silu, mul_1051, view_1505, linear_5, permute_1347, mm_580], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf872, (128, 8192), (8192, 1), 0), primals_11, out=buf874)
            del buf872
            del primals_11
            buf876 = empty_strided_cuda((8192, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1504, linear_4, silu, linear_5, mul_1052, convert_element_type_1967, reciprocal_27, mul_1053, mul_1054, sub_60, mul_1055, add_579, mul_1056, convert_element_type_1969, view_1507, permute_1349, mm_581], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf875, (8192, 128), (1, 8192), 0), view_31, out=buf876)
            del view_31
            assert_size_stride(primals_10, (8192, 3072), (3072, 1), 'input')
            buf877 = buf863; del buf863  # reuse
            # Topologically Sorted Source Nodes: [view_1504, linear_4, silu, linear_5, mul_1052, convert_element_type_1967, reciprocal_27, mul_1053, mul_1054, sub_60, mul_1055, add_579, mul_1056, convert_element_type_1969, view_1507, permute_1351, mm_582], Original ATen: [aten.view, aten._unsafe_view, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf875, (128, 8192), (8192, 1), 0), primals_10, out=buf877)
            del buf875
            del primals_10
            assert_size_stride(primals_9, (3072, ), (1, ), 'input')
            assert_size_stride(embedding, (1, 128, 3072), (393216, 3072, 1), 'input')
            assert_size_stride(mm_3, (128, 3072), (3072, 1), 'input')
            assert_size_stride(rsqrt_1, (1, 128, 1), (128, 1, 1), 'input')
            buf880 = buf869; del buf869  # reuse
            # Topologically Sorted Source Nodes: [view_1506, view_1508, add_580, mul_1057, attn_output_3, hidden_states_5, hidden_states_6, convert_element_type_1974, mul_1059, mul_1060, sum_225, pow_168, mul_1061, mul_1062, expand_228, div_113, pow_169, mul_1063, mul_1064, add_581, convert_element_type_1975, add_582], Original ATen: [aten.view, aten.add, aten.mul, aten._unsafe_view, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_div_expand_mul_pow_sum_view_14.run(buf880, buf874, buf877, primals_9, embedding, mm_3, rsqrt_1, 128, 3072, stream=raw_stream0)
            del primals_9
            assert_size_stride(primals_8, (3072, 3072), (3072, 1), 'input')
            buf882 = buf860; del buf860  # reuse
            # Topologically Sorted Source Nodes: [view_1510, attn_output_3, permute_1355, mm_584], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf880, (128, 3072), (3072, 1), 0), primals_8, out=buf882)
            del primals_8
            assert_size_stride(view_29, (128, 3072), (3072, 1), 'input')
            buf881 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1510, permute_1353, mm_583], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf880, (3072, 128), (1, 3072), 0), view_29, out=buf881)
            del view_29
            assert_size_stride(permute_1358, (24, 128, 128), (16384, 1, 128), 'input')
            buf883 = reinterpret_tensor(buf854, (24, 128, 128), (16384, 128, 1), 0); del buf854  # reuse
            # Topologically Sorted Source Nodes: [view_1511, view_1512, permute_1357, view_1513, bmm_164], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(permute_1358, reinterpret_tensor(buf882, (24, 128, 128), (128, 3072, 1), 0), out=buf883)
            del permute_1358
            assert_size_stride(permute_1359, (24, 128, 128), (16384, 1, 128), 'input')
            buf884 = reinterpret_tensor(buf822, (24, 128, 128), (16384, 128, 1), 0); del buf822  # reuse
            # Topologically Sorted Source Nodes: [view_1511, view_1512, permute_1357, view_1513, bmm_165], Original ATen: [aten.view, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf882, (24, 128, 128), (128, 3072, 1), 0), permute_1359, out=buf884)
            del permute_1359
            buf890 = reinterpret_tensor(buf861, (1, 128, 8, 128), (131072, 1024, 128, 1), 0); del buf861  # reuse
            # Topologically Sorted Source Nodes: [view_1514, view_1519, sum_227, squeeze_55, permute_1363, clone_195], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_squeeze_sum_transpose_view_7.run(buf883, buf890, 131072, stream=raw_stream0)
            assert_size_stride(add_4, (1, 24, 128, 128), (393216, 16384, 128, 1), 'input')
            assert_size_stride(amax, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            assert_size_stride(sum_1, (1, 24, 128, 1), (3072, 128, 1, 1), 'input')
            buf886 = add_4; del add_4  # reuse
            # Topologically Sorted Source Nodes: [view_1515, convert_element_type_1984, softmax, mul_1065, sum_226, neg_194, fma_27, convert_element_type_1985, mul_1066], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax__softmax_backward_data__to_copy_mul_view_8.run(buf886, buf884, amax, sum_1, 3072, 128, stream=raw_stream0)
            del amax
            del sum_1
            assert_size_stride(primals_7, (1024, 3072), (3072, 1), 'input')
            buf892 = reinterpret_tensor(buf884, (128, 3072), (3072, 1), 0); del buf884  # reuse
            # Topologically Sorted Source Nodes: [view_1514, view_1519, sum_227, squeeze_55, permute_1363, clone_195, view_1521, view_1522, linear_2, permute_1366, mm_586], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf890, (128, 1024), (1024, 1), 0), primals_7, out=buf892)
            del primals_7
            assert_size_stride(view_11, (128, 3072), (3072, 1), 'input')
            buf891 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1514, view_1519, sum_227, squeeze_55, permute_1363, clone_195, view_1521, view_1522, permute_1364, mm_585], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf890, (1024, 128), (1, 1024), 0), view_11, out=buf891)
            assert_size_stride(permute_1360, (24, 128, 128), (128, 1, 3072), 'input')
            buf887 = buf883; del buf883  # reuse
            # Topologically Sorted Source Nodes: [view_1515, convert_element_type_1984, softmax, mul_1065, neg_194, fma_27, convert_element_type_1985, mul_1066, view_1516, bmm_166], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(permute_1360, reinterpret_tensor(buf886, (24, 128, 128), (16384, 128, 1), 0), out=buf887)
            del permute_1360
            assert_size_stride(permute_1361, (24, 128, 128), (16384, 128, 1), 'input')
            buf888 = reinterpret_tensor(buf882, (24, 128, 128), (16384, 128, 1), 0); del buf882  # reuse
            # Topologically Sorted Source Nodes: [view_1515, convert_element_type_1984, softmax, mul_1065, neg_194, fma_27, convert_element_type_1985, mul_1066, view_1516, bmm_167], Original ATen: [aten.view, aten._to_copy, aten._softmax, aten._softmax_backward_data, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf886, (24, 128, 128), (16384, 128, 1), 0), permute_1361, out=buf888)
            del buf886
            del permute_1361
            buf889 = reinterpret_tensor(buf890, (1, 8, 128, 128), (131072, 16384, 1, 128), 0); del buf890  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_8, cos, cos_1, cos_2, cos_3, view_1517, permute_1362, view_1520, sum_228, squeeze_56, mul_1067, slice_225, slice_226, neg_195, add_583, mul_1068, add_584], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_cos_expand_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_9.run(buf887, primals_3, buf889, 131072, stream=raw_stream0)
            buf896 = reinterpret_tensor(buf887, (1, 128, 24, 128), (393216, 3072, 128, 1), 0); del buf887  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1518, mul_1069, slice_227, slice_228, neg_196, add_585, mul_1070, add_586, permute_1373, clone_197], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bmm_cat_clone_cos_expand_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_view_10.run(buf888, primals_3, buf896, 393216, stream=raw_stream0)
            del primals_3
            buf893 = reinterpret_tensor(buf857, (128, 1024), (1024, 1), 0); del buf857  # reuse
            # Topologically Sorted Source Nodes: [permute_1368, clone_196, view_1524, view_1525], Original ATen: [aten.transpose, aten.clone, aten._unsafe_view, aten.view]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_transpose_view_11.run(buf889, buf893, 128, 1024, stream=raw_stream0)
            del buf889
            assert_size_stride(primals_5, (3072, 3072), (3072, 1), 'input')
            buf898 = reinterpret_tensor(buf888, (128, 3072), (3072, 1), 0); del buf888  # reuse
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1518, mul_1069, slice_227, slice_228, neg_196, add_585, mul_1070, add_586, permute_1373, clone_197, view_1527, view_1528, linear, permute_1376, mm_590], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf896, (128, 3072), (3072, 1), 0), primals_5, out=buf898)
            del primals_5
            buf897 = empty_strided_cuda((3072, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_1, expand, getitem_2, position_ids_expanded, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_10, view_1518, mul_1069, slice_227, slice_228, neg_196, add_585, mul_1070, add_586, permute_1373, clone_197, view_1527, view_1528, permute_1374, mm_589], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten.cos, aten.slice_backward, aten.view, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf896, (3072, 128), (1, 3072), 0), view_11, out=buf897)
            assert_size_stride(primals_6, (1024, 3072), (3072, 1), 'input')
            buf895 = reinterpret_tensor(buf896, (128, 3072), (3072, 1), 0); del buf896  # reuse
            # Topologically Sorted Source Nodes: [linear_1, permute_1371, mm_588], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf893, primals_6, out=buf895)
            del primals_6
            assert_size_stride(rsqrt, (1, 128, 1), (128, 1, 1), 'input')
            buf878 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            buf899 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1506, view_1508, add_580, attn_output_3, hidden_states_5, hidden_states_6, hidden_states_7, to_9, mul_1058, sum_224, view_1523, view_1526, add_587, view_1529, add_588, hidden_states, hidden_states_1, to_6, mul_1072, sum_229], Original ATen: [aten.view, aten.add, aten._unsafe_view, aten._to_copy, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mul_sum_view_15.run(buf874, buf877, embedding, mm_3, rsqrt_1, buf892, buf895, buf898, rsqrt, buf878, buf899, 3072, 128, stream=raw_stream0)
            del buf874
            del buf877
            del mm_3
            del rsqrt_1
            assert_size_stride(primals_4, (3072, ), (1, ), 'input')
            assert_size_stride(primals_1, (1, 128), (128, 1), 'input')
            buf901 = empty_strided_cuda((1, 128, 3072), (393216, 3072, 1), torch.float32)
            # Topologically Sorted Source Nodes: [loss, view_1523, view_1526, add_587, view_1529, add_588, mul_1071, hidden_states, convert_element_type_2002, mul_1073, mul_1074, sum_230, pow_170, mul_1075, mul_1076, expand_229, div_114, pow_171, mul_1077, mul_1078, add_589, convert_element_type_2003, add_590, convert_element_type_2004, eq_1, unsqueeze_119, where_5], Original ATen: [aten.nll_loss_forward, aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div, aten.embedding_dense_backward]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_16.run(buf892, buf895, buf898, primals_4, embedding, primals_1, buf880, rsqrt, buf901, 128, 3072, stream=raw_stream0)
            del buf880
            del buf892
            del buf895
            del buf898
            del embedding
            del primals_4
            del rsqrt
            aten.index_put_(buf902, [primals_1], buf901, True)
            del buf901
            del primals_1
            buf904 = empty_strided_cuda((128256, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [convert_element_type_2005], Original ATen: [aten.embedding_dense_backward]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_embedding_dense_backward_17.run(buf902, buf904, 394002432, stream=raw_stream0)
            del buf902
            assert_size_stride(view_739, (128, 3072), (3072, 1), 'input')
            buf905 = empty_strided_cuda((128256, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [div_57, getitem_200, shift_labels_1, unsqueeze_118, ne_4, loss, where_3, where_self, where_4, mul_285, logits, logits_1, logits_2, exp_57, mul_286, sub_32, view_743, convert_element_type_737, view_744, permute_338, convert_element_type_2005, addmm_default], Original ATen: [aten.nll_loss_backward, aten.slice, aten.view, aten.nll_loss_forward, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._unsafe_view, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.t, aten.embedding_dense_backward, aten.add]
            extern_kernels.addmm(buf904, reinterpret_tensor(buf1, (128256, 128), (1, 128256), 0), view_739, alpha=1, beta=1, out=buf905)
            del buf1
            del buf904
            del view_739
            buf894 = empty_strided_cuda((1024, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_1369, mm_587], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf893, (1024, 128), (1, 1024), 0), view_11, out=buf894)
            del buf893
            del view_11
        return (None, buf905, None, reinterpret_tensor(buf899, (3072, ), (1, ), 0), buf897, buf894, buf891, buf881, reinterpret_tensor(buf878, (3072, ), (1, ), 0), buf876, buf873, buf870, reinterpret_tensor(buf867, (3072, ), (1, ), 0), buf865, buf862, buf859, buf849, reinterpret_tensor(buf846, (3072, ), (1, ), 0), buf844, buf841, buf838, reinterpret_tensor(buf835, (3072, ), (1, ), 0), buf833, buf830, buf827, buf817, reinterpret_tensor(buf814, (3072, ), (1, ), 0), buf812, buf809, buf806, reinterpret_tensor(buf803, (3072, ), (1, ), 0), buf801, buf798, buf795, buf785, reinterpret_tensor(buf782, (3072, ), (1, ), 0), buf780, buf777, buf774, reinterpret_tensor(buf771, (3072, ), (1, ), 0), buf769, buf766, buf763, buf753, reinterpret_tensor(buf750, (3072, ), (1, ), 0), buf748, buf745, buf742, reinterpret_tensor(buf739, (3072, ), (1, ), 0), buf737, buf734, buf731, buf721, reinterpret_tensor(buf718, (3072, ), (1, ), 0), buf716, buf713, buf710, reinterpret_tensor(buf707, (3072, ), (1, ), 0), buf705, buf702, buf699, buf689, reinterpret_tensor(buf686, (3072, ), (1, ), 0), buf684, buf681, buf678, reinterpret_tensor(buf675, (3072, ), (1, ), 0), buf673, buf670, buf667, buf657, reinterpret_tensor(buf654, (3072, ), (1, ), 0), buf652, buf649, buf646, reinterpret_tensor(buf643, (3072, ), (1, ), 0), buf641, buf638, buf635, buf625, reinterpret_tensor(buf622, (3072, ), (1, ), 0), buf620, buf617, buf614, reinterpret_tensor(buf611, (3072, ), (1, ), 0), buf609, buf606, buf603, buf593, reinterpret_tensor(buf590, (3072, ), (1, ), 0), buf588, buf585, buf582, reinterpret_tensor(buf579, (3072, ), (1, ), 0), buf577, buf574, buf571, buf561, reinterpret_tensor(buf558, (3072, ), (1, ), 0), buf556, buf553, buf550, reinterpret_tensor(buf547, (3072, ), (1, ), 0), buf545, buf542, buf539, buf529, reinterpret_tensor(buf526, (3072, ), (1, ), 0), buf524, buf521, buf518, reinterpret_tensor(buf515, (3072, ), (1, ), 0), buf513, buf510, buf507, buf497, reinterpret_tensor(buf494, (3072, ), (1, ), 0), buf492, buf489, buf486, reinterpret_tensor(buf483, (3072, ), (1, ), 0), buf481, buf478, buf475, buf465, reinterpret_tensor(buf462, (3072, ), (1, ), 0), buf460, buf457, buf454, reinterpret_tensor(buf451, (3072, ), (1, ), 0), buf449, buf446, buf443, buf433, reinterpret_tensor(buf430, (3072, ), (1, ), 0), buf428, buf425, buf422, reinterpret_tensor(buf419, (3072, ), (1, ), 0), buf417, buf414, buf411, buf401, reinterpret_tensor(buf398, (3072, ), (1, ), 0), buf396, buf393, buf390, reinterpret_tensor(buf387, (3072, ), (1, ), 0), buf385, buf382, buf379, buf369, reinterpret_tensor(buf366, (3072, ), (1, ), 0), buf364, buf361, buf358, reinterpret_tensor(buf355, (3072, ), (1, ), 0), buf353, buf350, buf347, buf337, reinterpret_tensor(buf334, (3072, ), (1, ), 0), buf332, buf329, buf326, reinterpret_tensor(buf323, (3072, ), (1, ), 0), buf321, buf318, buf315, buf305, reinterpret_tensor(buf302, (3072, ), (1, ), 0), buf300, buf297, buf294, reinterpret_tensor(buf291, (3072, ), (1, ), 0), buf289, buf286, buf283, buf273, reinterpret_tensor(buf270, (3072, ), (1, ), 0), buf268, buf265, buf262, reinterpret_tensor(buf259, (3072, ), (1, ), 0), buf257, buf254, buf251, buf241, reinterpret_tensor(buf238, (3072, ), (1, ), 0), buf236, buf233, buf230, reinterpret_tensor(buf227, (3072, ), (1, ), 0), buf225, buf222, buf219, buf209, reinterpret_tensor(buf206, (3072, ), (1, ), 0), buf204, buf201, buf198, reinterpret_tensor(buf195, (3072, ), (1, ), 0), buf193, buf190, buf187, buf177, reinterpret_tensor(buf174, (3072, ), (1, ), 0), buf172, buf169, buf166, reinterpret_tensor(buf163, (3072, ), (1, ), 0), buf161, buf158, buf155, buf145, reinterpret_tensor(buf142, (3072, ), (1, ), 0), buf140, buf137, buf134, reinterpret_tensor(buf131, (3072, ), (1, ), 0), buf129, buf126, buf123, buf113, reinterpret_tensor(buf110, (3072, ), (1, ), 0), buf108, buf105, buf102, reinterpret_tensor(buf99, (3072, ), (1, ), 0), buf97, buf94, buf91, buf81, reinterpret_tensor(buf78, (3072, ), (1, ), 0), buf76, buf73, buf70, reinterpret_tensor(buf67, (3072, ), (1, ), 0), buf65, buf62, buf59, buf49, reinterpret_tensor(buf46, (3072, ), (1, ), 0), buf44, buf41, buf38, reinterpret_tensor(buf35, (3072, ), (1, ), 0), buf33, buf30, buf27, buf17, reinterpret_tensor(buf14, (3072, ), (1, ), 0), buf12, buf9, buf6, reinterpret_tensor(buf3, (3072, ), (1, ), 0), )

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
    embedding = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_4 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_1 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_29 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_3 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_31 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_4 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_5 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_35 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_8 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_37 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_12 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_1 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_2 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_55 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_13 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_57 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_11 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_12 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_61 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_16 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_63 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_20 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_2 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_3 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_81 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_21 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_83 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_18 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_19 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_87 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_24 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_28 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_3 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_4 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_29 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_7 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_109 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_25 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_26 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_113 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_32 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_115 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_36 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_4 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_5 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_133 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_37 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_135 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_32 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_33 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_139 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_40 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_141 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_44 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_5 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_6 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_159 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_45 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_11 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_161 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_39 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_40 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_165 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_48 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_167 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_52 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_6 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_7 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_185 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_53 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_187 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_46 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_47 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_191 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_56 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_193 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_60 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_7 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_8 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_211 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_61 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_15 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_213 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_53 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_54 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_217 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_64 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_219 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_68 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_8 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_9 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_237 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_69 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_239 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_60 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_61 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_243 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_72 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_245 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_76 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_9 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_10 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_263 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_77 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_19 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_265 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_67 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_68 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_269 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_80 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_271 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_84 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_10 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_11 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_289 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_85 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_291 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_74 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_75 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_295 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_88 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_297 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_92 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_11 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_12 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_315 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_93 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_23 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_317 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_81 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_82 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_321 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_96 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_323 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_100 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_12 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_13 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_341 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_101 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_343 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_88 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_89 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_347 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_104 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_349 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_108 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_13 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_14 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_367 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_109 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_27 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_369 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_95 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_96 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_373 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_112 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_375 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_116 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_14 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_15 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_393 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_117 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_395 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_102 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_103 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_399 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_120 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_401 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_124 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_15 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_16 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_419 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_125 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_31 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_421 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_109 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_110 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_425 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_128 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_427 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_132 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_16 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_17 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_445 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_133 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_447 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_116 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_117 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_451 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_136 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_34 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_453 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_140 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_17 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_18 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_471 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_141 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_35 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_473 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_123 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_124 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_477 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_144 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_36 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_479 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_148 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_18 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_19 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_497 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_149 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_37 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_499 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_130 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_131 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_503 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_152 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_38 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_505 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_156 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_19 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_20 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_523 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_157 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_39 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_525 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_137 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_138 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_529 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_160 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_40 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_531 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_164 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_20 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_21 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_549 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_165 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_41 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_551 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_144 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_145 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_555 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_168 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_42 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_557 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_172 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_21 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_22 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_575 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_173 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_43 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_577 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_151 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_152 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_581 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_176 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_44 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_583 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_180 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_22 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_23 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_601 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_181 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_45 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_603 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_158 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_159 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_607 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_184 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_46 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_609 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_188 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_23 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_24 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_627 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_189 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_47 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_629 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_165 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_166 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_633 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_192 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_48 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_635 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_196 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_24 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_25 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_653 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_197 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_49 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_655 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_172 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_173 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_659 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_200 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_50 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_661 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_204 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_25 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_26 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_679 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_205 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_51 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_681 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_179 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_180 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_685 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_208 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_52 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_687 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_212 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_26 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_27 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_705 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_213 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_53 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_707 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_186 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_187 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_711 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_216 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_54 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_713 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_220 = rand_strided((1, 24, 128, 128), (393216, 16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    amax_27 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_28 = rand_strided((1, 24, 128, 1), (3072, 128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_731 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    add_221 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_55 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_733 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_193 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_194 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    view_737 = rand_strided((128, 8192), (8192, 1), device='cuda:0', dtype=torch.bfloat16)
    add_224 = rand_strided((1, 128, 3072), (393216, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_56 = rand_strided((1, 128, 1), (128, 1, 1), device='cuda:0', dtype=torch.float32)
    view_739 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_196 = rand_strided((128, 128256), (128256, 1), device='cuda:0', dtype=torch.bfloat16)
    constant_pad_nd = rand_strided((1, 129), (129, 1), device='cuda:0', dtype=torch.int64)
    amax_28 = rand_strided((128, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((128, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_736 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    permute_359 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_360 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_361 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_362 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_396 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_397 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_398 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_399 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_433 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_434 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_435 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_436 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_470 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_471 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_472 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_473 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_507 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_508 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_509 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_510 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_544 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_545 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_546 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_547 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_581 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_582 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_583 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_584 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_618 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_619 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_620 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_621 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_655 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_656 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_657 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_658 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_692 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_693 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_694 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_695 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_729 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_730 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_731 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_732 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_766 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_767 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_768 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_769 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_803 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_804 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_805 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_806 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_840 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_841 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_842 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_843 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_877 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_878 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_879 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_880 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_914 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_915 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_916 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_917 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_951 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_952 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_953 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_954 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_988 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_989 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_990 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_991 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1025 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1026 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1027 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1028 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1062 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1063 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1064 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1065 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1099 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1100 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1101 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1102 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1136 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1137 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1138 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1139 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1173 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1174 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1175 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1176 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1210 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1211 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1212 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1213 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1247 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1248 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1249 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1250 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1284 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1285 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1286 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1287 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1321 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1322 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1323 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1324 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1358 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1359 = rand_strided((24, 128, 128), (16384, 1, 128), device='cuda:0', dtype=torch.bfloat16)
    permute_1360 = rand_strided((24, 128, 128), (128, 1, 3072), device='cuda:0', dtype=torch.bfloat16)
    permute_1361 = rand_strided((24, 128, 128), (16384, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, embedding, rsqrt, view_11, add_4, amax, sum_1, view_29, mm_3, rsqrt_1, view_31, mm_4, mm_5, view_35, add_8, rsqrt_2, view_37, add_12, amax_1, sum_2, view_55, add_13, rsqrt_3, view_57, mm_11, mm_12, view_61, add_16, rsqrt_4, view_63, add_20, amax_2, sum_3, view_81, add_21, rsqrt_5, view_83, mm_18, mm_19, view_87, add_24, rsqrt_6, view_89, add_28, amax_3, sum_4, view_107, add_29, rsqrt_7, view_109, mm_25, mm_26, view_113, add_32, rsqrt_8, view_115, add_36, amax_4, sum_5, view_133, add_37, rsqrt_9, view_135, mm_32, mm_33, view_139, add_40, rsqrt_10, view_141, add_44, amax_5, sum_6, view_159, add_45, rsqrt_11, view_161, mm_39, mm_40, view_165, add_48, rsqrt_12, view_167, add_52, amax_6, sum_7, view_185, add_53, rsqrt_13, view_187, mm_46, mm_47, view_191, add_56, rsqrt_14, view_193, add_60, amax_7, sum_8, view_211, add_61, rsqrt_15, view_213, mm_53, mm_54, view_217, add_64, rsqrt_16, view_219, add_68, amax_8, sum_9, view_237, add_69, rsqrt_17, view_239, mm_60, mm_61, view_243, add_72, rsqrt_18, view_245, add_76, amax_9, sum_10, view_263, add_77, rsqrt_19, view_265, mm_67, mm_68, view_269, add_80, rsqrt_20, view_271, add_84, amax_10, sum_11, view_289, add_85, rsqrt_21, view_291, mm_74, mm_75, view_295, add_88, rsqrt_22, view_297, add_92, amax_11, sum_12, view_315, add_93, rsqrt_23, view_317, mm_81, mm_82, view_321, add_96, rsqrt_24, view_323, add_100, amax_12, sum_13, view_341, add_101, rsqrt_25, view_343, mm_88, mm_89, view_347, add_104, rsqrt_26, view_349, add_108, amax_13, sum_14, view_367, add_109, rsqrt_27, view_369, mm_95, mm_96, view_373, add_112, rsqrt_28, view_375, add_116, amax_14, sum_15, view_393, add_117, rsqrt_29, view_395, mm_102, mm_103, view_399, add_120, rsqrt_30, view_401, add_124, amax_15, sum_16, view_419, add_125, rsqrt_31, view_421, mm_109, mm_110, view_425, add_128, rsqrt_32, view_427, add_132, amax_16, sum_17, view_445, add_133, rsqrt_33, view_447, mm_116, mm_117, view_451, add_136, rsqrt_34, view_453, add_140, amax_17, sum_18, view_471, add_141, rsqrt_35, view_473, mm_123, mm_124, view_477, add_144, rsqrt_36, view_479, add_148, amax_18, sum_19, view_497, add_149, rsqrt_37, view_499, mm_130, mm_131, view_503, add_152, rsqrt_38, view_505, add_156, amax_19, sum_20, view_523, add_157, rsqrt_39, view_525, mm_137, mm_138, view_529, add_160, rsqrt_40, view_531, add_164, amax_20, sum_21, view_549, add_165, rsqrt_41, view_551, mm_144, mm_145, view_555, add_168, rsqrt_42, view_557, add_172, amax_21, sum_22, view_575, add_173, rsqrt_43, view_577, mm_151, mm_152, view_581, add_176, rsqrt_44, view_583, add_180, amax_22, sum_23, view_601, add_181, rsqrt_45, view_603, mm_158, mm_159, view_607, add_184, rsqrt_46, view_609, add_188, amax_23, sum_24, view_627, add_189, rsqrt_47, view_629, mm_165, mm_166, view_633, add_192, rsqrt_48, view_635, add_196, amax_24, sum_25, view_653, add_197, rsqrt_49, view_655, mm_172, mm_173, view_659, add_200, rsqrt_50, view_661, add_204, amax_25, sum_26, view_679, add_205, rsqrt_51, view_681, mm_179, mm_180, view_685, add_208, rsqrt_52, view_687, add_212, amax_26, sum_27, view_705, add_213, rsqrt_53, view_707, mm_186, mm_187, view_711, add_216, rsqrt_54, view_713, add_220, amax_27, sum_28, view_731, add_221, rsqrt_55, view_733, mm_193, mm_194, view_737, add_224, rsqrt_56, view_739, mm_196, constant_pad_nd, amax_28, log, convert_element_type_736, permute_359, permute_360, permute_361, permute_362, permute_396, permute_397, permute_398, permute_399, permute_433, permute_434, permute_435, permute_436, permute_470, permute_471, permute_472, permute_473, permute_507, permute_508, permute_509, permute_510, permute_544, permute_545, permute_546, permute_547, permute_581, permute_582, permute_583, permute_584, permute_618, permute_619, permute_620, permute_621, permute_655, permute_656, permute_657, permute_658, permute_692, permute_693, permute_694, permute_695, permute_729, permute_730, permute_731, permute_732, permute_766, permute_767, permute_768, permute_769, permute_803, permute_804, permute_805, permute_806, permute_840, permute_841, permute_842, permute_843, permute_877, permute_878, permute_879, permute_880, permute_914, permute_915, permute_916, permute_917, permute_951, permute_952, permute_953, permute_954, permute_988, permute_989, permute_990, permute_991, permute_1025, permute_1026, permute_1027, permute_1028, permute_1062, permute_1063, permute_1064, permute_1065, permute_1099, permute_1100, permute_1101, permute_1102, permute_1136, permute_1137, permute_1138, permute_1139, permute_1173, permute_1174, permute_1175, permute_1176, permute_1210, permute_1211, permute_1212, permute_1213, permute_1247, permute_1248, permute_1249, permute_1250, permute_1284, permute_1285, permute_1286, permute_1287, permute_1321, permute_1322, permute_1323, permute_1324, permute_1358, permute_1359, permute_1360, permute_1361, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
