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


# kernel path: /tmp/torchinductor_tzh/6g/c6gopqfzfys4rhqhbxarcyygmmrq4e64mp422llzknfjcv7vr33f.py
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
#   %constant_pad_nd : Tensor "i64[1, 513][513, 1]cuda:0" = PlaceHolder[target=constant_pad_nd]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type_736 : Tensor "f32[][]cuda:0" = PlaceHolder[target=convert_element_type_736]
#   %mm_196 : Tensor "bf16[512, 128256][128256, 1]cuda:0" = PlaceHolder[target=mm_196]
#   %amax_28 : Tensor "f32[512, 1][1, 1]cuda:0" = PlaceHolder[target=amax_28]
#   %log : Tensor "f32[512, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_32 : Tensor "f32[512, 1][1, 512]cuda:0" = PlaceHolder[target=sum_32]
#   %div_57 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%tangents_1, %convert_element_type_736), kwargs = {})
#   %slice_116 : Tensor "i64[1, 512][513, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, 1, 1, 9223372036854775807), kwargs = {})
#   %view_742 : Tensor "i64[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%slice_116, [-1]), kwargs = {})
#   %unsqueeze_118 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_742, 1), kwargs = {})
#   %ne_4 : Tensor "b8[512, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_118, -100), kwargs = {})
#   %full_default_3 : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "i64[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %unsqueeze_118, %full_default_3), kwargs = {})
#   %iota_default : Tensor "i64[128256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (128256,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view_default : Tensor "i64[1, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_default, [1, 128256]), kwargs = {})
#   %expand_default : Tensor "i64[512, 128256][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%where_3, [512, 128256]), kwargs = {})
#   %eq_tensor : Tensor "b8[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%expand_default, %view_default), kwargs = {})
#   %scalar_tensor_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0})
#   %scalar_tensor_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (-1.0,), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0})
#   %where_self : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_tensor, %scalar_tensor_default_1, %scalar_tensor_default), kwargs = {})
#   %full_default_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %div_57, %full_default_4), kwargs = {})
#   %mul_285 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_self, %where_4), kwargs = {})
#   %view_740 : Tensor "bf16[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_196, [1, 512, 128256]), kwargs = {})
#   %convert_element_type_735 : Tensor "f32[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_740, torch.float32), kwargs = {})
#   %view_741 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_735, [-1, 128256]), kwargs = {})
#   %sub_30 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_741, %amax_28), kwargs = {})
#   %sub_31 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_30, %log), kwargs = {})
#   %exp_57 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_31,), kwargs = {})
#   %sum_32 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_285, [1], True), kwargs = {})
#   %mul_286 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_57, %sum_32), kwargs = {})
#   %sub_32 : Tensor "f32[512, 128256][128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_285, %mul_286), kwargs = {})
#   %view_743 : Tensor "f32[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_32, [1, 512, 128256]), kwargs = {})
#   %convert_element_type_737 : Tensor "bf16[1, 512, 128256][65667072, 128256, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_743, torch.bfloat16), kwargs = {})
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
    size_hints={'x': 512, 'r0_': 131072},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': True, 'tiling_scores': {'x': 8192, 'r0_': 394002432}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_tzh/5s/c5shft2yijwvclzeosetsv4xjtyupbpudkn4q32a5jjd2a6o66l7.py
# Topologically Sorted Source Nodes: [view_745, mul_287, sum_33], Original ATen: [aten.view, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_287 => mul_287
#   sum_33 => sum_33
#   view_745 => view_745
# Graph fragment:
#   %mm_197 : Tensor "bf16[512, 3072][3072, 1]cuda:0" = PlaceHolder[target=mm_197]
#   %convert_element_type_732 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0" = PlaceHolder[target=convert_element_type_732]
#   %view_745 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_197, [1, 512, 3072]), kwargs = {})
#   %mul_287 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_745, %convert_element_type_732), kwargs = {})
#   %sum_33 : Tensor "bf16[1, 1, 3072][3072, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_287, [0, 1], True), kwargs = {})
#   return %buf3
triton_red_fused_mul_sum_view_1 = async_compile.triton('triton_red_fused_mul_sum_view_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_mul_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 6389760, 'r0_': 0}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_red_fused_mul_sum_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 12288
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_2 + 393216*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 3072*r0_2 + 393216*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_tzh/bf/cbfx6lapdyhdvlyys3aivb6ymv6u6klmltyjrl6jmeu6apxqsbug.py
# Topologically Sorted Source Nodes: [view_745, mul_287, sum_33], Original ATen: [aten.view, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_287 => mul_287
#   sum_33 => sum_33
#   view_745 => view_745
# Graph fragment:
#   %buf3 : Tensor "f32[1, 1, 3072, 4][12288, 12288, 1, 3072]cuda:0" = PlaceHolder[target=buf3]
#   %view_745 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_197, [1, 512, 3072]), kwargs = {})
#   %mul_287 : Tensor "bf16[1, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_745, %convert_element_type_732), kwargs = {})
#   %sum_33 : Tensor "bf16[1, 1, 3072][3072, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_287, [0, 1], True), kwargs = {})
#   return %sum_33
triton_per_fused_mul_sum_view_2 = async_compile.triton('triton_per_fused_mul_sum_view_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=84, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_mul_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'has_loadstore_with_contiguous_rdim': False, 'tiling_scores': {'x': 61440, 'r0_': 0}, 'backend_hash': '3135183A137B399FE9A48BE977B8D54EFEE9046D08738DFB1BFCF6EE9DF99547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': True, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': True}
)
@triton.jit
def triton_per_fused_mul_sum_view_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3072
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
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
    tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
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
        primals_2, convert_element_type_732, mm_196, constant_pad_nd, amax_28, log, convert_element_type_736, tangents_1 = args
        args.clear()
        assert_size_stride(constant_pad_nd, (1, 513), (513, 1), 'input')
        assert_size_stride(tangents_1, (), (), 'input')
        assert_size_stride(convert_element_type_736, (), (), 'input')
        assert_size_stride(mm_196, (512, 128256), (128256, 1), 'input')
        assert_size_stride(amax_28, (512, 1), (1, 1), 'input')
        assert_size_stride(log, (512, 1), (1, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            tangents_1 = copy_if_misaligned(tangents_1)
            buf1 = reinterpret_tensor(mm_196, (1, 512, 128256), (65667072, 128256, 1), 0); del mm_196  # reuse
            # Topologically Sorted Source Nodes: [div_57, getitem_200, shift_labels_1, unsqueeze_118, ne_4, loss, where_3, where_self, where_4, mul_285, logits, logits_1, logits_2, exp_57, sum_32, mul_286, sub_32, view_743, convert_element_type_737], Original ATen: [aten.nll_loss_backward, aten.slice, aten.view, aten.nll_loss_forward, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._unsafe_view, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy__unsafe_view_arange_eq_expand_nll_loss_backward_nll_loss_forward_scalar_tensor_slice_view_0.run(buf1, constant_pad_nd, tangents_1, convert_element_type_736, amax_28, log, 512, 128256, stream=raw_stream0)
            del amax_28
            del constant_pad_nd
            del convert_element_type_736
            del log
            del tangents_1
            assert_size_stride(primals_2, (128256, 3072), (3072, 1), 'input')
            buf2 = empty_strided_cuda((512, 3072), (3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [div_57, getitem_200, shift_labels_1, unsqueeze_118, ne_4, loss, where_3, where_self, where_4, mul_285, logits, logits_1, logits_2, exp_57, mul_286, sub_32, view_743, convert_element_type_737, view_744, permute_338, mm_197], Original ATen: [aten.nll_loss_backward, aten.slice, aten.view, aten.nll_loss_forward, aten.arange, aten.expand, aten.eq, aten.scalar_tensor, aten._unsafe_view, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1, (512, 128256), (128256, 1), 0), primals_2, out=buf2)
            del buf1
            del primals_2
            assert_size_stride(convert_element_type_732, (1, 512, 3072), (1572864, 3072, 1), 'input')
            buf3 = empty_strided_cuda((1, 1, 3072, 4), (12288, 12288, 1, 3072), torch.float32)
            # Topologically Sorted Source Nodes: [view_745, mul_287, sum_33], Original ATen: [aten.view, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_view_1.run(buf2, convert_element_type_732, buf3, 12288, 128, stream=raw_stream0)
            del buf2
            del convert_element_type_732
            buf4 = empty_strided_cuda((1, 1, 3072), (3072, 3072, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_745, mul_287, sum_33], Original ATen: [aten.view, aten.mul, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_mul_sum_view_2.run(buf3, buf4, 3072, 4, stream=raw_stream0)
            del buf3
        return (None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, reinterpret_tensor(buf4, (3072, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_2 = rand_strided((128256, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_732 = rand_strided((1, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_196 = rand_strided((512, 128256), (128256, 1), device='cuda:0', dtype=torch.bfloat16)
    constant_pad_nd = rand_strided((1, 513), (513, 1), device='cuda:0', dtype=torch.int64)
    amax_28 = rand_strided((512, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((512, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_736 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    return [primals_2, convert_element_type_732, mm_196, constant_pad_nd, amax_28, log, convert_element_type_736, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
