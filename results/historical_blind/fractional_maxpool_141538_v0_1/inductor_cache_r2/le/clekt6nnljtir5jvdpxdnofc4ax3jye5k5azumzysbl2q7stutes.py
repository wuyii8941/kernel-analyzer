
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
