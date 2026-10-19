"""Finite MX block mathematics under an explicitly selected conversion rule.

Migration helper, not a certified OCP contract. The retained local Triton
numerics_details/mxfp_details/_downcast_to_mxfp.py distinguishes ROUND_DOWN
(amax divided by the largest power of two in the element range) and ROUND_UP
(amax divided by the maximum finite element). Here all arithmetic is exact,
scales are clamped to finite E8M0 exponents, and elements use saturating RNE.
This does not reproduce that kernel's intermediate floating-point rounding.
OCP section 6.3, other legal choices, and full layout binding remain to review.
"""
from fractions import Fraction as Q
from itertools import product

from . import ieee,d2_common as U
from .real import R
from .num import Invalid,Unsupported,NotEstablished,rational,integer

_MAX={'e2m1':Q(6),'e4m3':Q(448),'e5m2':Q(57344)}
_FP4=(Q(0),Q(1,2),Q(1),Q(3,2),Q(2),Q(3),Q(4),Q(6))
_TORCHAO016_MXFP8 = {
    'e4m3': {'maximum': Q(448), 'maximum_power_exponent': 8},
    'e5m2': {'maximum': Q(57344), 'maximum_power_exponent': 15},
}


def _format(fmt):
    if not isinstance(fmt,str) or fmt not in _MAX:
        raise Unsupported('supported finite element formats: e2m1, e4m3, e5m2')
    return _MAX[fmt]


def _floor_log2(x):
    exponent=x.numerator.bit_length()-x.denominator.bit_length()
    return exponent-1 if Q(2)**exponent>x else exponent


def _zero_scale(value):
    if value is None:return None
    value=rational(value,'zero_scale')
    if value<=0:
        raise Invalid('zero_scale must be a positive finite E8M0 power of two')
    exponent=_floor_log2(value)
    if not -127<=exponent<=127 or value!=Q(2)**exponent:
        raise Invalid('zero_scale must be a finite E8M0 power of two')
    return value


def round_element(value,fmt):
    """Finite real value -> exact saturating nearest-even element value.

    Numerical values only: signed zero and byte packing are outside this helper.
    """
    maximum=_format(fmt);x=rational(value,'element')
    if fmt=='e2m1':
        # Encoding parity is the retained significand's low bit, including
        # normal/subnormal boundaries. Distance is exact; there is no epsilon.
        index=min(range(8),key=lambda i:(abs(abs(x)-_FP4[i]),i%2))
        return -_FP4[index] if x<0 else _FP4[index]
    clamped=max(-maximum,min(maximum,x))
    return ieee.round_to(R(clamped),'f8e4m3' if fmt=='e4m3' else 'f8e5m2').v


def quantize(block,fmt,*,profile,zero_scale=None):
    """Logical block of 1..32 values; return scale, all elements and dequantized values.

    ocp_floor and ceil_pow2 are distinct selected formulas, not a correct/wrong
    pair. The historical name ocp_floor is not a standards-conformance claim.
    For an all-zero block the caller must explicitly select a finite E8M0 scale;
    absent that choice, the full result is not established, never a fabricated
    'None' scale. This does not restrict what a future allowed-set contract may do.
    """
    maximum=_format(fmt)
    if profile not in ('ocp_floor','ceil_pow2'):
        raise Unsupported('select ocp_floor or ceil_pow2')
    if not isinstance(block,(list,tuple)) or not 1<=len(block)<=32:
        raise Invalid('logical MX block must contain 1..32 values')
    values=[rational(x,'block element') for x in block]
    zero_scale=_zero_scale(zero_scale)
    amax=max(map(abs,values))
    if amax==0:
        if zero_scale is None:
            raise NotEstablished('zero-block scale convention has not been selected')
        scale=zero_scale
    else:
        if zero_scale is not None:
            raise Unsupported('zero_scale is only applicable to an all-zero block')
        if profile=='ocp_floor':
            exponent=_floor_log2(amax)-_floor_log2(maximum)
        else:
            ratio=amax/maximum;exponent=_floor_log2(ratio)
            if Q(2)**exponent<ratio:exponent+=1
        scale=Q(2)**max(-127,min(127,exponent))
    elements=[round_element(x/scale,fmt) for x in values]
    return {'scale':scale,'elements':elements,'dequantized':[scale*x for x in elements]}


def quantize_tensor(x,fmt,*,axis,profile,partial_blocks=False,zero_scale=None):
    """Logical tensor blocks of 32 along an explicitly bound axis.

    Scale shape equals input shape with the axis replaced by ceil(length/32).
    By default the axis length must be divisible by 32, as in the retained
    Triton downcast API. partial_blocks=True explicitly selects an extended
    task: the last block uses only its present values, without padding reads.
    zero_scale, if selected, is a common policy for every all-zero block.
    This binds logical coordinates only, not packing, strides or device writes.
    """
    _format(fmt)
    if profile not in ('ocp_floor','ceil_pow2'):
        raise Unsupported('select ocp_floor or ceil_pow2')
    shape=U.shape_of(x,'x')
    if not shape:
        raise Invalid('MX tensor must have at least one dimension')
    axis=integer(axis,'axis')
    if not -len(shape)<=axis<len(shape):
        raise Invalid('quantization axis out of range')
    axis%=len(shape)
    if type(partial_blocks) is not bool:
        raise Invalid('partial_blocks must be an explicit boolean')
    if shape[axis]%32 and not partial_blocks:
        raise Unsupported('quantization axis length must be divisible by 32 unless logical partial blocks are selected')
    zero_scale=_zero_scale(zero_scale)
    values=U.plain(U.tensor(x,len(shape),'x',shape=shape)[0])
    scale_shape=list(shape);scale_shape[axis]=(shape[axis]+31)//32

    def allocate(dims):
        return Q(0) if not dims else [allocate(dims[1:]) for _ in range(dims[0])]

    def get(tensor,coord):
        for i in coord:tensor=tensor[i]
        return tensor

    def put(tensor,coord,value):
        for i in coord[:-1]:tensor=tensor[i]
        tensor[coord[-1]]=value

    scales=allocate(scale_shape);elements=allocate(shape);dequantized=allocate(shape)
    for block_coord in product(*(range(n) for n in scale_shape)):
        start=block_coord[axis]*32
        coords=[block_coord[:axis]+(i,)+block_coord[axis+1:] for i in range(start,min(start+32,shape[axis]))]
        block=[get(values,coord) for coord in coords]
        result=quantize(block,fmt,profile=profile,zero_scale=zero_scale if not any(block) else None)
        put(scales,block_coord,result['scale'])
        for i,coord in enumerate(coords):
            put(elements,coord,result['elements'][i]);put(dequantized,coord,result['dequantized'][i])
    return {'scales':scales,'elements':elements,'dequantized':dequantized}


def torchao016_floor_block(block, fmt):
    """torchao 0.16 ``to_mx`` FLOOR conversion for one finite MXFP8 block.

    The public result is the E8M0 byte code and the numerical FP8 elements.
    torchao derives the scale exponent from the float32 exponent bits.  Its
    current implementation reconstructs code 0 as float32 zero and clamps the
    divisor to 2^-126, while the returned E8M0 code 0 denotes 2^-127.  The
    distinction matters for nonzero float32/bfloat16 subnormals and is kept
    here instead of silently replacing the implementation with ideal real
    arithmetic.
    """
    if fmt not in _TORCHAO016_MXFP8:
        raise Unsupported('torchao 0.16 FLOOR profile supports e4m3 or e5m2')
    if not isinstance(block, (list, tuple)) or len(block) != 32:
        raise Invalid('torchao 0.16 MXFP8 block must contain exactly 32 values')
    values = [rational(value, 'block element') for value in block]
    amax = max(map(abs, values))

    # A float32 zero or subnormal has a stored exponent field of zero.  The
    # torchao code subtracts the float32 bias from that field, producing -127.
    source_exponent = -127 if amax < Q(2) ** -126 else _floor_log2(amax)
    info = _TORCHAO016_MXFP8[fmt]
    scale_exponent = max(-127, min(128, source_exponent - info['maximum_power_exponent']))
    scale_code = scale_exponent + 127

    # The released implementation clamps its reconstructed divisor to the
    # smallest normal float32 value.  For finite f32/bf16 input, scale_code 255
    # cannot be produced here.
    divisor = Q(2) ** max(-126, scale_exponent)
    elements = [round_element(max(-info['maximum'], min(info['maximum'], value / divisor)), fmt)
                for value in values]
    return {'scale_code': scale_code, 'elements': elements}


def torchao016_floor_tensor(x, fmt):
    """Finite, non-swizzled torchao 0.16 MXFP8 FLOOR conversion.

    Blocks are consecutive groups of 32 on the last logical dimension.  This
    helper models the public numerical result, not strides or storage aliasing.
    Input dtype/physical-value validation belongs to the adapter because dtype
    is part of the public contract rather than this arithmetic helper.
    """
    if fmt not in _TORCHAO016_MXFP8:
        raise Unsupported('torchao 0.16 FLOOR profile supports e4m3 or e5m2')
    shape = U.shape_of(x, 'x')
    if not shape:
        raise Invalid('MXFP8 input must have at least one dimension')
    if shape[-1] <= 0 or shape[-1] % 32:
        raise Invalid('the last input dimension must be a positive multiple of 32')
    values = U.plain(U.tensor(x, len(shape), 'x', shape=shape)[0])
    scale_shape = shape[:-1] + (shape[-1] // 32,)

    def allocate(dims, fill=Q(0)):
        return fill if not dims else [allocate(dims[1:], fill) for _ in range(dims[0])]

    def get(tensor, coord):
        for index in coord:
            tensor = tensor[index]
        return tensor

    def put(tensor, coord, value):
        for index in coord[:-1]:
            tensor = tensor[index]
        tensor[coord[-1]] = value

    scale_codes = allocate(scale_shape, 0)
    elements = allocate(shape)
    leading_shape = shape[:-1]
    for leading in product(*(range(n) for n in leading_shape)) if leading_shape else [()]:
        for block_index in range(scale_shape[-1]):
            coords = [leading + (block_index * 32 + offset,) for offset in range(32)]
            result = torchao016_floor_block([get(values, coord) for coord in coords], fmt)
            put(scale_codes, leading + (block_index,), result['scale_code'])
            for coord, value in zip(coords, result['elements']):
                put(elements, coord, value)
    return {'scale_codes': scale_codes, 'elements': elements}
