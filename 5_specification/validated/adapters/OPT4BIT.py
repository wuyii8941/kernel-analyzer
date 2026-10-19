"""torchao 0.16 OptimState4bit float32 copy, including its physical state."""
from __future__ import annotations

from fractions import Fraction

from kernel_spec import ieee
from kernel_spec.num import Invalid, Result, Unsupported, integer, rational, require_fields
from kernel_spec.real import R


PROFILE = "torchao016_state_copy_f32"
REQUIRED = {"src", "input_dtype", "signed", "block_size", "qmap"}
SCALE_MIN = Fraction(2305843, 2305843009213693952)  # float32(1e-12)
MIN_NORMAL_F32 = Fraction(1, 1 << 126)

QMAP_SIGNED = [Fraction(float.fromhex(h)) for h in (
    "-0x1.c666660000000p-1", "-0x1.5333340000000p-1", "-0x1.c000000000000p-2", "-0x1.b333320000000p-3",
    "-0x1.3d70a40000000p-4", "-0x1.0a3d720000000p-5", "-0x1.6872b00000000p-8", "0x0.0p+0",
    "0x1.6872b00000000p-8", "0x1.0a3d720000000p-5", "0x1.3d70a40000000p-4", "0x1.b333320000000p-3",
    "0x1.c000000000000p-2", "0x1.5333340000000p-1", "0x1.c666660000000p-1", "0x1.0000000000000p+0",
)]
QMAP_UNSIGNED = [Fraction(k, 16) for k in range(1, 17)]


def _f32(value, name):
    if isinstance(value, bool):
        raise Invalid(f"{name}: Boolean is not a float32 value")
    value = rational(value, name)
    rounded = ieee.round_to(R(value), "f32")
    if not rounded.finite:
        raise Unsupported(f"{name}: special or overflowing values are outside this profile")
    if not rounded.exact or rounded.v != value:
        raise Invalid(f"{name}: value is not exactly representable as float32")
    return value


def _round_f32(value, name="float32 operation"):
    rounded = ieee.round_to(R(value), "f32")
    if not rounded.finite:
        raise Unsupported(f"{name} overflowed the reviewed finite profile")
    if not rounded.exact:
        raise Unsupported(f"{name} could not be resolved to one float32 value")
    return rounded.v


def _parse(profile, inp):
    if profile != PROFILE:
        raise Unsupported(f"profile {profile!r}")
    require_fields(inp, REQUIRED, "OPT4BIT")
    extra = set(inp) - REQUIRED
    if extra:
        raise Unsupported(f"fields outside the reviewed state-copy task: {sorted(extra)}")
    if inp["input_dtype"] != "f32":
        raise Unsupported("the reviewed state-copy profile fixes the source dtype at float32")
    if type(inp["signed"]) is not bool:
        raise Invalid("signed must be a Boolean")
    if not isinstance(inp["src"], (list, tuple)):
        raise Invalid("src must be a one-dimensional float32 tensor")
    src = [_f32(value, f"src[{i}]") for i, value in enumerate(inp["src"])]
    if any(value and abs(value) < MIN_NORMAL_F32 for value in src):
        raise Unsupported("float32 subnormal source values are outside this device-portable profile")
    block_size = integer(inp["block_size"], "block_size", lo=1)
    if block_size % 2:
        raise Unsupported("this reviewed physical packing profile requires an even block_size")
    if not src or len(src) % block_size or len(src) % 2:
        raise Invalid("src length must be positive and divisible by block_size and two")
    if not inp["signed"] and any(value < 0 for value in src):
        raise Unsupported("the unsigned optimizer-state profile is limited to nonnegative second moments")
    if not isinstance(inp["qmap"], (list, tuple)) or len(inp["qmap"]) != 16:
        raise Invalid("qmap must be a one-dimensional float32 tensor of length16")
    qmap = [_f32(value, f"qmap[{i}]") for i, value in enumerate(inp["qmap"])]
    canonical = QMAP_SIGNED if inp["signed"] else QMAP_UNSIGNED
    if qmap != canonical:
        raise Unsupported("a mutated or replacement qmap is a different state-copy task")
    return src, inp["signed"], block_size


def _scale_and_normalize(src, block_size):
    scales, normalized = [], []
    for offset in range(0, len(src), block_size):
        block = src[offset:offset + block_size]
        scale = max(max(abs(value) for value in block), SCALE_MIN)
        scales.append(scale)
        for value in block:
            scaled = _round_f32(value / scale, "block normalization")
            # CUDA may flush subnormal arithmetic. Excluding it makes the exact code result portable across the
            # retained CPU and CUDA executions.
            if scaled and abs(scaled) < MIN_NORMAL_F32:
                raise Unsupported("a normalized value is float32 subnormal")
            normalized.append(scaled)
    return scales, normalized


def _code(value, qmap, *, floor=False):
    code = max((i for i, mapped in enumerate(qmap) if value >= mapped), default=0)
    if floor:
        return code
    up = min(code + 1, 15)
    residual = _round_f32(value - qmap[code], "quantization residual")
    gap = _round_f32(qmap[up] - qmap[code], "quantization gap")
    threshold = _round_f32(gap * Fraction(1, 2), "quantization midpoint")
    return up if residual >= threshold else code


def _state(src, signed, block_size, mode=None):
    qmap = list(QMAP_SIGNED if signed else QMAP_UNSIGNED)
    if mode == "unsigned_zero_map" and not signed:
        qmap = [Fraction(k, 15) for k in range(16)]
    scales, normalized = _scale_and_normalize(src, len(src) if mode == "one_scale" else block_size)
    unpacked = [_code(value, qmap, floor=(mode == "floor")) for value in normalized]
    if mode == "swap_nibbles":
        codes = [(lo << 4) | hi for hi, lo in zip(unpacked[::2], unpacked[1::2])]
    else:
        codes = [(hi << 4) | lo for hi, lo in zip(unpacked[::2], unpacked[1::2])]
    result = {
        "codes": codes,
        "scale": scales,
    }
    if mode == "reverse_bytes":
        result["codes"] = list(reversed(result["codes"]))
    if mode == "zero_scale":
        result["scale"] = [Fraction(0) for _ in result["scale"]]
    return result


def reference(profile, inp):
    src, signed, block_size = _parse(profile, inp)
    return Result("value", _state(src, signed, block_size))


def _tensor_record(value, *, dtype, length, name, discrete=False):
    if isinstance(value, dict):
        if set(value) != {"t", "shape", "v"}:
            raise Invalid(f"observed {name} tensor record is incomplete")
        if value["t"] != dtype or value["shape"] != [length]:
            raise Invalid(f"observed {name} has the wrong dtype or shape")
        value = value["v"]
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise Invalid(f"observed {name} has the wrong length")
    if discrete:
        out = [integer(item, f"observed {name}", lo=0) for item in value]
        if any(item > 255 for item in out):
            raise Invalid(f"observed {name} is outside uint8")
        return out
    return [_f32(item, f"observed {name}") for item in value]


def check_output(profile, inp, observed):
    try:
        src, signed, block_size = _parse(profile, inp)
        if not isinstance(observed, dict) or set(observed) != {"codes", "scale"}:
            return False
        candidate = {
            "codes": _tensor_record(observed["codes"], dtype="u8", length=len(src) // 2,
                                    name="codes", discrete=True),
            "scale": _tensor_record(observed["scale"], dtype="f32", length=len(src) // block_size,
                                    name="scale"),
        }
    except (Invalid, Unsupported):
        return False
    return candidate == reference(profile, inp).fields


def _mutant(mode):
    def run(profile, inp):
        src, signed, block_size = _parse(profile, inp)
        return Result("value", _state(src, signed, block_size, mode))
    return run


MUTANTS = {
    "Q4.M1": _mutant("swap_nibbles"),
    "Q4.M2": _mutant("one_scale"),
    "Q4.M3": _mutant("floor"),
    "Q4.M4": _mutant("unsigned_zero_map"),
    "Q4.M5": _mutant("reverse_bytes"),
    "Q4.M6": _mutant("zero_scale"),
}
READINGS = {}
