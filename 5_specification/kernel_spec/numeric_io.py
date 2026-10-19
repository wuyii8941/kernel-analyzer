"""JSON records for exact finite rationals and outward interval enclosures.

Integer numerator/denominator strings use hexadecimal solely to represent
arbitrarily large integers without JSON floating-point conversion or Python's
decimal digit limit. They are numbers, not fingerprints. Interval endpoints
come from num.ends; decoding rounds outward at the current working precision.
Thus a restored interval contains the saved enclosure (it may be wider).
This format carries values, not evidence that a reference is correct.
"""
import re
from fractions import Fraction

from .num import IV, Invalid, Undefined, ends, from_interval
from .real import R

_HEX = re.compile(r'-?0x(?:0|[1-9a-f][0-9a-f]*)\Z')


def _rational_record(q):
    return {'kind': 'rational', 'numerator_hex': hex(int(q.numerator)),
            'denominator_hex': hex(int(q.denominator))}


def encode_finite(value):
    try:
        value = R(value)
    except (TypeError, ValueError, Undefined) as exc:
        raise Invalid('cannot persist a non-finite numeric value') from exc
    if not value.finite:
        raise Invalid('cannot persist infinity as a finite value')
    if isinstance(value.v, IV):
        lo, hi = ends(value.v)
        return {'kind': 'interval', 'lower': _rational_record(lo), 'upper': _rational_record(hi)}
    return _rational_record(value.v)


def _read_rational(record):
    if not isinstance(record, dict) or set(record) != {'kind', 'numerator_hex', 'denominator_hex'}:
        raise Invalid('expected complete rational record')
    if record['kind'] != 'rational':
        raise Invalid('interval endpoints must be exact rational records')
    values = []
    for field in ('numerator_hex', 'denominator_hex'):
        text = record[field]
        if not isinstance(text, str) or not _HEX.fullmatch(text):
            raise Invalid('expected hexadecimal integer string')
        values.append(int(text, 16))
    if values[1] <= 0:
        raise Invalid('rational denominator must be positive')
    return Fraction(*values)


def decode_finite(record):
    if not isinstance(record, dict):
        raise Invalid('numeric record must be a mapping')
    if record.get('kind') == 'rational':
        return _read_rational(record)
    if record.get('kind') != 'interval' or set(record) != {'kind', 'lower', 'upper'}:
        raise Invalid('expected complete rational or interval record')
    lo, hi = _read_rational(record['lower']), _read_rational(record['upper'])
    if lo > hi:
        raise Invalid('interval lower endpoint exceeds upper endpoint')
    return from_interval(lo, hi)
