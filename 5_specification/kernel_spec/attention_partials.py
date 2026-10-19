"""Finite-real partial attention; no registered contract or kernel binding yet.

A nonempty state (shift, mass, numerator) represents
  mass = sum_j exp(score_j - shift),
  numerator[c] = sum_j exp(score_j - shift) * value[j,c].
The shift need not be the maximum. Changing it rescales both sums equally.
This is the unnormalized accumulator convention in retained VLLM_UA.py.gz,
lines 591-645 and 734-764. `normalized` is a separate, explicit conversion
from an output vector plus natural-log mass, as in the historical split-KV
interface. Neither convention can be inferred from a kernel's name.

Empty mass has shift=None, mass=0 and a full zero numerator. An engine's
infinity sentinel is accepted only by normalized(..., is_empty=True), never
used in arithmetic. All-empty output is undefined unless zero is selected.
All states are validated before merging, including empty participants.
Intervals remain numeric values; display strings are not parsed. dumps_state /
loads_state persist all coordinates using exact rational endpoint records;
restoring at lower precision may widen an enclosure but cannot narrow it.
This helper does not certify disjoint keys, completeness, buffer layouts,
log2 engine values, quantization, or any actual producer/consumer pairing.
"""
import json

from . import real as rl
from .numeric_io import encode_finite, decode_finite
from .num import Invalid, NotEstablished, Undefined, integer
from .real import R


def _finite(value):
    try:
        result = R(value)
    except (TypeError, ValueError, Undefined) as exc:
        raise Invalid('partial requires finite numeric values') from exc
    if not result.finite:
        raise Invalid('partial requires finite numeric values')
    return result


def _vector(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise Invalid('partial numerator/output must be a nonempty vector')
    return [_finite(x) for x in values]


def empty(width):
    width = integer(width, 'value width', lo=1)
    return {'shift': None, 'mass': 0, 'numerator': [0]*width}


def scaled(shift, mass, numerator):
    """Convert a shifted, unnormalized accumulator. Nonempty mass must be positive."""
    shift, mass, numerator = _finite(shift), _finite(mass), _vector(numerator)
    lo, hi = mass.bounds()
    if hi <= 0:
        raise Invalid('nonempty partial requires positive mass')
    if lo <= 0:
        raise NotEstablished('mass enclosure does not establish positivity')
    return {'shift': shift.v, 'mass': mass.v, 'numerator': [x.v for x in numerator]}


def normalized(out, lse, *, is_empty=False):
    """Convert normalized output and natural-log mass; emptiness is an explicit fact.

    With is_empty=True, finite placeholder output coordinates are checked but
    discarded. The sentinel is required to be an explicit infinity. This does
    not establish that an engine's sentinel actually denotes an empty key set.
    """
    if type(is_empty) is not bool:
        raise Invalid('is_empty must be boolean')
    out = _vector(out)
    if is_empty:
        if not isinstance(lse, str) or lse not in ('+inf', '-inf'):
            raise Invalid('empty normalized partial requires an explicit infinity sentinel')
        return empty(len(out))
    return scaled(lse, 1, out)


def _validate(part):
    if not isinstance(part, dict) or set(part) != {'shift', 'mass', 'numerator'}:
        raise Invalid('partial requires exactly shift, mass and numerator')
    if part['shift'] is not None:
        return scaled(part['shift'], part['mass'], part['numerator'])
    values = [_finite(part['mass']), *_vector(part['numerator'])]
    if any(x.bounds() != (0, 0) for x in values):
        raise Invalid('empty partial must have zero mass and complete zero numerator')
    return empty(len(values)-1)


def merge(parts):
    if not isinstance(parts, (list, tuple)) or not parts:
        raise Invalid('merge requires a nonempty sequence of partials')
    parts = [_validate(p) for p in parts]
    width = len(parts[0]['numerator'])
    if any(len(p['numerator']) != width for p in parts):
        raise Invalid('all partials must have the same value width')
    active = [p for p in parts if p['shift'] is not None]
    if not active:
        return empty(width)
    if len(active) == 1:
        return active[0]
    # Any finite shift is valid. Upper bounds avoid an undecidable interval max.
    shift = max(R(p['shift']).bounds()[1] for p in active)
    factors = [rl.exp(R(p['shift']) - shift) for p in active]
    mass = sum((a*R(p['mass']) for a, p in zip(factors, active)), R(0))
    numerator = [sum((a*R(p['numerator'][c]) for a, p in zip(factors, active)), R(0))
                 for c in range(width)]
    return scaled(shift, mass, numerator)


def finish(part, *, empty_policy='undefined'):
    if empty_policy not in ('undefined', 'zero'):
        raise Invalid('empty_policy must be undefined or zero')
    part = _validate(part)
    if part['shift'] is None:
        if empty_policy == 'undefined':
            raise Undefined('attention with zero total mass')
        return {'out': part['numerator'], 'lse': '-inf'}
    mass = R(part['mass'])
    return {'out': [(R(x)/mass).v for x in part['numerator']],
            'lse': (R(part['shift']) + rl.log(mass)).v}


def dumps_state(part):
    """Persist every state coordinate without decimal display conversions."""
    part = _validate(part)
    state = {'shift': None if part['shift'] is None else encode_finite(part['shift']),
             'mass': encode_finite(part['mass']),
             'numerator': [encode_finite(x) for x in part['numerator']]}
    return json.dumps({'format': 'attention_partial', 'version': 1,
                       'width': len(part['numerator']), 'state': state}, allow_nan=False)


def loads_state(text):
    """Restore an enclosing state; reject incomplete records and duplicate fields."""
    if not isinstance(text, str):
        raise Invalid('saved state must be JSON text')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Invalid('duplicate field in saved state')
            result[key] = value
        return result

    def nonfinite(_):
        raise Invalid('non-finite JSON literal in saved state')

    try:
        record = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, RecursionError) as exc:
        raise Invalid('malformed saved state') from exc
    if not isinstance(record, dict) or set(record) != {'format', 'version', 'width', 'state'}:
        raise Invalid('incomplete saved state record')
    if (record['format'] != 'attention_partial' or type(record['version']) is not int
            or record['version'] != 1):
        raise Invalid('unsupported saved state format/version')
    width = integer(record['width'], 'saved value width', lo=1)
    state = record['state']
    if not isinstance(state, dict) or set(state) != {'shift', 'mass', 'numerator'}:
        raise Invalid('incomplete saved state fields')
    if not isinstance(state['numerator'], list) or len(state['numerator']) != width:
        raise Invalid('saved state numerator does not match declared width')
    return _validate({'shift': None if state['shift'] is None else decode_finite(state['shift']),
                      'mass': decode_finite(state['mass']),
                      'numerator': [decode_finite(x) for x in state['numerator']]})
