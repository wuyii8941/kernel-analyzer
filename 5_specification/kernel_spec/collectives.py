"""Selected finite-real collective value tasks, including every participant output.

The supplied participant order is part of the task. This module does not infer
process groups, physical ranks, asynchronous completion or communication effects.
"""
from fractions import Fraction

from .num import Invalid, Undefined, Unsupported, integer, matrix, require_fields, vector

PROFILES = {'all_reduce_sum': 'all_reduce', 'all_reduce_mean': 'all_reduce',
            'all_reduce_weighted_mean': 'all_reduce', 'all_reduce_count_mean': 'all_reduce',
            'reduce_scatter_sum': 'reduce_scatter', 'all_gather_equal': 'all_gather'}


def reference(profile, inp):
    if profile not in PROFILES:
        raise Unsupported('select an explicit collective task')
    weight_field = ('weights' if profile == 'all_reduce_weighted_mean' else
                    'counts' if profile == 'all_reduce_count_mean' else None)
    fields = {'operation', 'participant_ids', 'values'} | ({weight_field} if weight_field else set())
    require_fields(inp, fields, 'collective')
    if set(inp) != fields:
        raise Unsupported('parameters outside the selected collective task')
    if inp['operation'] != PROFILES[profile]:
        raise Unsupported('operation does not match collective profile')
    ids = inp['participant_ids']
    if (not isinstance(ids, (list, tuple)) or not ids
            or any(not isinstance(i, str) or not i.strip() for i in ids)
            or len(set(ids)) != len(ids)):
        raise Invalid('participants must be distinct nonblank strings in declared rank order')
    values = matrix(inp['values'], 'values', rows=len(ids))
    width, participants = len(values[0]), len(ids)
    weights = [Fraction(1)] * participants
    if weight_field:
        weights = vector(inp[weight_field], weight_field, n=participants,
                         **({'item': integer} if weight_field == 'counts' else {}))
        if any(w < 0 for w in weights):
            raise Invalid('weights/counts must be nonnegative')
        if sum(weights) == 0:
            raise Undefined('weighted mean has zero total weight/count')
    if profile == 'reduce_scatter_sum' and width % participants:
        raise Invalid('reduce-scatter requires equally sized shards')
    if profile == 'all_gather_equal':
        gathered = [x for row in values for x in row]
        return [list(gathered) for _ in ids]
    denominator = (sum(weights) if weight_field else
                   participants if profile == 'all_reduce_mean' else 1)
    total = [sum((weights[r]*values[r][c] for r in range(participants)), Fraction(0))/denominator
             for c in range(width)]
    if profile == 'reduce_scatter_sum':
        size = width // participants
        return [total[r*size:(r+1)*size] for r in range(participants)]
    return [list(total) for _ in ids]
