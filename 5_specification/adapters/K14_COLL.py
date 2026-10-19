"""K14 explicit partial-merge and collective value tasks; no communication certification."""
from kernel_spec import attention_partials as P
from kernel_spec import collectives as C
from kernel_spec.compare import check_enclosed_output
from kernel_spec.num import Invalid, Unsupported, Result, require_fields


def _parts(profile, inp):
    if profile not in ('scaled_numerator', 'normalized_lse', 'scaled_numerator_out'):
        raise Unsupported('select an explicit partial attention representation')
    keys = {'operation', 'parts', 'participant_ids', 'empty_policy'}
    require_fields(inp, keys)
    if set(inp) != keys:
        raise Unsupported('parameters outside the selected partial merge task')
    if inp['operation'] != 'merge':
        raise Unsupported('other historical K14 operations have not been migrated')
    parts, ids = inp['parts'], inp['participant_ids']
    if not isinstance(parts, (list, tuple)) or not parts:
        raise Invalid('parts must be a nonempty sequence')
    if not isinstance(ids, (list, tuple)) or len(ids) != len(parts):
        raise Invalid('every partial requires one participant ID')
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise Invalid('participant IDs must be distinct nonblank strings')
    if inp['empty_policy'] not in ('undefined', 'zero'):
        raise Invalid('select undefined or zero for all-empty output')
    if profile in ('scaled_numerator', 'scaled_numerator_out'):
        # merge validates the full records, including empty participants.
        return list(parts)
    converted = []
    for part in parts:
        required = {'out', 'lse', 'is_empty'}
        require_fields(part, required, 'normalized partial')
        if set(part) != required:
            raise Invalid('normalized partial contains undeclared fields')
        converted.append(P.normalized(part['out'], part['lse'], is_empty=part['is_empty']))
    return converted


def reference(profile, inp):
    if profile in C.PROFILES:
        return Result('value', {'rank_outputs': C.reference(profile, inp)})
    parts = _parts(profile, inp)
    fields = P.finish(P.merge(parts), empty_policy=inp['empty_policy'])
    if profile == 'scaled_numerator_out':
        fields = {'out': fields['out']}
    return Result('value', fields)


def check_output(profile, inp, observed):
    # Interval membership alone must never become proof of exact equality.
    return check_enclosed_output(reference(profile, inp).fields, observed)


def _drop_coordinate(profile, inp):
    result = reference(profile, inp)
    return Result('value', dict(result.fields, out=result.fields['out'][:-1]))


def _drop_lse(profile, inp):
    return Result('value', {'out': reference(profile, inp).fields['out']})


def _drop_part(profile, inp):
    return reference(profile, dict(inp, parts=inp['parts'][:-1], participant_ids=inp['participant_ids'][:-1]))


def _ignore_weights(profile, inp):
    return reference('all_reduce_mean', {k: v for k, v in inp.items() if k not in ('weights', 'counts')})


def _reverse_shards(profile, inp):
    return Result('value', {'rank_outputs': reference(profile, inp).fields['rank_outputs'][::-1]})


def _omit_rank(profile, inp):
    return Result('value', {'rank_outputs': reference(profile, inp).fields['rank_outputs'][:-1]})


MUTANTS = {'K14.M1': _drop_coordinate, 'K14.M2': _drop_lse, 'K14.M3': _drop_part,
           'K14.M4': _ignore_weights, 'K14.M5': _reverse_shards, 'K14.M6': _omit_rank}
READINGS = {}
