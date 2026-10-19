"""Draft A1 core adapter. Complete named outputs; no historical imports or fallback."""
import inspect
from copy import deepcopy

from kernel_spec import linear_attention as L,d2_common as U
from kernel_spec.num import Invalid,Unsupported,Result,require_fields

PROFILES={'unnormalized_core','normalized_core','scalar_decay_core','diagonal_decay_core',
          'delta_core','gated_delta_core','kda_core'}


def reference(profile,inp):
    if profile not in PROFILES:
        raise Unsupported(f'unknown A1 profile {profile!r}')
    require_fields(inp,{'q','k','v','operation'})
    op=inp['operation']
    if op not in ('forward','backward'):
        raise Unsupported('A1 operation must be forward or backward')
    varlen='cu_seqlens' in inp
    args={k:v for k,v in inp.items() if k!='operation'}
    if profile=='unnormalized_core':
        if op=='backward':fn=L.backward_varlen if varlen else L.backward
        elif varlen:fn=L.varlen
        else:fn=L.chunked if 'chunk_size' in args else L.recurrent
    elif profile=='normalized_core':
        fn=(L.normalized_backward_varlen if varlen else L.normalized_backward) if op=='backward' else (L.normalized_varlen if varlen else L.normalized)
    elif profile in ('scalar_decay_core','diagonal_decay_core'):
        fn=(L.decayed_backward_varlen if varlen else L.decayed_backward) if op=='backward' else (L.decayed_varlen if varlen else L.decayed)
    else:
        fn=(L.delta_backward_varlen if varlen else L.delta_backward) if op=='backward' else (L.delta_varlen if varlen else L.delta)
    signature=inspect.signature(fn)
    extra=set(args)-set(signature.parameters)
    if extra or 'profile' in args:
        raise Unsupported(f'parameters outside selected A1 operation: {sorted(extra | ({"profile"} if "profile" in args else set()))}')
    args['profile']=profile
    try:signature.bind(**args)
    except TypeError as exc:raise Invalid(str(exc)) from exc
    result=fn(**args)
    return Result('value',{'result':{key:U.plain(value) for key,value in result.items()}})


def check_output(profile,inp,observed):
    """Compare every declared result, including final states and initial-state VJPs.

    Containment in a nonpoint enclosure does not prove equality. Contract
    admission and correspondence to an actual kernel remain separate checks.
    """
    from kernel_spec.compare import check_enclosed_output
    return check_enclosed_output(reference(profile,inp).fields,observed)


def _omit(*names):
    def mutant(profile,inp):return reference(profile,{k:v for k,v in inp.items() if k not in names})
    return mutant


def _scale_one(profile,inp):return reference(profile,dict(inp,scale=1))


def _first_coordinate(profile,inp):
    result=reference(profile,inp)
    fields=deepcopy(result.fields)
    fields['result']['out']=[row[:1] for row in fields['result']['out']]
    return Result('value',fields)


def _scale_epsilon(profile,inp):
    # Diagnostic alternative: scale*(q.Z+epsilon), rather than scale*q.Z+epsilon.
    return reference(profile,dict(inp,epsilon=U.leaf(inp['epsilon'],'epsilon')*
                                  U.leaf(inp['scale'],'scale')))


MUTANTS={'A1.M1':_omit('S0','S0_list'),'A1.M2':_omit('dST','dST_list','dZT','dZT_list'),
         'A1.M3':_scale_one,'A1.M4':_omit('Z0','Z0_list'),'A1.M5':_first_coordinate,
         'A1.M6':_scale_epsilon}
READINGS={}
