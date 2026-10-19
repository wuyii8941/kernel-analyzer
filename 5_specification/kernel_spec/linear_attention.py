"""Current A1 migration: linear, normalized, decay and delta cores.

S[t] = S[t-1] + k[t] v[t]^T; out[t] = scale * S[t]^T q[t].
The retained FLA linear_attn/naive.py and fused_recurrent.py use these updates
and default scale=1/sqrt(K) for normalize=False. The normalized core additionally
tracks Z and requires an explicit epsilon. This is not the complete
normalized paper algorithm, and not a registered or certified contract.
All values are finite exact inputs; irrational default scales are enclosed.
Backward treats scale as fixed and returns every q/k/v/initial-state gradient
of <dO,out> + <dST,final_state>; the normalized backward also includes
<dZT,final_Z> and returns dZ0.
The scalar/diagonal decay helpers additionally return gradients in a, or g when
ln/log2 parameterization is explicitly selected. Delta cores reuse d2_linear,
add the beta gradient, and distinguish scalar decay from KDA's key-wise decay.
"""
from . import d2_common as U
from . import real as rl
from . import d2_linear as DL
from .num import Invalid, Unsupported, integer
from .real import R


def _zeros(rows,cols):
    return [[R(0) for _ in range(cols)] for _ in range(rows)]


def _inputs(q,k,v,S0,scale,profile):
    if profile != 'unnormalized_core':
        raise Unsupported('only the unnormalized_core profile is implemented')
    if isinstance(q,(list,tuple)) and len(q)==0:
        if not isinstance(k,(list,tuple)) or not isinstance(v,(list,tuple)) or len(k) or len(v):
            raise Invalid('empty q requires empty k and v')
        if S0 is None:
            raise Invalid('empty sequence requires S0 to declare state dimensions')
        state,shape=U.tensor(S0,2,'S0')
        K,V=shape;T=0;q,k,v=[],[],[]
    else:
        q,(T,K)=U.tensor(q,2,'q')
        k,_=U.tensor(k,2,'k',shape=(T,K))
        v,(_,V)=U.tensor(v,2,'v',shape=(T,None))
        state=_zeros(K,V) if S0 is None else U.tensor(S0,2,'S0',shape=(K,V))[0]
    if K<=0 or V<=0:
        raise Invalid('key and value dimensions must be positive')
    scale=R(1)/rl.sqrt(R(K)) if scale is None else U.leaf(scale,'scale')
    return q,k,v,state,scale,T,K,V


def _advance(state,k,v,K,V):
    return [[state[i][j]+k[i]*v[j] for j in range(V)] for i in range(K)]


def recurrent(q,k,v,*,S0=None,scale=None,profile='unnormalized_core'):
    return _recurrent(_inputs(q,k,v,S0,scale,profile))


def _recurrent(inputs):
    q,k,v,state,scale,T,K,V=inputs
    output=[]
    for t in range(T):
        state=_advance(state,k[t],v[t],K,V)
        output.append([scale*U.rsum([q[t][i]*state[i][j] for i in range(K)]) for j in range(V)])
    return {'out':output,'state':state}


def chunked(q,k,v,chunk_size,*,S0=None,scale=None,profile='unnormalized_core'):
    size=integer(chunk_size,'chunk_size',lo=1)
    return _chunked(_inputs(q,k,v,S0,scale,profile),size)


def _chunked(inputs,size):
    q,k,v,state,scale,T,K,V=inputs
    output=[]
    for start in range(0,T,size):
        end=min(T,start+size)
        for t in range(start,end):
            weights=[U.rsum([q[t][i]*k[s][i] for i in range(K)]) for s in range(start,t+1)]
            output.append([scale*(U.rsum([q[t][i]*state[i][j] for i in range(K)])+
                                 U.rsum([weights[s-start]*v[s][j] for s in range(start,t+1)])) for j in range(V)])
        state=[[state[i][j]+U.rsum([k[s][i]*v[s][j] for s in range(start,end)]) for j in range(V)] for i in range(K)]
    return {'out':output,'state':state}


def backward(q,k,v,dO,*,dST=None,S0=None,scale=None,profile='unnormalized_core'):
    q,k,v,state,scale,T,K,V=_inputs(q,k,v,S0,scale,profile)
    dO,carry=_cotangents(dO,dST,T,K,V)
    return _backward((q,k,v,state,scale,T,K,V),dO,carry)


def _cotangents(dO,dST,T,K,V):
    if T==0:
        if not isinstance(dO,(list,tuple)) or len(dO):
            raise Invalid('empty sequence requires empty dO')
        dO=[]
    else:
        dO,_=U.tensor(dO,2,'dO',shape=(T,V))
    carry=_zeros(K,V) if dST is None else U.tensor(dST,2,'dST',shape=(K,V))[0]
    return dO,carry


def _backward(inputs,dO,carry):
    q,k,v,state,scale,T,K,V=inputs
    states=[]
    for t in range(T):
        state=_advance(state,k[t],v[t],K,V)
        states.append(state)
    dq,dk,dv=[None]*T,[None]*T,[None]*T
    for t in reversed(range(T)):
        carry=[[carry[i][j]+scale*q[t][i]*dO[t][j] for j in range(V)] for i in range(K)]
        dq[t]=[scale*U.rsum([states[t][i][j]*dO[t][j] for j in range(V)]) for i in range(K)]
        dk[t]=[U.rsum([carry[i][j]*v[t][j] for j in range(V)]) for i in range(K)]
        dv[t]=[U.rsum([carry[i][j]*k[t][i] for i in range(K)]) for j in range(V)]
    return {'dq':dq,'dk':dk,'dv':dv,'dS0':carry}



def _varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,profile):
    if not isinstance(cu_seqlens,(list,tuple)) or len(cu_seqlens)<2:
        raise Invalid('cu_seqlens must declare at least one sequence')
    offsets=[integer(x,'cu_seqlens',lo=0) for x in cu_seqlens]
    if offsets[0]!=0 or any(b<a for a,b in zip(offsets,offsets[1:])):
        raise Invalid('cu_seqlens must start at zero and be nondecreasing')
    N=len(offsets)-1
    if S0_list is not None and (not isinstance(S0_list,(list,tuple)) or len(S0_list)!=N):
        raise Invalid('S0_list must contain exactly one state per sequence')
    initial=None if S0_list is None else S0_list[0]
    q,k,v,_,scale,T,K,V=_inputs(q,k,v,initial,scale,profile)
    if offsets[-1]!=T:
        raise Invalid('cu_seqlens must end at the full token count')
    states=[_zeros(K,V) for _ in range(N)] if S0_list is None else U.tensor(S0_list,3,'S0_list',shape=(N,K,V))[0]
    return q,k,v,states,scale,T,K,V,offsets


def varlen(q,k,v,cu_seqlens,*,S0_list=None,scale=None,chunk_size=None,profile='unnormalized_core'):
    """Flattened sequences, each with its own state; repeated offsets keep empty segments.

    When chunk_size is absent use the recurrence, otherwise the chunk formula.
    Both return all token outputs and one final state per declared sequence.
    """
    size=None if chunk_size is None else integer(chunk_size,'chunk_size',lo=1)
    q,k,v,states,scale,T,K,V,offsets=_varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,profile)
    output,finals=[],[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        inputs=(q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V)
        result=_recurrent(inputs) if size is None else _chunked(inputs,size)
        output.extend(result['out']);finals.append(result['state'])
    return {'out':output,'states':finals}


def backward_varlen(q,k,v,dO,cu_seqlens,*,dST_list=None,S0_list=None,scale=None,profile='unnormalized_core'):
    """VJP of all flattened outputs and every final state, including empty segments."""
    q,k,v,states,scale,T,K,V,offsets=_varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,profile)
    dO,_=_cotangents(dO,None,T,K,V)
    N=len(states)
    carries=[_zeros(K,V) for _ in range(N)] if dST_list is None else U.tensor(dST_list,3,'dST_list',shape=(N,K,V))[0]
    dq,dk,dv,initial_grads=[],[],[],[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        result=_backward((q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V),dO[start:end],carries[n])
        dq.extend(result['dq']);dk.extend(result['dk']);dv.extend(result['dv']);initial_grads.append(result['dS0'])
    return {'dq':dq,'dk':dk,'dv':dv,'dS0_list':initial_grads}


def _normalized_inputs(q,k,v,S0,Z0,scale,epsilon,profile):
    if profile != 'normalized_core':
        raise Unsupported('normalized evaluation requires the normalized_core profile')
    inputs=_inputs(q,k,v,S0,scale,'unnormalized_core')
    K=inputs[6]
    z=[R(0) for _ in range(K)] if Z0 is None else U.tensor(Z0,1,'Z0',shape=(K,))[0]
    return inputs,z,_epsilon(epsilon)


def _epsilon(epsilon):
    eps=U.leaf(epsilon,'epsilon')
    if eps.sign()<0:
        raise Invalid('epsilon must be nonnegative')
    return eps


def _normalized_trace(inputs,z,epsilon,size=None):
    q,k,v,state,scale,T,K,V=inputs
    numerator=_recurrent(inputs) if size is None else _chunked(inputs,size)
    output,zs,denominators=[],[],[]
    for t in range(T):
        z=[z[i]+k[t][i] for i in range(K)]
        denominator=scale*U.rsum([q[t][i]*z[i] for i in range(K)])+epsilon
        output.append([value/denominator for value in numerator['out'][t]])
        zs.append(z);denominators.append(denominator)
    return {'out':output,'state':numerator['state'],'zstate':z},numerator['out'],zs,denominators


def normalized(q,k,v,*,epsilon,S0=None,Z0=None,scale=None,chunk_size=None,profile='normalized_core'):
    """Explicit task: y=scale*S^T*q/(scale*<q,Z>+epsilon), Z accumulates k.

    epsilon must be selected by the caller: no default equates the FLA epsilon
    convention with the paper's normalized algorithm. q/k are supplied feature
    vectors; feature-map selection and kernel binding are outside this helper.
    Empty sequences retain both initial states. Zero denominators are undefined.
    """
    size=None if chunk_size is None else integer(chunk_size,'chunk_size',lo=1)
    inputs,z,eps=_normalized_inputs(q,k,v,S0,Z0,scale,epsilon,profile)
    return _normalized_trace(inputs,z,eps,size)[0]


def normalized_backward(q,k,v,dO,*,epsilon,dST=None,dZT=None,S0=None,Z0=None,scale=None,profile='normalized_core'):
    """Full VJP in q,k,v,S0,Z0; epsilon and scale are fixed task parameters."""
    inputs,z,eps=_normalized_inputs(q,k,v,S0,Z0,scale,epsilon,profile)
    q,k,v,state,scale,T,K,V=inputs
    dO,carry=_cotangents(dO,dST,T,K,V)
    dz=[R(0) for _ in range(K)] if dZT is None else U.tensor(dZT,1,'dZT',shape=(K,))[0]
    return _normalized_backward(inputs,z,eps,dO,carry,dz)


def _normalized_backward(inputs,z,eps,dO,carry,dz):
    q,k,v,state,scale,T,K,V=inputs
    _,numerator,zs,denominators=_normalized_trace(inputs,z,eps)
    dN,dD=[],[]
    for t in range(T):
        denominator=denominators[t]
        dN.append([g/denominator for g in dO[t]])
        dD.append(-U.rsum([dO[t][j]*numerator[t][j] for j in range(V)])/(denominator*denominator))
    result=_backward(inputs,dN,carry)
    for t in reversed(range(T)):
        dz=[dz[i]+dD[t]*scale*q[t][i] for i in range(K)]
        result['dq'][t]=[result['dq'][t][i]+dD[t]*scale*zs[t][i] for i in range(K)]
        result['dk'][t]=[result['dk'][t][i]+dz[i] for i in range(K)]
    result['dZ0']=dz
    return result


def _normalized_varlen_inputs(q,k,v,cu_seqlens,S0_list,Z0_list,scale,epsilon,profile):
    if profile != 'normalized_core':
        raise Unsupported('normalized evaluation requires the normalized_core profile')
    inputs=_varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,'unnormalized_core')
    K=inputs[6];N=len(inputs[8])-1
    zs=_zeros(N,K) if Z0_list is None else U.tensor(Z0_list,2,'Z0_list',shape=(N,K))[0]
    return inputs,zs,_epsilon(epsilon)


def normalized_varlen(q,k,v,cu_seqlens,*,epsilon,S0_list=None,Z0_list=None,scale=None,chunk_size=None,profile='normalized_core'):
    """Normalized flattened sequences with one S and Z state per segment.

    Repeated boundaries retain both states, including leading/trailing empty
    segments. The explicit epsilon and feature-vector convention match normalized.
    """
    size=None if chunk_size is None else integer(chunk_size,'chunk_size',lo=1)
    inputs,zs,eps=_normalized_varlen_inputs(q,k,v,cu_seqlens,S0_list,Z0_list,scale,epsilon,profile)
    q,k,v,states,scale,T,K,V,offsets=inputs
    output,finals,zfinals=[],[],[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        segment=(q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V)
        result=_normalized_trace(segment,zs[n],eps,size)[0]
        output.extend(result['out']);finals.append(result['state']);zfinals.append(result['zstate'])
    return {'out':output,'states':finals,'zstates':zfinals}


def normalized_backward_varlen(q,k,v,dO,cu_seqlens,*,epsilon,dST_list=None,dZT_list=None,S0_list=None,Z0_list=None,scale=None,profile='normalized_core'):
    """VJP of all normalized outputs and both final states of every segment."""
    inputs,zs,eps=_normalized_varlen_inputs(q,k,v,cu_seqlens,S0_list,Z0_list,scale,epsilon,profile)
    q,k,v,states,scale,T,K,V,offsets=inputs
    dO,_=_cotangents(dO,None,T,K,V)
    N=len(states)
    carries=[_zeros(K,V) for _ in range(N)] if dST_list is None else U.tensor(dST_list,3,'dST_list',shape=(N,K,V))[0]
    dzs=_zeros(N,K) if dZT_list is None else U.tensor(dZT_list,2,'dZT_list',shape=(N,K))[0]
    dq,dk,dv,initial_grads,zinitial_grads=[],[],[],[],[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        segment=(q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V)
        result=_normalized_backward(segment,zs[n],eps,dO[start:end],carries[n],dzs[n])
        dq.extend(result['dq']);dk.extend(result['dk']);dv.extend(result['dv'])
        initial_grads.append(result['dS0']);zinitial_grads.append(result['dZ0'])
    return {'dq':dq,'dk':dk,'dv':dv,'dS0_list':initial_grads,'dZ0_list':zinitial_grads}


def _decay_gates(a,T,K,profile):
    if profile not in ('scalar_decay_core','diagonal_decay_core'):
        raise Unsupported('select scalar_decay_core or diagonal_decay_core')
    scalar=profile=='scalar_decay_core'
    if T==0:
        if not isinstance(a,(list,tuple)) or len(a):
            raise Invalid('empty sequence requires empty gates')
        return [],scalar
    gates=U.tensor(a,1 if scalar else 2,'a',shape=(T,) if scalar else (T,K))[0]
    return [[value]*K for value in gates] if scalar else gates,scalar


def _decayed_recurrent(inputs,gates,trace=False):
    q,k,v,state,scale,T,K,V=inputs
    output,states=[],[state]
    for t in range(T):
        state=[[gates[t][i]*state[i][j]+k[t][i]*v[t][j] for j in range(V)] for i in range(K)]
        output.append([scale*U.rsum([q[t][i]*state[i][j] for i in range(K)]) for j in range(V)])
        states.append(state)
    result={'out':output,'state':state}
    return (result,states) if trace else result


def _decayed_chunked(inputs,gates,size):
    q,k,v,state,scale,T,K,V=inputs
    output=[]
    for start in range(0,T,size):
        end=min(T,start+size)
        # Products of the gates strictly after each write, never ratios of
        # cumulative products: a zero gate resets old state without losing kv.
        weights=[];initial=[R(1) for _ in range(K)]
        for t in range(start,end):
            initial=[initial[i]*gates[t][i] for i in range(K)]
            weights=[[w[i]*gates[t][i] for i in range(K)] for w in weights]
            weights.append([R(1) for _ in range(K)])
            current=[[initial[i]*state[i][j]+U.rsum([weights[s-start][i]*k[s][i]*v[s][j]
                       for s in range(start,t+1)]) for j in range(V)] for i in range(K)]
            output.append([scale*U.rsum([q[t][i]*current[i][j] for i in range(K)]) for j in range(V)])
        state=current
    return {'out':output,'state':state}


def _convert_decay_gates(gates,parameterization):
    if parameterization not in ('direct','ln','log2'):
        raise Unsupported('gate_parameterization must be direct, ln or log2')
    if parameterization=='direct':
        return gates
    factor=R(1) if parameterization=='ln' else rl.log(R(2))
    return [[rl.exp(g*factor) for g in row] for row in gates]


def _log_gate_gradient(result,gates,scalar,parameterization):
    if parameterization=='direct':
        return result
    factor=R(1) if parameterization=='ln' else rl.log(R(2))
    da=result.pop('da')
    if scalar:
        result['dg']=[d*gates[t][0]*factor for t,d in enumerate(da)]
    else:
        result['dg']=[[d*gates[t][i]*factor for i,d in enumerate(row)] for t,row in enumerate(da)]
    return result


def decayed(q,k,v,a,*,profile,S0=None,scale=None,chunk_size=None,gate_parameterization='direct'):
    """Direct-gate task S_t=diag(a_t)S_{t-1}+k_t v_t^T (scalar a broadcasts).

    The scalar/diagonal transitions agree with the retained simple_gla/gla
    naive recurrences after their exp(g) conversion. By default a is supplied
    directly; explicit gate_parameterization='ln'/'log2' instead interprets that
    argument as g and computes exp(g)/2**g with strict enclosures. No feature map
    or fused model operation is implied. Any finite direct a is allowed by this
    algebraic task, including zero. This does not
    assert that every such a can arise from a particular model's gate function.
    """
    size=None if chunk_size is None else integer(chunk_size,'chunk_size',lo=1)
    inputs=_inputs(q,k,v,S0,scale,'unnormalized_core')
    gates,_=_decay_gates(a,inputs[5],inputs[6],profile)
    gates=_convert_decay_gates(gates,gate_parameterization)
    return _decayed_recurrent(inputs,gates) if size is None else _decayed_chunked(inputs,gates,size)


def _decayed_backward(inputs,gates,scalar,dO,carry):
    q,k,v,state,scale,T,K,V=inputs
    _,states=_decayed_recurrent(inputs,gates,trace=True)
    dq,dk,dv,da=[None]*T,[None]*T,[None]*T,[None]*T
    for t in reversed(range(T)):
        carry=[[carry[i][j]+scale*q[t][i]*dO[t][j] for j in range(V)] for i in range(K)]
        dq[t]=[scale*U.rsum([states[t+1][i][j]*dO[t][j] for j in range(V)]) for i in range(K)]
        dk[t]=[U.rsum([carry[i][j]*v[t][j] for j in range(V)]) for i in range(K)]
        dv[t]=[U.rsum([carry[i][j]*k[t][i] for i in range(K)]) for j in range(V)]
        rowgrad=[U.rsum([carry[i][j]*states[t][i][j] for j in range(V)]) for i in range(K)]
        da[t]=U.rsum(rowgrad) if scalar else rowgrad
        carry=[[gates[t][i]*carry[i][j] for j in range(V)] for i in range(K)]
    return {'dq':dq,'dk':dk,'dv':dv,'da':da,'dS0':carry}


def decayed_backward(q,k,v,a,dO,*,profile,dST=None,S0=None,scale=None,gate_parameterization='direct'):
    """Full VJP with fixed scale; return da for direct inputs, dg for log inputs."""
    inputs=_inputs(q,k,v,S0,scale,'unnormalized_core')
    T,K,V=inputs[5:]
    gates,scalar=_decay_gates(a,T,K,profile)
    dO,carry=_cotangents(dO,dST,T,K,V)
    gates=_convert_decay_gates(gates,gate_parameterization)
    return _log_gate_gradient(_decayed_backward(inputs,gates,scalar,dO,carry),gates,scalar,gate_parameterization)


def decayed_varlen(q,k,v,a,cu_seqlens,*,profile,S0_list=None,scale=None,chunk_size=None,gate_parameterization='direct'):
    """Selected gate recurrence per sequence, retaining every final state."""
    size=None if chunk_size is None else integer(chunk_size,'chunk_size',lo=1)
    q,k,v,states,scale,T,K,V,offsets=_varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,'unnormalized_core')
    gates,_=_decay_gates(a,T,K,profile)
    gates=_convert_decay_gates(gates,gate_parameterization)
    output,finals=[],[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        segment=(q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V)
        result=_decayed_recurrent(segment,gates[start:end]) if size is None else _decayed_chunked(segment,gates[start:end],size)
        output.extend(result['out']);finals.append(result['state'])
    return {'out':output,'states':finals}


def decayed_backward_varlen(q,k,v,a,dO,cu_seqlens,*,profile,dST_list=None,S0_list=None,scale=None,gate_parameterization='direct'):
    """All token/state gradients, including da or dg according to the input."""
    q,k,v,states,scale,T,K,V,offsets=_varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,'unnormalized_core')
    gates,scalar=_decay_gates(a,T,K,profile)
    dO,_=_cotangents(dO,None,T,K,V)
    N=len(states)
    carries=[_zeros(K,V) for _ in range(N)] if dST_list is None else U.tensor(dST_list,3,'dST_list',shape=(N,K,V))[0]
    gates=_convert_decay_gates(gates,gate_parameterization)
    dq,dk,dv,da,initial_grads=[],[],[],[],[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        segment=(q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V)
        result=_decayed_backward(segment,gates[start:end],scalar,dO[start:end],carries[n])
        dq.extend(result['dq']);dk.extend(result['dk']);dv.extend(result['dv']);da.extend(result['da'])
        initial_grads.append(result['dS0'])
    return _log_gate_gradient({'dq':dq,'dk':dk,'dv':dv,'da':da,'dS0_list':initial_grads},gates,scalar,gate_parameterization)


def _delta_parameters(a,beta,T,K,profile,parameterization):
    if profile not in ('delta_core','gated_delta_core','kda_core'):
        raise Unsupported('select delta_core, gated_delta_core or kda_core')
    if T==0:
        if not isinstance(beta,(list,tuple)) or len(beta):
            raise Invalid('empty sequence requires empty beta')
        betas=[]
    else:
        betas=U.tensor(beta,1,'beta',shape=(T,))[0]
    if profile=='delta_core':
        if a is not None or parameterization!='direct':
            raise Unsupported('delta_core has no decay input; select a gated profile')
        return [[R(1)]*K for _ in range(T)],True,betas
    decay_profile='scalar_decay_core' if profile=='gated_delta_core' else 'diagonal_decay_core'
    gates,scalar=_decay_gates(a,T,K,decay_profile)
    return gates,scalar,betas


def _delta_forward(inputs,gates,scalar,beta):
    q,k,v,state,scale,T,K,V=inputs
    alpha=[row[0] for row in gates] if scalar else gates
    output,final=DL.recurrent(q,k,v,alpha,beta,scale,state)
    return {'out':output,'state':final}


def _delta_backward(inputs,gates,scalar,beta,dO,carry,profile,parameterization):
    q,k,v,state,scale,T,K,V=inputs
    alpha=[row[0] for row in gates] if scalar else gates
    result=DL.backward(q,k,v,alpha,beta,scale,state,dO,carry)
    da=result.pop('dalpha')
    if profile!='delta_core':
        result['da']=da
        result=_log_gate_gradient(result,gates,scalar,parameterization)
    return result


def delta(q,k,v,beta,*,profile,a=None,S0=None,scale=None,gate_parameterization='direct'):
    """Selected core: A=diag(a)S; S=A+beta*k*(v-k^T*A)^T.

    delta_core fixes a=1; gated_delta_core uses scalar a; kda_core uses a per key.
    q/k are supplied features, beta is supplied directly, and no normalization,
    sigmoid, head expansion or state transpose is implicit. Full outputs and
    final state are returned. This helper does not certify a complete model.
    """
    inputs=_inputs(q,k,v,S0,scale,'unnormalized_core')
    gates,scalar,betas=_delta_parameters(a,beta,inputs[5],inputs[6],profile,gate_parameterization)
    gates=_convert_decay_gates(gates,gate_parameterization)
    return _delta_forward(inputs,gates,scalar,betas)


def delta_backward(q,k,v,beta,dO,*,profile,a=None,dST=None,S0=None,scale=None,gate_parameterization='direct'):
    """VJP in every supplied core input, including beta and both uses of k."""
    inputs=_inputs(q,k,v,S0,scale,'unnormalized_core')
    T,K,V=inputs[5:]
    gates,scalar,betas=_delta_parameters(a,beta,T,K,profile,gate_parameterization)
    dO,carry=_cotangents(dO,dST,T,K,V)
    gates=_convert_decay_gates(gates,gate_parameterization)
    return _delta_backward(inputs,gates,scalar,betas,dO,carry,profile,gate_parameterization)


def delta_varlen(q,k,v,beta,cu_seqlens,*,profile,a=None,S0_list=None,scale=None,gate_parameterization='direct'):
    q,k,v,states,scale,T,K,V,offsets=_varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,'unnormalized_core')
    gates,scalar,betas=_delta_parameters(a,beta,T,K,profile,gate_parameterization)
    gates=_convert_decay_gates(gates,gate_parameterization)
    output,finals=[],[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        segment=(q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V)
        result=_delta_forward(segment,gates[start:end],scalar,betas[start:end])
        output.extend(result['out']);finals.append(result['state'])
    return {'out':output,'states':finals}


def delta_backward_varlen(q,k,v,beta,dO,cu_seqlens,*,profile,a=None,dST_list=None,S0_list=None,scale=None,gate_parameterization='direct'):
    q,k,v,states,scale,T,K,V,offsets=_varlen_inputs(q,k,v,cu_seqlens,S0_list,scale,'unnormalized_core')
    gates,scalar,betas=_delta_parameters(a,beta,T,K,profile,gate_parameterization)
    dO,_=_cotangents(dO,None,T,K,V)
    N=len(states)
    carries=[_zeros(K,V) for _ in range(N)] if dST_list is None else U.tensor(dST_list,3,'dST_list',shape=(N,K,V))[0]
    gates=_convert_decay_gates(gates,gate_parameterization)
    result={key:[] for key in ('dq','dk','dv','dbeta','dS0_list')}
    if profile!='delta_core':result['da' if gate_parameterization=='direct' else 'dg']=[]
    for n,(start,end) in enumerate(zip(offsets,offsets[1:])):
        segment=(q[start:end],k[start:end],v[start:end],states[n],scale,end-start,K,V)
        grad=_delta_backward(segment,gates[start:end],scalar,betas[start:end],dO[start:end],carries[n],profile,gate_parameterization)
        result['dS0_list'].append(grad.pop('dS0'))
        for key,value in grad.items():result[key].extend(value)
    return result
