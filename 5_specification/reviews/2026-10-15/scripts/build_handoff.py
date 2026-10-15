from pathlib import Path
import json,yaml,hashlib
R=Path(__file__).resolve().parent
D=R/'handoff'
for x in ['contracts','tools','tests','evidence']:(D/x).mkdir(parents=True,exist_ok=True)
HASH='a9a2a9123c2f39f5f76d1c82d61d70869bc0ec8236cce16d978f9556293d15a3'
sources={
 'PCE':{'kind':'versioned_document','title':'PyTorch CrossEntropyLoss','version':'2.10','url':'https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html','reviewed_on':'2026-10-15','verification':'Official versioned page read using web; no original HTML snapshot hash claimed','items':{
 'INDEX':'Definition of unreduced class-index loss and indicator for ignored targets; API documentation loss formula.',
 'REDUCE':'Class-index mean divides by summed target-class weights of non-ignored rows; sum is sum of row losses; none returns row losses.',
 'PROB':'Probability-target loss weights each class term; mean divides by N.',
 'IGNORE':'Ignore index may lie outside the class range; ignored rows do not contribute input gradient; class-index targets only.',
 'SMOOTH':'Targets are mixed with the uniform class distribution.',
 'SHAPE':'Target/index/probability shapes and reduction output shape; probability constraints are caller responsibilities.'}},
 'PAW':{'kind':'versioned_document','title':'PyTorch AdamW','version':'2.10','url':'https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html','reviewed_on':'2026-10-15','verification':'Official versioned algorithm box read using web; no original HTML snapshot hash claimed','items':{
 'DECAY':'Decay old parameter before adaptive parameter update; weight decay is not added to moments.',
 'MOMENTS':'m uses beta1 and gradient; v uses beta2 and gradient squared.',
 'CORRECTION':'Current step t appears in 1-beta**t; amsgrad=False uses the current second moment.',
 'UPDATE':'Subtract lr*m_hat/(sqrt(v_hat)+eps); eps is outside sqrt.',
 'NONE':'The zero_grad note distinguishes None (step skipped) from a zero gradient (full step).'}},
 'HF':{'kind':'supporting_example','title':'Fixing Gradient Accumulation','version':'published 2024-10-16; read 2026-10-15','url':'https://huggingface.co/blog/gradient_accumulation','reviewed_on':'2026-10-15','verification':'Official author blog read using web; supporting task example, not authority for all distributed objectives','items':{'TOKEN_MEAN':'For the discussed token-level task, use accumulated loss sum divided by all valid token counts rather than mean of batch means.'}}
}
(D/'SOURCES.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')

def c(id,basis,statement,fields,evidence='definition'):
 return {'id':id,'basis':basis,'statement':statement,'observes':fields,'evidence_kind':evidence}
def e(id,clauses,inp,expected,profile='base',note=''):
 return {'id':id,'profile':profile,'clauses':clauses,'input':inp,'expected':expected,**({'note':note} if note else {})}
def val(**kw):return {'outcome':'value','fields':kw}
def outcome(s):return {'outcome':s}
def base(id,role,desc,profiles,outputs):
 return {'schema_version':'1.1','id':id,'role':role,'title':desc,
 'review':{'status':'reviewed_for_implementation','date':'2026-10-15','reviewer':'ChatGPT review of uploaded contract draft','input_package_sha256':HASH,'scope':'Contract content and explicit profiles, not all-domain oracle certification'},
 'activation':{'implementation_status':'pending','runtime_binding_status':'pending','requires':['source_profile_matched','input_contract_validated','all_required_outputs_bound','reference_semantics_and_guarantees_validated','no_unresolved_contradiction','contract_and_implementation_hashes_bound']},
 'profiles':profiles,'outputs':outputs,
 'comparison':{'input_binding':'All arithmetic values are the exactly received rational values; published decimal constants are a separate interface query, not silently substituted.','quality':'No default semantic tolerance is inferred from these examples. Numerical resolution and application delta are separate declarations.','test_evidence':'Exact rational equality or certified relation when available; numerical enclosure compatibility is not a proof of equality.','error_handling':'No dropped outputs, silently truncated lists, coerced counts, or exceptions converted by message text.'},
 'decisions':[], 'clauses':[], 'examples':[]}

aw=base('ADAMW','ADAM','Dense AdamW state transition, PyTorch 2.10 mathematical profile',{
 'base':{'meaning':'One step, dense vector, maximize=False, amsgrad=False; arbitrary supplied finite rational pre-state.','must_match':{'maximize':False,'amsgrad':False,'decoupled':True},'task_choice_required':False}},
 {'theta':{'shape':'[d]','guarantee':'rigorous_enclosure'},'m':{'shape':'[d]','guarantee':'exact'},'v':{'shape':'[d]','guarantee':'exact'},'t':{'shape':'scalar integer','guarantee':'exact'}})
aw['signature']={'inputs':['gradient [d] or None','lr','beta1','beta2','eps','weight_decay'], 'pre_state':['theta [d]','m [d]','v [d]','t nonnegative integer'], 'post_state':['theta [d]','m [d]','v [d]','t'], 'observation_unit':'one parameter tensor, one optimizer step; explicit step sequence is composition, not g-vector-as-time'}
aw['domain']={'valid':['d >= 1; theta,m,v,gradient have equal dimensions when gradient is not None','finite rational theta,m,v and hyperparameters; each v_i >= 0','t is a nonnegative integer, not bool; 0 <= beta1,beta2 < 1','lr,weight_decay,eps >= 0'], 'domain_undefined':['gradient is not None and at least one sqrt(v_new_i/(1-beta2**(t+1))) + eps == 0'], 'unsupported':['amsgrad=True','maximize=True for this pilot','sparse gradients','differentiating through the optimizer','unverified interval-valued external prestate'], 'invalid_input':['wrong shapes','negative second moment','nonintegral or negative step counter','NaN/Inf in this finite-real profile','hyperparameters outside the stated range']}
aw['clauses']=[
 c('AW.C1',['PAW.DECAY'],'theta_decay_i = theta_old_i * (1-lr*weight_decay); adaptive step then subtracted from this value',['theta']),
 c('AW.C2',['PAW.MOMENTS','PAW.DECAY'],'m_new_i=beta1*m_old_i+(1-beta1)*g_i; v_new_i=beta2*v_old_i+(1-beta2)*g_i**2; no weight decay contribution in g',['m','v']),
 c('AW.C3',['PAW.CORRECTION'],'t_new=t_old+1; m_hat=m_new/(1-beta1**t_new), v_hat=v_new/(1-beta2**t_new)',['theta','t']),
 c('AW.C4',['PAW.UPDATE'],'theta_new_i=theta_decay_i-lr*m_hat_i/(sqrt(v_hat_i)+eps)',['theta']),
 c('AW.C5',['PAW.NONE'],'gradient=None leaves theta,m,v,t unchanged; gradient=0 still executes the transition',['theta','m','v','t'])]
zero={'theta':[1],'m':[0],'v':[0],'t':0}
def awin(g,state=None,eps=0,beta1='9/10',beta2='99/100'):
 return {'state':state or zero,'gradient':g,'lr':'1/10','beta1':beta1,'beta2':beta2,'eps':eps,'weight_decay':'1/10','maximize':False,'amsgrad':False}
aw['examples']=[
 e('AW.E1',['AW.C1','AW.C2','AW.C3','AW.C4'],awin([2]),val(theta=['89/100'],m=['1/5'],v=['1/25'],t=1)),
 e('AW.E2',['AW.C4'],awin([2],eps='1/10'),val(theta=['1879/2100'],m=['1/5'],v=['1/25'],t=1)),
 e('AW.E3',['AW.C1','AW.C2','AW.C3','AW.C4'],awin([-2],{'theta':[1],'m':[1],'v':[2],'t':1},beta1='1/2',beta2='1/2'),val(theta=['307/300'],m=['-1/2'],v=[3],t=2),note='Nonzero warm state; current parameter differs from reset-state evaluation.'),
 e('AW.E4',['AW.C5'],awin(None,{'theta':[1],'m':[1],'v':[2],'t':1}),val(theta=[1],m=[1],v=[2],t=1)),
 e('AW.E5',['AW.C2','AW.C3','AW.C5'],awin([0],{'theta':[1],'m':[1],'v':[2],'t':1},beta1='1/2',beta2='1/2'),val(theta=['99/100-sqrt(3)/30'],m=['1/2'],v=[1],t=2)),
 e('AW.E6',['AW.C4'],awin([0]),outcome('domain_undefined')),
 e('AW.N1',['AW.C2'],awin([1,2]),outcome('invalid_input')),
 e('AW.N2',['AW.C3'],{**awin([2]),'amsgrad':True},outcome('unsupported'))]
aw['mutation_requirements']=[{'id':'AW.M1','misreading':'decay updated rather than old theta','witnesses':['AW.E1','AW.E3']},{'id':'AW.M2','misreading':'coupled L2 rather than AdamW','witnesses':['AW.E1']},{'id':'AW.M3','misreading':'omit bias correction or use t_old','witnesses':['AW.E2','AW.E3']},{'id':'AW.M4','misreading':'eps inside sqrt','witnesses':['AW.E2']},{'id':'AW.M5','misreading':'ignore supplied warm state','witnesses':['AW.E3']},{'id':'AW.M6','misreading':'theta correct but post-state not returned or corrupted','witnesses':['AW.E1','AW.E3']},{'id':'AW.M7','misreading':'None treated as zero gradient','witnesses':['AW.E4','AW.E5']}]
aw['nonclaims']=['No automatic coverage of coupled Adam under the ADAMW contract; map this as a profile under the existing ADAM role.','No device rounding or actual parameter writeback certification merely from this ideal-state contract.','A rejected mutant witnesses a particular error; killing all listed mutants does not prove completeness.']

ce=base('CE','CE','Cross entropy with explicit target, reduction and cotangent profiles',{
 'index_unsmoothed':{'meaning':'Class-index target, eps=0; arbitrary declared ignore_index.','task_choice_required':False},
 'index_smoothing_unweighted':{'meaning':'Class-index target, weights absent, 0<=eps<=1.','task_choice_required':False},
 'index_smoothing_R_A':{'meaning':'Per-class weighted smoothing; selected R_A interpretation; not a universally forced reading of incomplete prose.','task_choice_required':True},
 'probability_unsmoothed':{'meaning':'Valid probability rows, eps=0; no ignored probability rows.','task_choice_required':False}},
 {'loss':{'shape':'scalar for sum/mean; [N] for none','guarantee':'rigorous_enclosure'},'gradient':{'shape':'[N,C] VJP against explicit upstream','guarantee':'rigorous_enclosure'}})
ce['signature']={'inputs':['logits [N,C]','target [N] or [N,C] selected by profile','weights [C] or None','ignore_index integer for index profiles','eps','reduction sum/mean/none','upstream scalar for sum/mean; [N] for none'], 'pre_state':[],'post_state':[], 'observation_unit':'loss and complete VJP, not loss-only'}
ce['domain']={'valid':['N >= 1, C >= 2 in this pilot; finite rectangular rational logits','index targets are integers in [0,C) or equal to the declared ignore_index','weights absent or finite nonnegative [C]; eps in [0,1]','probability rows finite, nonnegative, sum to one; probability profile eps=0','upstream has the exact reduction-dependent shape'], 'unsupported':['C=1 is an API-valid case outside this pilot, not an API violation','unbatched and higher-rank flattening until mapping validated','negative class weights for this pilot','probability-target smoothing','deprecated size_average/reduce overrides','unselected R_B or R_C diagnostic conventions'], 'domain_undefined':['mean normalization mass D=0, including but not limited to all-ignored rows'], 'invalid_input':['ragged shapes or nonmatching weights','nonintegral or out-of-range nonignored index','invalid probability row','invalid upstream shape','nonfinite logits in this finite-real profile']}
ce['decisions']=[{'id':'CE.R_A','status':'needs_binding','statement':'Use per-class weights a_nc=w_c*((1-eps)onehot(y)_c+eps/C) for this selected profile. Requires explicit profile choice; R_B remains a diagnostic alternative, not automatic acceptance of every implementation.','basis':['PCE.SMOOTH','PCE.PROB']},{'id':'CE.FINITE_TARGET','status':'reviewed','statement':'When mean denominator is zero, finite-real loss and gradient are undefined. Does not impose a NaN/zero device-output convention.','basis':['PCE.REDUCE']},{'id':'CE.VJP','status':'reviewed','statement':'Gradient means VJP of the selected loss against the explicitly supplied upstream. This follows by differentiating the declared loss on finite logits.','basis':['PCE.INDEX','PCE.PROB','PCE.REDUCE']}]
ce['clauses']=[
 c('CE.C1',['PCE.INDEX'],'p_nc=exp(x_nc)/sum_j exp(x_nj); unsmoothed row ell_n=-w_y*log(p_ny); ignored row ell=0',['loss']),
 c('CE.C2',['PCE.IGNORE','PCE.REDUCE'],'Ignored index rows contribute zero loss, zero gradient, zero class-weight mass; ignore_index is an input, not hard-coded -100',['loss','gradient']),
 c('CE.C3',['PCE.REDUCE'],'Index mean D=sum_nonignored w_y; sum and none use no normalization; weights default to one',['loss','gradient']),
 c('CE.C4',['PCE.SMOOTH','CE.R_A'],'For selected R_A, a_nc=w_c*((1-eps)onehot(y)_c+eps/C); ell_n=-sum_c a_nc*log(p_nc)',['loss','gradient']),
 c('CE.C5',['CE.FINITE_TARGET'],'D=0 for mean yields domain_undefined, not an allowed_set and not a required NaN; all-ignored sum/none return zero with zero VJP',['loss','gradient']),
 c('CE.C6',['CE.VJP'],'A_n=sum_c a_nc; grad_nc=u_n*(A_n*p_nc-a_nc) for sum/none; divide additionally by D for mean. For scalar reduction the single upstream is reused on each row.',['gradient'],'derived'),
 c('CE.C7',['PCE.PROB','PCE.SHAPE'],'Probability profile: a_nc=w_c*y_nc; mean D=N (not class-weight mass); target rows match logits shape',['loss','gradient'])]
def cei(x,y,red='sum',u=1,w=None,eps=0,ii=-100):
 return {'logits':x,'target':y,'weights':w,'eps':eps,'ignore_index':ii,'reduction':red,'upstream':u}
ce['examples']=[
 e('CE.E1',['CE.C1','CE.C6'],cei([[0,0]],[0]),val(loss='log(2)',gradient=[['-1/2','1/2']]),'index_unsmoothed'),
 e('CE.E2',['CE.C1','CE.C6'],cei([[0,1]],[1]),val(loss='log(1+exp(1))-1',gradient=[['1/(1+exp(1))','-1/(1+exp(1))']]),'index_unsmoothed'),
 e('CE.E3',['CE.C2','CE.C3','CE.C6'],cei([[0,0],[0,0],[1,0]],[0,1,-100],red='mean',w=[1,3]),val(loss='log(2)',gradient=[['-1/8','1/8'],['3/8','-3/8'],[0,0]]),'index_unsmoothed'),
 e('CE.E4',['CE.C4','CE.C6'],cei([[0,1]],[0],eps='1/2'),val(loss='log(1+exp(1))-1/4',gradient=[['1/(1+exp(1))-3/4','exp(1)/(1+exp(1))-1/4']]),'index_smoothing_unweighted'),
 e('CE.E5',['CE.C4','CE.C6'],cei([[0,1]],[0],w=[1,3],eps='1/2'),val(loss='3*log(1+exp(1))/2-3/4',gradient=[['3/(2*(1+exp(1)))-3/4','3*exp(1)/(2*(1+exp(1)))-3/4']]),'index_smoothing_R_A'),
 e('CE.E6',['CE.C5'],cei([[0,0]],[-100],red='mean'),outcome('domain_undefined'),'index_unsmoothed'),
 e('CE.E7',['CE.C2','CE.C5','CE.C6'],cei([[0,1]],[-7],ii=-7),val(loss=0,gradient=[[0,0]]),'index_unsmoothed'),
 e('CE.E8',['CE.C3','CE.C6'],cei([[0,0],[0,0]],[0,1],red='none',u=[2,-1]),val(loss=['log(2)','log(2)'],gradient=[[-1,1],['-1/2','1/2']]),'index_unsmoothed'),
 e('CE.E9',['CE.C7','CE.C6'],cei([[0,0]],[['1/4','3/4']],red='mean'),val(loss='log(2)',gradient=[['1/4','-1/4']]),'probability_unsmoothed'),
 e('CE.E10',['CE.C4','CE.C5'],cei([[0,0]],[0],red='mean',w=[0,1],eps='1/2'),outcome('domain_undefined'),'index_smoothing_R_A',note='Numerator is log(2)/4, not zero; a zero denominator is not always 0/0.'),
 e('CE.N1',['CE.C6'],cei([[0,0]],[0],red='none',u=[1,1]),outcome('invalid_input'),'index_unsmoothed')]
ce['mutation_requirements']=[{'id':'CE.M1','misreading':'mean by N instead of target-class weight mass','witnesses':['CE.E3']},{'id':'CE.M2','misreading':'mean by nonignored row count','witnesses':['CE.E3']},{'id':'CE.M3','misreading':'smoothing over C-1 alternatives','witnesses':['CE.E4']},{'id':'CE.M4','misreading':'ignored rows counted in denominator','witnesses':['CE.E3']},{'id':'CE.M5','misreading':'loss correct but gradient all zero','witnesses':['CE.E1','CE.E2']},{'id':'CE.M6','misreading':'hard-coded ignore_index','witnesses':['CE.E7']},{'id':'CE.M7','misreading':'none output collapsed to scalar, dropped row, or upstream ignored','witnesses':['CE.E8']},{'id':'CE.M8','misreading':'probability-target mean by sum of target weights','witnesses':[], 'note':'Add nonunit-weight probability-target witness before claiming this mutation covered.'}]
ce['nonclaims']=['R_A versus diagnostic R_B alone is not evidence that an implementation violates explicit PyTorch wording.','These formulas do not choose a task-quality delta.','An unimplemented profile is unsupported, not an illegal API input.']

rm=base('RANKMEAN','RANKMEAN','Equal-valid-token objective over declared participants',{
 'base':{'meaning':'One declared task: L=sum per-rank token loss sums / sum valid-token counts.','task_choice_required':True}},
 {'loss':{'shape':'scalar','guarantee':'exact'}})
rm['signature']={'inputs':['S [R]: sums of per-token losses, not means','n [R]: exact nonnegative integer token counts','participant IDs [R], or an immutable binding record establishing them'], 'pre_state':[],'post_state':[],'observation_unit':'aggregation only; not an entire distributed training-step equivalence claim'}
rm['domain']={'valid':['R>=1; len(S)==len(n)==len(participant_ids); IDs unique','S finite rational; n integral and nonnegative (do not cast floats to int)','n_r=0 implies S_r=0','same valid-token set, equal token objective weights, and each contribution counted once, established by binding or declared as an unverified precondition'], 'domain_undefined':['sum(n)==0, with all S zero'], 'invalid_input':['shape mismatch','negative or fractional counts','nonzero S paired with zero n','duplicate participant IDs'],'unsupported':['sequence-equal objectives','weighted means normalized by weight mass rather than token count','changing token membership or per-token losses across compared partitions']}
rm['decisions']=[{'id':'RM.TASK','status':'needs_binding','statement':'Task owner selects equal-valid-token normalization. HF article is supporting motivation only. No default verdict against a sequence-averaged task.','basis':['HF.TOKEN_MEAN']},{'id':'RM.ZERO','status':'reviewed','statement':'No valid tokens means the finite-real average is undefined; do not demand any specific NaN/zero device convention.','basis':['RM.TASK']}]
rm['clauses']=[
 c('RM.C1',['RM.TASK'],'N=sum_r n_r; if N>0, L=(sum_r S_r)/N',['loss']),
 c('RM.C2',['RM.ZERO'],'N=0 yields domain_undefined; an individual empty rank contributes exact zero to numerator and denominator',['loss']),
 c('RM.C3',['RM.TASK'],'Participant and contribution provenance is required; permutations or repartitions preserve L only when sums, counts and membership are transformed consistently',['loss'],'derived')]
def rmi(s,n):return {'S':s,'n':n,'participant_ids':[f'r{i}' for i in range(len(s))]}
rm['examples']=[
 e('RM.E1',['RM.C1'],rmi([6,4],[3,1]),val(loss='5/2')),
 e('RM.E2',['RM.C1'],rmi([6,2],[3,1]),val(loss=2),note='Equal local means: mean-of-means coincides. Keep as normal control; do not require mutant failure.'),
 e('RM.E3',['RM.C2'],rmi([0,0],[0,0]),outcome('domain_undefined')),
 e('RM.E4',['RM.C1','RM.C2'],rmi([0,4],[0,1]),val(loss=4)),
 e('RM.E5',['RM.C1','RM.C3'],rmi([4,6],[1,3]),val(loss='5/2')),
 e('RM.N1',['RM.C1'],rmi([6,4],[3,1,1]),outcome('invalid_input')),
 e('RM.N2',['RM.C1'],rmi([6,4],['39/10',1]),outcome('invalid_input')),
 e('RM.N3',['RM.C1'],rmi([6,4],[-1,2]),outcome('invalid_input')),
 e('RM.N4',['RM.C2'],rmi([6,4],[0,1]),outcome('invalid_input'))]
rm['mutation_requirements']=[{'id':'RM.M1','misreading':'mean of rank-local means','witnesses':['RM.E1']},{'id':'RM.M2','misreading':'sum rank-local means','witnesses':['RM.E1','RM.E2']},{'id':'RM.M3','misreading':'divide by rank or sequence count','witnesses':['RM.E1']},{'id':'RM.M4','misreading':'silently cast fractional n to int','witnesses':['RM.N2']},{'id':'RM.M5','misreading':'ignore mismatched count array','witnesses':['RM.N1']}]
rm['composition_binding']=['A local CE mean is not S_r. Use its loss sum or undo the exact declared local normalization with provenance.','PyTorch class-weighted CE mean uses weight mass, not token count; choose the intended denominator explicitly.','This contract neither validates NCCL semantics nor proves BatchNorm/microbatch equivalence.']
rm['nonclaims']=['Task-selection still required per runtime use, independent of code correctness.','Empty/unsupported/invalid are distinct outcomes; no finiteness filtering or drop-rank policy may be introduced silently.']

# Complete the initially missing probability-target/weight-mass witness.
ce['examples'].append(e('CE.E11',['CE.C7','CE.C6'],cei([[0,0]],[['1/4','3/4']],red='mean',w=[2,4]),val(loss='7*log(2)/2',gradient=[['5/4','-5/4']]),'probability_unsmoothed'))
ce['mutation_requirements'][-1]['witnesses']=['CE.E11'];ce['mutation_requirements'][-1].pop('note',None)
for x in [aw,ce,rm]:
 (D/'contracts'/f"{x['id']}.yaml").write_text(yaml.safe_dump(x,allow_unicode=True,sort_keys=False,width=110))
counts={'contracts':3,'clauses':sum(len(x['clauses']) for x in [aw,ce,rm]),'profiles':sum(len(x['profiles']) for x in [aw,ce,rm]),'examples':sum(len(x['examples']) for x in [aw,ce,rm]),'mutation_requirements':sum(len(x['mutation_requirements']) for x in [aw,ce,rm]),'runtime_activated':0}
(D/'MANIFEST.json').write_text(json.dumps({'input_sha256':HASH,'counts':counts,'notes':['Reviewed implementation contracts, not an activated oracle package.','Existing numerical specs are not modified.','Private or API credentials not required.']},ensure_ascii=False,indent=2)+'\n')
print(counts)
