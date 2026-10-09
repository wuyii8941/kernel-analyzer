"""CPU-only design, bookkeeping and counterexample checks; not tool certification."""
from __future__ import annotations
from fractions import Fraction as Q
from math import isqrt, log, sqrt
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import random
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def load_tool(name):
    sp = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(module)
    return module

reconciler = load_tool('reconcile_official_inventory')
validator = load_tool('validate_contracts')
EMPTY = (Q(0), Q(0), Q(0))

def stat(xs):
    if not xs:
        return EMPTY
    n = Q(len(xs)); m = sum(xs, Q(0))/n
    return n, m, sum(((x-m)**2 for x in xs), Q(0))

def merge(a,b):
    na,ma,sa=a; nb,mb,sb=b
    if na == 0:
        if a != EMPTY: raise ValueError('invalid empty')
        return b
    if nb == 0:
        if b != EMPTY: raise ValueError('invalid empty')
        return a
    if na < 0 or nb < 0: raise ValueError('negative count')
    n=na+nb; delta=mb-ma
    return n, ma+delta*nb/n, sa+sb+delta*delta*na*nb/n

def phi(a):
    n,m,s=a
    return n,n*m,s+n*m*m

def fixture():
    records = [{'id':'tt.add::f32','contract_hash':'h1'}, {'id':'tt.reduce::tuple','contract_hash':'h2'}]
    invs={k: {'kind':k, 'profile_id':'toy', 'source_commit':'lock',
              'producer':'fixture-only', 'receipt':'fixture', 'complete_for_profile':True,
              'records':copy.deepcopy(records)} for k in ('source','registered','observed')}
    rules=[{'id':x['id'],'route':'toy','implementation_status':'planned'} for x in records]
    return invs,rules

class EvidenceChecks(unittest.TestCase):
    def test_125_original_ids_preserved(self):
        with zipfile.ZipFile(ROOT/'history/input_rc2.zip') as z:
            p=next(n for n in z.namelist() if n.endswith('/coverage/checklist.json'))
            old=json.loads(z.read(p))['items']
        new=json.loads((ROOT/'coverage/checklist.json').read_text())['items']
        self.assertEqual(len(new),125)
        self.assertEqual({i['id'] for i in new},{i['id'] for i in old})
        for n,o in zip(new,old):
            self.assertEqual(n['original_item'],o['item'])
    def test_checklist_traceability(self):
        items=json.loads((ROOT/'coverage/checklist.json').read_text())['items']
        md=(ROOT/'COVERAGE_CHECKLIST.md').read_text()
        for i in items:
            self.assertIn(i['anchor'],(ROOT/i['file']).read_text(),i['id'])
            self.assertIn('| '+i['id']+' |',md)
    def test_all_official_requirement_families_retained(self):
        d=json.loads((ROOT/'coverage/official_surface_matrix.json').read_text())
        self.assertEqual(len(d['entries']),61)
        self.assertEqual(len({e['id'] for e in d['entries']}),61)
        self.assertTrue(all(e['target_scope']=='included' for e in d['entries']))
    def test_each_family_has_source_reference(self):
        src={s['id'] for s in json.loads((ROOT/'evidence/official_sources.json').read_text())['sources']}
        for e in json.loads((ROOT/'coverage/official_surface_matrix.json').read_text())['entries']:
            self.assertTrue(e['source_ids'])
            self.assertTrue(set(e['source_ids'])<=src,e['id'])
    def test_public_export_snapshot_not_full_inventory(self):
        d=json.loads((ROOT/'coverage/official_public_exports_snapshot.json').read_text())
        names=d.get('exports',d.get('names'))
        if names is None:
            names=d['entries']
        self.assertEqual(len(names),148)
        self.assertEqual(len(set(names)),148)
    def test_legacy_list_is_not_new_scope(self):
        d=json.loads((ROOT/'coverage/legacy_229_route.json').read_text())
        self.assertEqual(len(d['entries']),229)
        self.assertTrue(all('in_ttir_denominator' not in e for e in d['entries']))
        for e in d['entries']:
            self.assertIn('legacy_ttir_profile_membership',e)
    def test_official_profiles_remain_in_target(self):
        s=(ROOT/'design/00_official_scope.md').read_text()
        for name in ('Gluon','AMD','NVIDIA','proton','nvg','nvws','CAS'):
            self.assertIn(name,s)

class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.schema=json.loads((ROOT/'design/rule_contract.schema.json').read_text())
        self.record=json.loads((ROOT/'examples/reduce_rule_contract.json').read_text())
    def test_examples_are_structurally_valid(self):
        for p in (ROOT/'examples').glob('*_rule_contract.json'):
            validator.validate_record(json.loads(p.read_text()),self.schema)
    def test_examples_do_not_claim_production_completion(self):
        for p in (ROOT/'examples').glob('*_rule_contract.json'):
            self.assertEqual(json.loads(p.read_text())['implementation_status'],'planned')
    def test_validated_status_requires_evidence(self):
        r=copy.deepcopy(self.record); r['implementation_status']='validated-scope'
        with self.assertRaises(ValueError): validator.validate_record(r,self.schema)
    def test_device_validation_requires_run_evidence(self):
        r=copy.deepcopy(self.record)
        r['device_validation']=[{'target_profile':'toy','status':'device-tested','evidence':[]}]
        with self.assertRaises(ValueError): validator.validate_record(r,self.schema)
    def test_precision_is_per_mode_and_condition(self):
        ps=self.record['policy_contracts']
        self.assertGreater(len(ps),1)
        self.assertEqual({x['precision'] for x in ps},{'exact','refinable','enclosure-only'})
        self.assertNotIn('precision_capability',self.record)

class InventoryChecks(unittest.TestCase):
    def test_consistent_fixture_is_only_ready_for_review(self):
        invs,r=fixture(); out=reconciler.reconcile(invs,r)
        self.assertEqual(out['status'],'READY_FOR_MANUAL_REVIEW')
        self.assertFalse(out['is_semantic_certificate'])
        self.assertEqual(out['validated_route_claims'],0)
    def test_missing_registry_fails(self):
        invs,r=fixture(); del invs['registered']
        self.assertEqual(reconciler.reconcile(invs,r)['status'],'INCOMPLETE')
    def test_empty_source_fails(self):
        invs,r=fixture(); invs['source']['records']=[]
        self.assertEqual(reconciler.reconcile(invs,r)['status'],'INCOMPLETE')
    def test_incomplete_enumeration_fails(self):
        invs,r=fixture(); invs['source']['complete_for_profile']=False
        self.assertEqual(reconciler.reconcile(invs,r)['status'],'INCOMPLETE')
    def test_profile_mismatch_fails(self):
        invs,r=fixture(); invs['registered']['source_commit']='other'
        self.assertEqual(reconciler.reconcile(invs,r)['status'],'INCOMPLETE')
    def test_missing_target_route_fails(self):
        invs,r=fixture()
        self.assertEqual(reconciler.reconcile(invs,r[:1])['status'],'INCOMPLETE')
    def test_new_observed_op_cannot_disappear(self):
        invs,r=fixture(); invs['observed']['records'].append({'id':'new','contract_hash':'h3'})
        out=reconciler.reconcile(invs,r)
        self.assertIn('new',out['differences']['observed_not_registered'])
        self.assertEqual(out['status'],'INCOMPLETE')
    def test_changed_contract_fails(self):
        invs,r=fixture(); invs['observed']['records'][0]['contract_hash']='changed'
        self.assertEqual(reconciler.reconcile(invs,r)['status'],'INCOMPLETE')
    def test_documented_disabled_source_retained(self):
        invs,r=fixture(); invs['registered']['records'].pop(); invs['observed']['records'].pop()
        exp=[{'difference':'source_only','id':'tt.reduce::tuple','reason':'not loaded in toy profile','evidence':'toy config'}]
        out=reconciler.reconcile(invs,r,exp)
        self.assertEqual(out['status'],'READY_FOR_MANUAL_REVIEW')
        self.assertEqual(out['target_records'],2)

class MathematicsChecks(unittest.TestCase):
    def test_welford_300_exact_groups(self):
        rng=random.Random(20261009)
        for _ in range(300):
            gs=[[Q(rng.randrange(-50,51),rng.randrange(1,17)) for _ in range(rng.randrange(8))] for _ in range(3)]
            a,b,c=map(stat,gs); ab=merge(a,b)
            self.assertEqual(phi(ab),tuple(x+y for x,y in zip(phi(a),phi(b))))
            self.assertEqual(merge(ab,c),merge(a,merge(b,c)))
            self.assertEqual(merge(a,b),merge(b,a))
            self.assertEqual(merge(ab,c),stat(sum(gs,[])))
    def test_unprotected_zero_weight_not_simplified(self):
        with self.assertRaises(ZeroDivisionError): Q(0)/Q(0)
        self.assertEqual(merge(EMPTY,stat([Q(3)])),stat([Q(3)]))
    def test_indefinite_branch_interval_not_automatically_refinable(self):
        for p in (16,64,128,256):
            q=1<<p; a=isqrt(2*q*q)
            self.assertLess(Q(a,q)**2-2,0)
            self.assertGreater(Q(a+1,q)**2-2,0)
    def test_cas_can_be_valid_but_schedule_dependent(self):
        def run(order):
            val=0; old={}
            for actor in order:
                old[actor]=val
                if val==0: val={'a':1,'b':2}[actor]
            return val,(old['a'],old['b'])
        self.assertEqual(run('ab'),(1,(0,1)))
        self.assertEqual(run('ba'),(2,(2,0)))
    def test_commutative_atomic_final_value_does_not_determine_old_returns(self):
        def run(order):
            val=0; old={}
            for actor in order:
                old[actor]=val; val+={'a':1,'b':2}[actor]
            return val,(old['a'],old['b'])
        self.assertEqual(run('ab'),(3,(0,1)))
        self.assertEqual(run('ba'),(3,(2,0)))
    def test_absolute_approximation_bound_is_not_relative_bound(self):
        exact=Q(0); candidate=Q(1,100); absolute=Q(1,100)
        self.assertLessEqual(abs(candidate-exact),absolute)
        self.assertNotEqual(candidate,exact*(1+absolute))
    def test_reordered_elements_change_order_sensitive_observation(self):
        a=(1,2,3); b=(3,2,1)
        self.assertEqual(sum(a),sum(b))
        self.assertNotEqual(a[0]-a[1],b[0]-b[1])
    def test_same_thread_program_order_needs_no_interthread_barrier(self):
        mem={}; mem[0]=7; result=mem[0]
        self.assertEqual(result,7) # toy sequential model, not a GPU race detector
    def test_unbounded_domain_can_have_bounded_residual(self):
        # Analytic definitions: K=0, G=x/(1+|x|); on all real x, |K-G|<1.
        for x in (Q(-10**100),Q(-1),Q(0),Q(1),Q(10**100)):
            self.assertLess(abs(x/(1+abs(x))),1)
    def test_zero_variance_does_not_collapse_hoeffding_interval(self):
        M,n,alpha=1.,64,.05
        radius=M*sqrt(2*log(2/alpha)/n)
        self.assertGreater(radius,0)
        self.assertLess(radius,1)
    def test_clipping_preserves_true_value_but_empty_is_conflict(self):
        M=Q(1); a=Q(9,10); lo,hi=Q(8,10),Q(11,10)
        self.assertTrue(max(lo,-M)<=a<=min(hi,M))
        self.assertGreater(max(Q(2),-M),min(Q(3),M))
    def test_finite_projections_can_miss_vector_mean(self):
        mean=(Q(1),Q(-1)); self.assertEqual(sum(mean),0)
        self.assertNotEqual(mean,(0,0))
    def test_rare_tail_zero_sample_is_not_zero_mean(self):
        self.assertEqual(Q(1,1000)*1000,1)
        self.assertGreater(Q(999,1000)**64,Q(93,100))
    def test_set_reference_can_certify_sign_without_unique_value(self):
        K=Q(3); possible_g=(Q(1),Q(2))
        self.assertGreater(min(K-g for g in possible_g),0)

if __name__=='__main__':
    unittest.main(verbosity=2)
