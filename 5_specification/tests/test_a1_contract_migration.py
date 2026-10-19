import sys,unittest
from copy import deepcopy
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import check_contracts
from kernel_spec import registry
from kernel_spec.num import evaluate, NotEstablished

class A1ContractMigration(unittest.TestCase):
    def setUp(self):self.contract=registry.load('A1_LINEAR')

    def test_complete_forward_output_membership(self):
        adapter=self.contract.adapter
        args=dict(operation='forward',q=[[1]],k=[[1]],v=[[3,4]],S0=[[2,5]])
        expected={'result':{'out':[[5,9]],'state':[[5,9]]}}
        self.assertTrue(adapter.check_output('unnormalized_core',args,expected))
        for field in ('out','state'):
            for change in ('missing','short','wrong'):
                bad=deepcopy(expected)
                if change=='missing':del bad['result'][field]
                elif change=='short':bad['result'][field][0].pop()
                else:bad['result'][field][0][1]+=1
                with self.subTest(field=field,change=change):
                    self.assertFalse(adapter.check_output('unnormalized_core',args,bad))
        self.assertFalse(adapter.check_output('unnormalized_core',args,dict(expected,extra=0)))

    def test_normalized_backward_requires_every_gradient(self):
        args=dict(operation='backward',q=[[1]],k=[[1]],v=[[3]],S0=[[2]],Z0=[4],epsilon=0,
                  dO=[[1]],dST=[[2]],dZT=[3])
        expected={'result':{'dq':[[0]],'dk':[[Q(47,5)]],'dv':[[Q(11,5)]],
                            'dS0':[[Q(11,5)]],'dZ0':[Q(14,5)]}}
        self.assertTrue(self.contract.adapter.check_output('normalized_core',args,expected))
        for field in expected['result']:
            bad=deepcopy(expected)
            del bad['result'][field]
            self.assertFalse(self.contract.adapter.check_output('normalized_core',args,bad))

    def test_nonpoint_output_is_not_certified_by_containment(self):
        args=dict(operation='forward',q=[[1,0]],k=[[1,0]],v=[[1]])
        reference=self.contract.adapter.reference('unnormalized_core',args)
        with self.assertRaises(NotEstablished):
            self.contract.adapter.check_output('unnormalized_core',args,reference.fields)

    def test_complete_state_and_vjp_reach_current_adapter(self):
        ref=self.contract.adapter.reference
        args=dict(operation='forward',q=[[1]],k=[[1]],v=[[3,4]],S0=[[2,5]])
        result=evaluate(ref,'unnormalized_core',args)
        self.assertEqual(result.outcome,'value')
        self.assertEqual(result.fields,{'result':{'out':[[5,9]],'state':[[5,9]]}})
        args=dict(operation='backward',q=[[1]],k=[[1]],v=[[3]],S0=[[2]],Z0=[4],epsilon=0,
                  dO=[[1]],dST=[[2]],dZT=[3])
        self.assertEqual(evaluate(ref,'normalized_core',args).fields['result'],
                         {'dq':[[0]],'dk':[[Q(47,5)]],'dv':[[Q(11,5)]],'dS0':[[Q(11,5)]],'dZ0':[Q(14,5)]})

    def test_wrong_or_unused_arguments_are_not_silently_discarded(self):
        base=dict(operation='forward',q=[[1]],k=[[1]],v=[[3]])
        for extra in ({'S0_list':[[[1]]]},{'dST':[[1]]},{'epsilon':0},{'typo':1}):
            self.assertEqual(evaluate(self.contract.adapter.reference,'unnormalized_core',base|extra).outcome,'unsupported')
        self.assertEqual(evaluate(self.contract.adapter.reference,'normalized_core',base).outcome,'invalid_input')
        self.assertEqual(evaluate(self.contract.adapter.reference,'unnormalized_core',base|{'operation':'backward'}).outcome,'invalid_input')

    def test_draft_migration_does_not_activate_contract(self):
        usable,reasons,labels=registry.activation(self.contract,'unnormalized_core',allow_draft=True,
                                                 task_bindings={'unnormalized_core':'Selected explicit core for development'})
        self.assertFalse(usable)
        self.assertEqual(labels['implementation_status'],'pending')
        self.assertEqual(labels['semantic_verdict'],'not_established')
        self.assertTrue(reasons)
        self.assertEqual(self.contract.path,ROOT/'contracts/A1_LINEAR.yaml')

    def test_retained_source_without_digest_is_not_mislabelled_standard(self):
        sources={'LOCAL':{'snapshot':'sources/A1_LOCAL_LINEAR.py','kind':'supporting_implementation'},
                 'STD':{'kind':'standard','edition':'1.0'},'MISSING':{'kind':'supporting_implementation'}}
        data={'clauses':[{'basis':['LOCAL.R','STD.R','MISSING.R']}]}
        self.assertEqual(check_contracts.source_status(data,sources),
                         {'LOCAL':'retained local source','MISSING':'no snapshot','STD':'standard, 1.0'})

if __name__=='__main__':unittest.main()
