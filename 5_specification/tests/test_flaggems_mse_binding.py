"""Task correspondence must consume both inputs and reduction, and retain the full result."""
import copy
import json
from pathlib import Path
import sys
import unittest

SPEC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SPEC), str(SPEC / 'corpus')]
import call_binding as B


class FlagGemsMSEBinding(unittest.TestCase):
    def fixture(self, reduction):
        plan = json.loads((SPEC / 'corpus/call_bindings/flaggems_mse.json').read_text())
        plan['status'] = 'drafted'
        x = {'t':'f32','shape':[7,41],'v':[[1]*41 for _ in range(7)]}
        target = {'t':'f32','shape':[7,41],'v':[[0]*41 for _ in range(7)]}
        out = copy.deepcopy(x) if reduction=='none' else {'t':'f32','shape':[],'v':287 if reduction=='sum' else 1}
        observation = {'subject':plan['subject'],'launch_scope_complete':True,'launches':['original_mse_kernel'],
            'tensor_arguments':{n:{'before':v,'after':copy.deepcopy(v)} for n,v in [('x',x),('target',target)]},
            'scalars':{'reduction':reduction},'returns':{'loss':out},'public_outputs':['/returns/loss']}
        return plan,observation

    def test_all_three_tasks_have_complete_references_without_draft_admission(self):
        for reduction in ['none','sum','mean']:
            with self.subTest(reduction=reduction):
                result=B.inspect(*self.fixture(reduction))
                self.assertEqual(result['binding_status'],'development_checked',result)
                self.assertEqual(result['reference_status'],'value',result)
                self.assertFalse(result['reference_usable_for_bias_measurement'])

    def test_missing_task_input_is_rejected(self):
        for source in ['/tensor_arguments/target/before','/scalars/reduction']:
            with self.subTest(source=source):
                plan,obs=self.fixture('sum')
                plan['tasks'][0]['input_map']=[m for m in plan['tasks'][0]['input_map'] if m.get('source')!=source]
                self.assertEqual(B.inspect(plan,obs)['binding_status'],'rejected')

    def test_missing_output_and_state_are_rejected(self):
        for field in ['output_map','state_map']:
            with self.subTest(field=field):
                plan,obs=self.fixture('none')
                if field=='output_map':plan['tasks'][0][field]=[]
                else:plan[field]=plan[field][:1]
                self.assertEqual(B.inspect(plan,obs)['binding_status'],'rejected')

    def test_partial_result_and_incomplete_launch_scope_are_rejected(self):
        plan,obs=self.fixture('none')
        obs['returns']['loss']={'t':'f32','shape':[],'v':287}
        self.assertEqual(B.inspect(plan,obs)['binding_status'],'rejected')
        plan,obs=self.fixture('sum');obs['launch_scope_complete']=False
        self.assertEqual(B.inspect(plan,obs)['binding_status'],'rejected')

    def test_changed_input_is_rejected(self):
        plan,obs=self.fixture('mean')
        obs['tensor_arguments']['target']['after']['v'][0][0]=1
        self.assertEqual(B.inspect(plan,obs)['binding_status'],'rejected')


if __name__=='__main__':
    unittest.main()
