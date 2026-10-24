import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,graph as G,lazy as L
from kernel_spec.num import Invalid,Unsupported

class GraphResultSelection(unittest.TestCase):
    def task(self,index,reference=False):
        nodes=[{'name':'pair','op':'aten.max','args':[{'ref':'x'},0]}]
        outputs=[{'ref':'pair','index':index}] if reference else ['y']
        if not reference:nodes.append({'name':'y','op':'operator.getitem','args':[{'ref':'pair'},index]})
        return {'inputs':{'x':{'t':'f64','v':[1,3]}},'nodes':nodes,'outputs':outputs}

    def test_invalid_indices_are_input_errors(self):
        for index in (2,-3,0.5,'0'):
            for reference in (False,True):
                task=self.task(index,reference)
                with self.subTest(index=index,reference=reference),self.assertRaises(Invalid):G.evaluate(task)
                with self.assertRaises(Invalid):
                    L.evaluate_lazy(task,{'x':L.as_lazy(A.tensor([1,3],'f64'))})

    def test_tensor_indexing_is_not_multi_result_projection(self):
        task={'inputs':{'x':{'t':'i64','v':[1,2]}},'nodes':[{'name':'y','op':'operator.getitem','args':[{'ref':'x'},0]}],'outputs':['y']}
        with self.assertRaises(Unsupported):G.evaluate(task)
        with self.assertRaises(Unsupported):
            L.evaluate_lazy(task,{'x':L.as_lazy(A.tensor([1,2],'i64'))})

    def test_positive_and_negative_indices_select_same_complete_result(self):
        for index in (1,-1):
            self.assertEqual(G.evaluate(self.task(index)),[1])
            self.assertEqual(G.evaluate(self.task(index,True)),[1])
            env=L.evaluate_lazy(self.task(index),{'x':L.as_lazy(A.tensor([1,3],'f64'))})
            self.assertEqual(env['y'].get(()),1)

    def test_absent_gradient_remains_none_for_both_selection_forms(self):
        t={'t':'f32','v':[[[1]]]}
        node={'name':'pair','op':'aten.convolution_backward','args':[t,t,t,[1],[1],[0],[1],False,[0],1,[False,True,True]]}
        task={'inputs':{},'nodes':[node],'outputs':[{'ref':'pair','index':0}]}
        self.assertEqual(G.evaluate(task),[None])
        task['nodes'].append({'name':'y','op':'operator.getitem','args':[{'ref':'pair'},0]});task['outputs']=['y']
        self.assertEqual(G.evaluate(task),[None])
        lazy_task={'nodes':[{'name':'y','op':'operator.getitem','args':[{'ref':'pair'},0]}],'outputs':['y']}
        self.assertIsNone(L.evaluate_lazy(lazy_task,{'pair':[None,A.tensor(1,'i64')]})['y'])

if __name__=='__main__':unittest.main()
