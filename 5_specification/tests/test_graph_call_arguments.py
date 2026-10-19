import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,graph as G,lazy as L
from kernel_spec.num import Invalid,Unsupported

class GraphCallArguments(unittest.TestCase):
    def fixture(self,op,args,kwargs=None,outputs=None):
        return {'inputs':{'x':{'t':'i64','v':[1,2]}},'nodes':[{'name':'y','op':op,'args':args,'kwargs':kwargs or {}}],'outputs':outputs or ['y']}

    def test_extra_positional_values_cannot_be_ignored(self):
        x={'ref':'x'}
        for op,args in [('aten.embedding',[x,x,-1,False]),('aten.clone',[x,True])]:
            task=self.fixture(op,args)
            for fn in (lambda:G.evaluate(task),lambda:L.evaluate_lazy(task,{'x':A.tensor([1,2],'i64')})):
                with self.subTest(op=op),self.assertRaises(Invalid):fn()

    def test_missing_and_duplicate_arguments_are_checked_even_in_unused_nodes(self):
        x={'ref':'x'}
        for op,args,kwargs in [('aten.add',[x],{}),('aten.full',[[2]],{}),('aten.mul',[x,2],{'b':3})]:
            task=self.fixture(op,args,kwargs,outputs=['x'])
            for fn in (lambda:G.evaluate(task),lambda:L.evaluate_lazy(task,{'x':A.tensor([1,2],'i64')})):
                with self.subTest(op=op),self.assertRaises(Invalid):fn()

    def test_declared_optional_arguments_still_work(self):
        task=self.fixture('aten.add',[{'ref':'x'},2],{'alpha':3})
        self.assertEqual(G.evaluate(task),[[7,8]])
        env=L.evaluate_lazy(task,{'x':A.tensor([1,2],'i64')})
        self.assertEqual([env['y'].get((i,)) for i in range(2)],[7,8])

    def test_canonical_keyword_names_reach_the_lazy_implementation(self):
        task=self.fixture('aten.full',[],{'size':[1],'value':7,'dtype':'i64'})
        self.assertEqual(G.evaluate(task),[[7]])
        env=L.evaluate_lazy(task,{'x':A.tensor([1,2],'i64')})
        self.assertEqual(env['y'].get((0,)),7)

    def test_copy_is_outside_graph_scope_but_helper_checks_non_blocking(self):
        self.assertNotIn('aten.copy',G.covered_ops())
        for flag in (False,True):
            task=self.fixture('aten.copy',[{'ref':'x'},{'ref':'x'}],{'non_blocking':flag})
            with self.assertRaises(Unsupported):
                G.evaluate(task)
            env=L.evaluate_lazy(task,{'x':A.tensor([1,2],'i64')})
            self.assertEqual([env['y'].get((i,)) for i in range(2)],[1,2])
            self.assertEqual(A.plain(A.copy(A.tensor([1,2],'i64'),A.tensor([3,4],'i64'),flag)),[3,4])
            lazy_copy=L.l_copy(A.tensor([1,2],'i64'),A.tensor([3,4],'i64'),flag)
            self.assertEqual([lazy_copy.get((i,)) for i in range(2)],[3,4])
        for fn in (A.copy,L.l_copy):
            with self.assertRaises(Invalid):fn(A.tensor([1],'i64'),A.tensor([2],'i64'),non_blocking='yes')

    def test_keyword_casts_are_bound_before_selecting_a_reading(self):
        task={'inputs':{'x':{'t':'f64','v':[1,2]}},'nodes':[
            {'name':'c','op':'prims.convert_element_type','args':[], 'kwargs':{'a':{'ref':'x'},'dtype':'f32'}},
            {'name':'y','op':'aten.mul','args':[{'ref':'c'},2]}], 'outputs':['y']}
        self.assertEqual(G.evaluate(task,elide_intermediate_casts=True),[[2,4]])
        env=L.evaluate_lazy(task,{'x':L.as_lazy(A.tensor([1,2],'f64'))},elide_intermediate_casts=True)
        self.assertEqual([env['y'].get((i,)).v for i in range(2)],[2,4])

if __name__=='__main__':unittest.main()
