import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L,graph as G
from kernel_spec.aten_adapter import make_reference
from kernel_spec.num import Unsupported

class UnimplementedOptions(unittest.TestCase):
    def test_direct_calls_do_not_ignore_unknown_options(self):
        x=A.tensor([1,2],'i64')
        calls=[lambda:A.full([2],1,'i64',out=x),lambda:L.l_full([2],1,'i64',out=x),
               lambda:A.zeros([2],layout='sparse_coo'),lambda:A.ones([2],unexpected=True),
               lambda:A.full_like(x,1,out=x),lambda:A.zeros_like(x,out=x),
               lambda:A.new_zeros(x,[2],out=x),lambda:A.scalar_tensor(1,out=x),
               lambda:A.arange(2,out=x),lambda:A.iota(2,unexpected=True),
               lambda:L.l_iota(2,unexpected=True),lambda:A.linspace(0,1,2,out=x),
               lambda:A.clone(x,memory_format='unreviewed'),lambda:A._to_copy(x,layout='sparse_coo')]
        for i,call in enumerate(calls):
            with self.subTest(call=i),self.assertRaises(Unsupported):call()

    def test_graph_and_lazy_cannot_bypass_keyword_checks(self):
        x={'t':'i64','v':[1,2]}
        for op,args,kwargs in [('aten.full',[[2],1],{'dtype':'i64','out':{'ref':'x'}}),
                               ('aten.clone',[{'ref':'x'}],{'unexpected':True}),
                               ('aten._to_copy',[{'ref':'x'}],{'layout':'sparse_coo'}),
                               ('aten.zeros',[[2]],{'out':{'ref':'x'}})]:
            task={'inputs':{'x':x},'nodes':[{'name':'y','op':op,'args':args,'kwargs':kwargs}],'outputs':['y']}
            for fn in (lambda:G.evaluate(task),lambda:L.evaluate_lazy(task,{'x':A.tensor([1,2],'i64')})):
                with self.subTest(op=op),self.assertRaises(Unsupported):fn()

    def test_adapter_does_not_treat_var_kwargs_as_support(self):
        ref=make_reference({'aten.full'})
        with self.assertRaises(Unsupported):ref('real',{'op':'aten.full','args':[[2],1],'kwargs':{'dtype':'i64','layout':'sparse_coo'}})
        self.assertEqual(ref('real',{'op':'aten.full','args':[[2],1],'kwargs':{'dtype':'i64'}}).fields['out'],[1,1])

if __name__=='__main__':unittest.main()
