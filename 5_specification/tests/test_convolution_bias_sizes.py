import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A
from kernel_spec.num import Invalid,NotEstablished

class ConvolutionBiasSizes(unittest.TestCase):
    def call(self,sizes,bias=True):
        x=A.tensor([[[1,2,3]]],'f64');w=A.tensor([[[1]],[[2]]],'f64');g=A.tensor([[[1,2,3]],[[4,5,6]]],'f64')
        g=A.view(g,[1,2,3])
        return A.convolution_backward(g,x,w,sizes,[1],[0],[1],False,[0],1,[True,True,bias])

    def test_declared_bias_metadata_cannot_silently_disagree(self):
        for sizes in ([],[1],[3],[1,2],[-1]):
            for bias in (False,True):
                with self.subTest(sizes=sizes,bias=bias),self.assertRaises(NotEstablished):self.call(sizes,bias)

    def test_bias_sizes_follow_the_integer_sequence_schema(self):
        for sizes in ([2.5],[2.0],[True],2,'2'):
            with self.subTest(sizes=sizes),self.assertRaises(Invalid):self.call(sizes)

    def test_omitted_or_consistent_metadata_preserve_all_gradients(self):
        for sizes in (None,[2],(2,)):
            out=self.call(sizes)
            self.assertEqual(len(out),3)
            self.assertEqual([A.plain(t) for t in out],[[[[9,12,15]]],[[[14]],[[32]]],[6,15]])
            masked=self.call(sizes,False)
            self.assertIsNone(masked[2])
            self.assertEqual([A.plain(t) for t in masked[:2]],[A.plain(t) for t in out[:2]])

if __name__=='__main__':unittest.main()
