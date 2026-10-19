import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import aten as A,lazy as L
from kernel_spec.num import Invalid

class PaddingDomains(unittest.TestCase):
    def test_all_padding_entries_and_crop_extents_are_validated(self):
        for shape,pad in [((2,),[1]),((2,),[0,0,1]),((2,),[1,1,1,1]),
                          ((2,),[1.5,0]),((2,),[True,0]),((2,),[-3,5]),
                          ((2,),[5,-3]),((2,),[-1,-2]),((0,),[-1,1]),
                          ((),[0,0]),((0,2),[1])]:
            for fn in (A.constant_pad_nd,L.l_pad):
                with self.subTest(fn=fn.__name__,shape=shape,pad=pad),self.assertRaises(Invalid):
                    fn(A.tensor(np.zeros(shape,dtype=np.int64),'i64'),pad,9)

    def test_coordinate_mapping_includes_crop_and_padding(self):
        x=A.tensor([[1,2,3],[4,5,6]],'i64')
        expected=[[9,9,9,9],[2,3,9,9],[5,6,9,9]]
        for fn in (A.constant_pad_nd,L.l_pad):
            out=fn(x,[-1,2,1,0],9)
            self.assertEqual(out.shape,(3,4))
            for i in np.ndindex(out.shape):self.assertEqual(out.get(i) if isinstance(out,L.Lazy) else out.a[i],expected[i[0]][i[1]])
        self.assertEqual(A.plain(x),[[1,2,3],[4,5,6]])

    def test_empty_result_and_empty_input_padding(self):
        for fn in (A.constant_pad_nd,L.l_pad):
            self.assertEqual(fn(A.tensor([1,2],'i64'),[-1,-1]).shape,(0,))
            out=fn(A.tensor([],'i64'),[1,2],7)
            self.assertEqual(out.shape,(3,))
            self.assertEqual([out.get((i,)) if isinstance(out,L.Lazy) else out.a[i] for i in range(3)],[7,7,7])
            self.assertEqual(fn(A.tensor(3,'i64'),[]).shape,())

if __name__=='__main__':unittest.main()
