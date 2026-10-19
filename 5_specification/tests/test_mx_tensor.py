import sys,unittest
from pathlib import Path
from fractions import Fraction as Q
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import mx
from kernel_spec.num import Invalid,Unsupported,NotEstablished
from test_mx_core import candidates

class MXTensor(unittest.TestCase):
    def test_axis_changes_block_membership_and_result(self):
        x=[[500,Q(1,1024)],[1,1]]
        row=mx.quantize_tensor(x,'e4m3',axis=1,profile='ocp_floor',partial_blocks=True)
        col=mx.quantize_tensor(x,'e4m3',axis=0,profile='ocp_floor',partial_blocks=True)
        self.assertEqual(row['scales'],[[1],[Q(1,256)]])
        self.assertEqual(col['scales'],[[1,Q(1,256)]])
        self.assertEqual(row['dequantized'],[[448,0],[1,1]])
        self.assertEqual(col['dequantized'],[[448,Q(1,1024)],[1,1]])

    def test_partial_tail_is_explicit_and_keeps_all_elements(self):
        x=[6]*32+[12]
        with self.assertRaises(Unsupported):mx.quantize_tensor(x,'e2m1',axis=0,profile='ocp_floor')
        result=mx.quantize_tensor(x,'e2m1',axis=-1,profile='ocp_floor',partial_blocks=True)
        self.assertEqual(result,{'scales':[1,2],'elements':[6]*33,'dequantized':x})

    def test_all_axes_and_coordinates_against_independent_index_enumeration(self):
        x=np.empty((2,3,35),dtype=object)
        for i in np.ndindex(x.shape):x[i]=Q((i[0]*31+i[1]*13+i[2]*7)%61+1,8)
        points=candidates('e2m1');powers=[Q(2)**e for e in range(-127,128)]
        for axis in range(3):
            for profile in ('ocp_floor','ceil_pow2'):
                result=mx.quantize_tensor(x.tolist(),'e2m1',axis=axis,profile=profile,partial_blocks=True)
                shapes=list(x.shape);shapes[axis]=(shapes[axis]+31)//32
                scales=np.empty(shapes,dtype=object);elements=np.empty(x.shape,dtype=object);deq=np.empty(x.shape,dtype=object)
                for block_index in np.ndindex(tuple(shapes)):
                    coords=[i for i in np.ndindex(x.shape) if all((i[d]//32 if d==axis else i[d])==block_index[d] for d in range(3))]
                    maximum=max(abs(x[i]) for i in coords)
                    target=maximum/(4 if profile=='ocp_floor' else 6)
                    scale=max(p for p in powers if p<=target) if profile=='ocp_floor' else min(p for p in powers if p>=target)
                    scales[block_index]=scale
                    for i in coords:
                        elements[i]=min(points,key=lambda p:(abs(x[i]/scale-p[0]),p[1]))[0]
                        deq[i]=elements[i]*scale
                self.assertEqual(result,{'scales':scales.tolist(),'elements':elements.tolist(),'dequantized':deq.tolist()})

    def test_zero_blocks_and_metadata_validation_precede_computation(self):
        x=[0]*32+[6]*32
        with self.assertRaises(NotEstablished):mx.quantize_tensor(x,'e2m1',axis=0,profile='ocp_floor')
        self.assertEqual(mx.quantize_tensor(x,'e2m1',axis=0,profile='ocp_floor',zero_scale=2)['scales'],[2,1])
        self.assertEqual(mx.quantize_tensor([],'e2m1',axis=0,profile='ocp_floor'),{'scales':[],'elements':[],'dequantized':[]})
        with patch.object(mx,'quantize',side_effect=AssertionError('computed before input validation')):
            for axis in (1,-2,True,0.5):
                with self.assertRaises(Invalid):mx.quantize_tensor(x,'e2m1',axis=axis,profile='ocp_floor')
            for kw in ({'partial_blocks':1},{'zero_scale':3}):
                with self.assertRaises(Invalid):mx.quantize_tensor(x,'e2m1',axis=0,profile='ocp_floor',**kw)
            for bad in ([1]*32+[float('inf')],[[1],[2,3]]):
                with self.assertRaises(Invalid):mx.quantize_tensor(bad,'e2m1',axis=0,profile='ocp_floor',partial_blocks=True)

if __name__=='__main__':unittest.main()
