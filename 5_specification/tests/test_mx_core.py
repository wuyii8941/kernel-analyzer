import sys,unittest
from pathlib import Path
from fractions import Fraction as Q
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_spec import mx
from kernel_spec.num import Invalid,Unsupported,NotEstablished

def candidates(fmt):
    ebits,mbits,bias={'e2m1':(2,1,1),'e4m3':(4,3,7),'e5m2':(5,2,15)}[fmt]
    out=[]
    for e in range(2**ebits):
        for m in range(2**mbits):
            if fmt=='e4m3' and e==15 and m==7:continue
            if fmt=='e5m2' and e==31:continue
            value=Q(m,2**mbits)*Q(2)**(1-bias) if e==0 else (1+Q(m,2**mbits))*Q(2)**(e-bias)
            out.append((value,m%2))
    return out

class MXCore(unittest.TestCase):
    def test_smallest_scale_is_used_for_division_not_float32_zero(self):
        # The finite E8M0 code 0 denotes 2^-127, not float32 zero. A selected
        # clamped scale must also be the scale used to form the element values.
        scale=Q(2)**-127
        for profile in ('ocp_floor','ceil_pow2'):
            for x,expected in ((scale,Q(1)),(scale/2,Q(1,2)),(scale/4,Q(0)),(-scale,Q(-1))):
                result=mx.quantize([x]*32,'e2m1',profile=profile)
                self.assertEqual(result,{'scale':scale,'elements':[expected]*32,
                                         'dequantized':[expected*scale]*32})

    def test_selected_scale_rule_is_not_ignored(self):
        args=([500,1,'-3/4'],'e4m3')
        self.assertEqual(mx.quantize(*args,profile='ocp_floor'),
                         {'scale':1,'elements':[448,1,Q(-3,4)],'dequantized':[448,1,Q(-3,4)]})
        self.assertEqual(mx.quantize(*args,profile='ceil_pow2'),
                         {'scale':2,'elements':[256,Q(1,2),Q(-3,8)],'dequantized':[512,1,Q(-3,4)]})

    def test_all_element_midpoints_and_neighbours_against_encoding_enumeration(self):
        for fmt in ('e2m1','e4m3','e5m2'):
            points=candidates(fmt)
            cases=[Q(0),points[-1][0]*2]
            for (lo,_),(hi,_) in zip(points,points[1:]):
                mid=(lo+hi)/2;step=(hi-lo)/1000000
                cases.extend((lo,mid-step,mid,mid+step,hi))
            for x in cases:
                expected=min(points,key=lambda p:(abs(x-p[0]),p[1]))[0]
                self.assertEqual(mx.round_element(x,fmt),expected)
                self.assertEqual(mx.round_element(-x,fmt),-expected)

    def test_scale_boundaries_against_enumerated_powers(self):
        powers=[Q(2)**e for e in range(-127,128)]
        for fmt in ('e2m1','e4m3','e5m2'):
            maximum=candidates(fmt)[-1][0]
            leading=max(p for p in powers if p<=maximum)
            for amax in (Q(1,3),Q(500),Q(2)**-150,Q(2)**140,maximum,maximum*2):
                for profile in ('ocp_floor','ceil_pow2'):
                    target=amax/(leading if profile=='ocp_floor' else maximum)
                    choices=[p for p in powers if p<=target] if profile=='ocp_floor' else [p for p in powers if p>=target]
                    expected=(max(choices) if choices else powers[0]) if profile=='ocp_floor' else (min(choices) if choices else powers[-1])
                    result=mx.quantize([amax,-amax,amax/3],fmt,profile=profile)
                    self.assertEqual(result['scale'],expected)
                    self.assertEqual(len(result['elements']),3)
                    self.assertEqual(result['dequantized'],[expected*v for v in result['elements']])

    def test_zero_block_requires_explicit_scale_and_validates_before_return(self):
        with self.assertRaises(NotEstablished):mx.quantize([0,0],'e2m1',profile='ocp_floor')
        self.assertEqual(mx.quantize([0,0],'e2m1',profile='ceil_pow2',zero_scale=Q(2)**-127),
                         {'scale':Q(2)**-127,'elements':[0,0],'dequantized':[0,0]})
        for scale in (0,-1,3,Q(2)**-128,Q(2)**128,True):
            with self.assertRaises(Invalid):mx.quantize([0],'e2m1',profile='ocp_floor',zero_scale=scale)
        with self.assertRaises(Unsupported):mx.quantize([0],'unknown',profile='ocp_floor')
        with self.assertRaises(Unsupported):mx.quantize([0],'e2m1',profile='unknown')
        with self.assertRaises(Unsupported):mx.quantize([1],'e2m1',profile='ocp_floor',zero_scale=1)
        for block in ([],[0]*33,[[1]],[True],[float('inf')]):
            with self.assertRaises(Invalid):mx.quantize(block,'e2m1',profile='ocp_floor')

if __name__=='__main__':unittest.main()
