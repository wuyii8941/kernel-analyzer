"""Independent exact explanation of the captured mean-MSE scalar coefficient."""
import argparse
from fractions import Fraction as Q
import json
from pathlib import Path
import struct
import sys

SPEC=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SPEC))
from kernel_spec import registry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.capture.read_text())
    case=next(c for c in report['cases'] if c['reduction']=='mean')
    assert len(case['launches'])==1
    launch=case['launches'][0]
    assert '%scale: f32' in launch['ttir']
    received=next(p['value'] for p in launch['parameters'] if p['name']=='scale')
    physical=Q(struct.unpack('f',struct.pack('f',received))[0])
    obs=case['observation']
    x=obs['tensor_arguments']['x']['before'];target=obs['tensor_arguments']['target']['before']
    assert x['shape']==target['shape']==[7,41]
    n=287
    squared=sum((Q(a)-Q(b))**2 for row_a,row_b in zip(x['v'],target['v']) for a,b in zip(row_a,row_b))
    mathematical=squared/n
    program=squared*physical
    ref=registry.query('MSE','torch210_unweighted_f32_same_shape',{
        'op':'aten.mse_loss','args':[x,target],'kwargs':{'reduction':'mean'}})
    assert ref.outcome=='value' and ref.fields['out']==mathematical
    relative=n*physical-1
    assert relative==Q(45,2**32)
    assert program-mathematical==relative*mathematical>0
    result={'scope':'Exact derivation from the retained single-program TTIR and original wrapper, not device-output inference',
            'received_python_scale':str(Q(received)),'physical_f32_scale':str(physical),
            'exact_task_scale':str(Q(1,n)),'relative_program_minus_task':str(relative),
            'mathematical_value':str(mathematical),'program_value':str(program),
            'program_minus_task':str(program-mathematical),
            'relation':'G_pi - f = (45 / 2^32) * f for this mean task; this does not determine K - f.'}
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('exact relative program/task difference:',relative)


if __name__=='__main__':main()
