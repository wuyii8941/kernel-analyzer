"""Independent audit of saved MX device bytes, using only Python's standard library.

Exit zero means the recorded evidence agrees with the independent calculation,
NOT that the device matches the selected mathematical task. Reported differences
are retained. This post-experiment audit is not a blind acceptance experiment.
"""
import argparse
import json
from fractions import Fraction as Q
from pathlib import Path


def require(condition,message):
    if not condition:raise ValueError(message)


def fp4(code):
    sign=-1 if code&8 else 1
    exponent=(code>>1)&3;mantissa=code&1
    return sign*(Q(mantissa,2) if exponent==0 else (1+Q(mantissa,2))*Q(2)**(exponent-1))


def verify(record):
    binding=record['binding']
    for name,value in {'input_shape':[1,32],'input_dtype':'float32','axis':1,
                       'element_format':'e2m1','scale_format':'finite E8M0','rounding_mode_argument':1,
                       'microblock_size':32,'output_packed_shape':[1,16],'output_scale_shape':[1,1],
                       'packing':'low nibble even logical coordinate, high nibble odd'}.items():
        require(binding.get(name)==value,f'unsupported binding {name}')
    # qmax=6 has leading power of two 4. For x=2**e the declared
    # profile selects scale=2**max(-127,e-2). Ratios below are exactly
    # representable E2M1 values; no rounding implementation is called.
    expected={Q(0):(Q(2)**-127,Q(0))}
    for exponent in (0,-124,-125,-126,-127):
        scale=Q(2)**max(-127,exponent-2)
        expected[Q(2)**exponent]=(scale,Q(2)**exponent/scale)
    cases=record['cases'];require(len(cases)==len(expected),'missing or extra cases')
    seen=set();rows=[]
    for case in cases:
        value=Q(case['input']);require(value in expected and value not in seen,'unexpected or repeated input')
        seen.add(value);scale,element=expected[value]
        exp=case['expected']
        require(set(exp)=={'scale','elements','dequantized'},'incomplete expected fields')
        require(Q(exp['scale'])==scale,'stored expected scale disagrees with hand calculation')
        require(list(map(Q,exp['elements']))==[element]*32,'stored expected elements disagree with hand calculation')
        require(list(map(Q,exp['dequantized']))==[scale*element]*32,'stored expected dequantization disagrees with hand calculation')
        runs=case['runs'];require(len(runs)==2,'both device replays required')
        decoded_runs=[]
        for run in runs:
            raw=run['packed_bytes'];code=run['scale_code']
            require(isinstance(raw,list) and len(raw)==16,'complete packed output required')
            require(all(type(v) is int and 0<=v<=255 for v in raw),'invalid packed byte')
            require(type(code) is int and 0<=code<=254,'invalid finite E8M0 scale code')
            actual_scale=Q(2)**(code-127)
            elements=[fp4(nibble) for byte in raw for nibble in (byte&15,byte>>4)]
            require(Q(run['scale'])==actual_scale,'reported scale differs from raw code')
            require(list(map(Q,run['elements']))==elements,'reported elements differ from raw bytes')
            equal=elements==[element]*32
            require(type(run['complete_elements_match']) is bool and run['complete_elements_match']==equal,'incorrect element-match flag')
            require(type(run['scale_matches']) is bool and run['scale_matches']==(actual_scale==scale),'incorrect scale-match flag')
            decoded_runs.append((actual_scale,elements))
        identical=decoded_runs[0]==decoded_runs[1]
        require(type(case['repeat_identical']) is bool and case['repeat_identical']==identical,'incorrect repeat flag')
        actual_scale,elements=decoded_runs[0]
        rows.append({'input':str(value),'expected_scale':str(scale),'expected_element':str(element),
                     'same_on_both_replays':identical,'scale_matches':actual_scale==scale,
                     'different_element_coordinates':[i for i,x in enumerate(elements) if x!=element],
                     'different_dequantized_coordinates':[i for i,x in enumerate(elements) if actual_scale*x!=scale*element]})
    return {'scope':'Independent post-experiment byte decoding and hand calculation, not kernel certification',
            'cases':rows,'cases_with_element_difference':sum(bool(r['different_element_coordinates']) for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,default=Path(__file__).with_name('observed.json'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),'output must be a new file')
    result=verify(json.loads(args.record.read_text()))
    with args.out.open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(f"Independent audit completed; {result['cases_with_element_difference']} cases differ from the selected task.")


if __name__=='__main__':main()
