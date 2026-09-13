"""Interpret predeclared signed-effect observations without calling zeros negative."""
import argparse
import hashlib
import json
from pathlib import Path
from kernel_analyzer.direction_interpretation import positive_direction_frequency

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('Use a new repository output')
    data=json.loads(a.input.read_text())
    if data.get('direction_fixed_before_observation') is not True:
        raise ValueError('Direction must have been fixed before measurement')
    if data.get('claim_scope')!='DECLARED_IID_UNIT_POPULATION':
        raise ValueError('Declare sampling scope; unit IDs alone do not establish independence')
    result=positive_direction_frequency(data['signed_inner_products'],
        null_positive_probability=data['null_positive_probability'],alpha=data['alpha'])
    result['input_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    result['sampling_assumptions_not_verified_by_this_program']=True
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
