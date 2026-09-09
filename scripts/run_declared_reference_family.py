#!/usr/bin/env python3
"""Use a reviewed new family without editing an active frozen registry.

The entry point is pinned inside the newly generated source manifest. Capture
checks that declaration before invoking the existing, unchanged capture engine.
This is an explicit extension, never a fallback for failed existing families.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

from kernel_analyzer.source_reference_registry import REFERENCES, ReferenceSpec

ROOT = Path(__file__).resolve().parents[1]


def declaration(family, module, suffix):
    if not re.fullmatch(r'[A-Z][A-Z0-9_]*',family) or not re.fullmatch(r'[a-z][a-z0-9_]*',module):
        raise ValueError('Use explicit local family/module identifiers')
    if not re.fullmatch(r'-[a-z0-9-]+',suffix):
        raise ValueError('Invalid case suffix')
    if family in REFERENCES:
        raise ValueError('Do not replace an already registered reference')
    adapter = ROOT/'src/kernel_analyzer'/ (module+'.py')
    if not adapter.is_file():
        raise ValueError('Missing reviewed local reference module')
    return dict(family=family,module=module,case_suffix=suffix,variants=['FP32_NATIVE'],
                entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest())


def argument(arguments,name):
    if arguments.count(name)!=1 or arguments.index(name)+1==len(arguments):
        raise ValueError('Expected exactly one value for '+name)
    return Path(arguments[arguments.index(name)+1])


def validate_capture(manifest, expected):
    if manifest.get('additional_reference_declaration')!=expected:
        raise ValueError('Additional reference declaration changed or was not frozen')
    if manifest.get('reference_family')!=expected['family'] or manifest.get('adapter_sha256')!=expected['adapter_sha256']:
        raise ValueError('Source manifest refers to a different reference')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('scan','capture'),required=True)
    p.add_argument('--new-family',required=True)
    p.add_argument('--module',required=True)
    p.add_argument('--case-suffix',required=True)
    a,remaining=p.parse_known_args()
    expected=declaration(a.new_family,a.module,a.case_suffix)
    if a.mode=='capture':
        manifest=argument(remaining,'--reference-manifest')
        validate_capture(json.loads(manifest.read_text()),expected)
    else:
        output=argument(remaining,'--output')
        if output.exists():
            raise ValueError('Never annotate an existing manifest')
        if '--family' in remaining:
            raise ValueError('Family is provided by the reviewed declaration')
        remaining=['--family',a.new_family,*remaining]
    REFERENCES[a.new_family]=ReferenceSpec(a.module,('FP32_NATIVE',),a.case_suffix)
    target='scripts.build_row_reduction_contracts' if a.mode=='scan' else 'scripts.run_row_reduction_capture'
    sys.argv=[target,*remaining]
    runpy.run_module(target,run_name='__main__')
    if a.mode=='scan':
        # This manifest was created by this command, not a historical artifact.
        # An interrupted unannotated manifest is rejected by capture above.
        result=json.loads(output.read_text())
        result['additional_reference_declaration']=expected
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
