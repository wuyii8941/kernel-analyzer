#!/usr/bin/env python3
"""Declare a logical CUDA index change, retaining every other AST field.

This is a new expected source, NOT evidence of actual execution. The unchanged
capture guard must still verify the full function AST on the selected device.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def declare(text, device):
    if type(device) is not int or device < 0:
        raise ValueError('Nonnegative logical device index required')
    tree=ast.parse(text)
    changed=[]
    for node in ast.walk(tree):
        if not (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call)
                and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='triton'):
            continue
        call=node.value
        if len(call.args)<2 or not isinstance(call.args[1],ast.Constant) or not isinstance(call.args[1].value,str):
            raise ValueError('Literal Triton source required')
        kernel=ast.parse(call.args[1].value)
        devices=[n for n in ast.walk(kernel) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Name) and n.func.id=='DeviceProperties']
        for properties in devices:
            indices=[kw for kw in properties.keywords if kw.arg=='index']
            if len(indices)!=1 or not isinstance(indices[0].value,ast.Constant) or type(indices[0].value.value) is not int:
                raise ValueError('Ambiguous device index')
            changed.append(dict(symbols=[ast.unparse(t) for t in node.targets],old_index=indices[0].value.value,new_index=device))
            indices[0].value=ast.Constant(device)
        call.args[1]=ast.Constant(ast.unparse(kernel))
    if not changed:
        raise ValueError('No explicit device declarations')
    return ast.unparse(tree)+'\n', changed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--device-index',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use new directory under /data1/tzh')
    original=a.source.read_bytes()
    text,changes=declare(original.decode(),a.device_index)
    a.output.mkdir(parents=True)
    (a.output/'output_code.py').write_text(text)
    (a.output/'declaration.json').write_text(json.dumps(dict(
        original_source=str(a.source.resolve()),original_sha256=hashlib.sha256(original).hexdigest(),
        declared_source_sha256=hashlib.sha256(text.encode()).hexdigest(),changes=changes,
        status='EXPECTED_DEVICE_DECLARATION_NOT_RUNTIME_EVIDENCE',
        runtime_guard_relaxed=False,numerical_results_read=False),indent=2)+'\n')
    print(json.dumps(dict(declarations=len(changes),output=str(a.output))))


if __name__=='__main__':main()
