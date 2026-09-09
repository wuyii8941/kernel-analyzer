#!/usr/bin/env python3
"""Discover continued segments without a manually supplied symbol or origin."""
import argparse
import ast
from pathlib import Path
from scripts import scan_recurrence_sources
from scripts.run_numerical_coverage import save, sha
from kernel_analyzer import continued_recurrence_source


def checked_candidate(source, symbol):
    module = ast.parse(source)
    assignment = module.body[0]
    call = assignment.value
    if (not isinstance(call, ast.Call) or len(call.args) < 2
            or not isinstance(call.args[1], ast.Constant)
            or not isinstance(call.args[1].value, str)):
        raise ValueError('Literal source unavailable')
    functions = [n for n in ast.parse(call.args[1].value).body
                 if isinstance(n, ast.FunctionDef) and n.name == symbol]
    if len(functions) != 1:
        raise ValueError('Unique function unavailable')
    offsets = []
    for statement in functions[0].body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call) or ast.unparse(node.func) != 'tl.load' or not node.args:
                continue
            address = node.args[0]
            if (isinstance(address, ast.BinOp) and isinstance(address.op, ast.Add)
                    and isinstance(address.left, ast.Name) and address.left.id == 'in_ptr2'):
                offset = address.right
                if (not isinstance(offset, ast.BinOp) or not isinstance(offset.op, ast.Add)
                        or not isinstance(offset.left, ast.Constant)
                        or type(offset.left.value) is not int
                        or not isinstance(offset.right, ast.Name) or offset.right.id != 'x1'):
                    raise ValueError('Unsupported time address')
                offsets.append(offset.left.value)
    if not offsets or offsets[0] % 1536:
        raise ValueError('No supported time origin')
    # This only proposes a parameter. Acceptance still checks the entire body,
    # every offset/write, arguments and storage types with the production checker.
    return continued_recurrence_source.check_source(source, symbol, time_start=offsets[0]//1536)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    sources = []
    for path in a.source:
        rows = scan_recurrence_sources.scan(path.read_text(), checker=checked_candidate)
        sources.append(dict(path=str(path.resolve()), sha256=sha(path), rows=rows))
        print(dict(source=str(path), definitions=len(rows),
                   checked=sum(r['status']=='SOURCE_CHECKED' for r in rows)), flush=True)
    dependencies = [Path(__file__), Path(scan_recurrence_sources.__file__),
                    Path(continued_recurrence_source.__file__)]
    save(a.output, dict(schema='continued-recurrence-source-scan-v1', sources=sources,
        numerical_results_read=False, runtime_measurement_complete=False,
        source_sha256={str(p.resolve()):sha(p) for p in dependencies}))


if __name__ == '__main__':
    main()
