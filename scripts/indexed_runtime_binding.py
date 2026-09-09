"""Resolve declared indexed calls in loaded generated modules, without outcomes.

Checks the selected call, not whole-model equivalence. Full source digests are
recorded so compilation changes remain visible instead of being called identity.
"""
import ast
import hashlib
from pathlib import Path
from scripts.build_indexed_call_contracts import check_call


def bind_live_call(contract,modules):
    saved=Path(contract['source_path'])
    if hashlib.sha256(saved.read_bytes()).hexdigest()!=contract['source_sha256']:
        raise ValueError('Declared source changed')
    expected=contract['call_ast_sha256']
    matches=[]; inspected=set()
    for module in modules:
        filename=getattr(module,'__file__',None)
        if filename is None: continue
        path=Path(filename).resolve()
        if path in inspected: continue
        inspected.add(path)
        source=path.read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node,ast.Call) or ast.unparse(node.func)!='aten.index_put_': continue
            try: current=check_call(ast.unparse(node))
            except ValueError: continue
            if current['call_ast_sha256']!=expected: continue
            # The observer identifies one physical generated call line.
            if node.lineno!=node.end_lineno:
                raise ValueError('Multiline call needs explicit observer support')
            line=source.splitlines()[node.lineno-1].strip()
            if line!=ast.get_source_segment(source,node):
                raise ValueError('Call must be a standalone generated expression')
            matches.append(dict(executing_filename=str(path),executing_line=node.lineno,
                source_line_sha256=hashlib.sha256(line.encode()).hexdigest(),
                executing_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                declared_source_sha256=contract['source_sha256'],call_ast_sha256=expected,
                claim_scope='SELECTED_CALL_ONLY_NOT_WHOLE_MODULE_IDENTITY'))
    if len(matches)!=1:
        raise ValueError('Expected exactly one live declared indexed call, found '+str(len(matches)))
    return matches[0]
