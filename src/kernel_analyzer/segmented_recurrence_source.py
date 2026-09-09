"""First-segment body template with an explicit descending time origin.

This validates arithmetic only, not storage dtypes or runtime identity. The
frozen single-segment checker remains unchanged for existing experiments.
"""
import ast
import hashlib
from kernel_analyzer.decayed_recurrence_source import expected_body


def first_segment_body(*,channels,width,outputs,time_start):
    if not isinstance(time_start,int) or isinstance(time_start,bool) or time_start<outputs:
        raise ValueError('Descending segment must not cross time row zero')
    body=ast.parse(expected_body(channels,width,outputs)).body
    changed=0
    for statement in body:
        for call in ast.walk(statement):
            if not isinstance(call,ast.Call) or ast.unparse(call.func)!='tl.load': continue
            pointer=call.args[0]
            if (not isinstance(pointer,ast.BinOp) or not isinstance(pointer.left,ast.Name)
                    or pointer.left.id!='in_ptr3'): continue
            offset=pointer.right
            if not isinstance(offset,ast.BinOp) or not isinstance(offset.left,ast.Constant):
                raise ValueError('Unexpected time-load template')
            offset.left.value+=(time_start-outputs)*channels
            changed+=1
    if changed!=outputs: raise ValueError('Time load count changed')
    return body


def first_segment_matches(function,**dimensions):
    return [ast.dump(x) for x in function.body]==[
        ast.dump(x) for x in first_segment_body(**dimensions)]


def check_first_segment(source,symbol,*,time_start):
    """Check the complete body, then reuse strict signature/type validation.

    Normalization is in-memory validation only. The returned identity always
    names the original function, never a rewritten executable.
    """
    from kernel_analyzer.decayed_recurrence_source import check_source
    assignments=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(assignments)!=1: raise ValueError('Missing or ambiguous segmented source')
    assignment=assignments[0]; call=assignment.value
    if (not isinstance(call,ast.Call) or len(call.args)<2
            or not isinstance(call.args[1],ast.Constant)
            or not isinstance(call.args[1].value,str)):
        raise ValueError('Literal segmented source required')
    tree=ast.parse(call.args[1].value)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1: raise ValueError('Unique segmented function required')
    fn=functions[0]; outputs=sum(a.arg.startswith('out_ptr') for a in fn.args.args)
    if not first_segment_matches(fn,channels=1536,width=16,outputs=outputs,time_start=time_start):
        raise ValueError('Segmented recurrence body differs')
    original_digest=hashlib.sha256(ast.dump(fn).encode()).hexdigest()
    fn.body=ast.parse(expected_body(outputs=outputs)).body
    call.args[1].value=ast.unparse(ast.fix_missing_locations(tree))
    checked=check_source(ast.unparse(assignment),symbol)
    return dict(checked,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                function_ast_sha256=original_digest,time_start=time_start,
                time_offsets_descending=list(range(time_start,time_start-outputs,-1)),
                segment_input_kind='BF16_OUTER_PRODUCT',runtime_binding_complete=False)
