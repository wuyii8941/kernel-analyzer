"""CPU-only declaration checks before launching a recurrence capture."""
import ast
import re
from pathlib import Path
from scripts.run_numerical_coverage import read, sha
from scripts.run_decayed_recurrence_capture import plan_method, select_contracts


def declared_device(source, symbol):
    assignments=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id==symbol for t in n.targets)]
    if len(assignments)!=1: raise ValueError('Ambiguous source symbol')
    call=assignments[0].value
    if (not isinstance(call,ast.Call) or len(call.args)<2
            or not isinstance(call.args[1],ast.Constant) or not isinstance(call.args[1].value,str)):
        raise ValueError('Literal source required')
    tree=ast.parse(call.args[1].value)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1: raise ValueError('Unique function required')
    indices=[]
    for decorator in functions[0].decorator_list:
        for node in ast.walk(decorator):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='DeviceProperties':
                values=[kw.value for kw in node.keywords if kw.arg=='index']
                if len(values)!=1 or not isinstance(values[0],ast.Constant) or type(values[0].value) is not int:
                    raise ValueError('Explicit device index required')
                indices.append(values[0].value)
    if len(indices)!=1: raise ValueError('Unique device declaration required')
    return indices[0]


def preflight(command):
    if not any(Path(x).name=='run_decayed_recurrence_capture.py' for x in command):
        return None
    def argument(name):
        if command.count(name)!=1: raise ValueError('Unique argument required: '+name)
        i=command.index(name)
        if i+1>=len(command): raise ValueError('Missing argument: '+name)
        return command[i+1]
    plan_path=Path(argument('--recurrence-plan'))
    partition_path=Path(argument('--case-plan'))
    plan=read(plan_path)
    plan_method(plan)
    select_contracts(plan,read(partition_path)['cases'])
    target=re.fullmatch(r'cuda:(\d+)',argument('--device'))
    if target is None: raise ValueError('Explicit logical CUDA index required')
    sources=[]
    for name,digest in plan['source_sha256'].items():
        path=Path(name)
        if sha(path)!=digest: raise ValueError('Frozen recurrence source changed: '+name)
        if digest==plan['contract']['source_sha256']: sources.append(path)
    if len(sources)!=1: raise ValueError('Unique declared source file required')
    device=declared_device(sources[0].read_text(),plan['contract']['symbol'])
    if device!=int(target[1]):
        raise ValueError(f'Declared source uses cuda:{device}, requested {target[0]}; explicitly redeclare device first')
    return dict(status='DECLARATIONS_CHECKED_NOT_RUNTIME_PROOF', device=target[0],
                plan_sha256=sha(plan_path),partition_sha256=sha(partition_path),
                checker_sha256=sha(Path(__file__)))
