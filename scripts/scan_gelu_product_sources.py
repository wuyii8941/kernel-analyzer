"""Discover supported GELU product layouts, then check the entire definition."""
import argparse
import ast
from pathlib import Path
from scripts.run_numerical_coverage import save, sha
from kernel_analyzer.gelu_product_source import check_source


def scan(text):
    records=[]
    for node in ast.parse(text).body:
        if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Call): continue
        args=node.value.args
        if len(args)<2 or not isinstance(args[1],ast.Constant) or not isinstance(args[1].value,str): continue
        try: definitions=ast.parse(args[1].value).body
        except SyntaxError: continue
        for fn in definitions:
            if not isinstance(fn,ast.FunctionDef) or 'gelu' not in fn.name: continue
            row=dict(symbol=fn.name,status='UNSUPPORTED_SOURCE_FORM',runtime_measurement_complete=False)
            try:
                values={t.id:a.value for a in fn.body if isinstance(a,ast.Assign)
                        for t in a.targets if isinstance(t,ast.Name)}
                elements=ast.literal_eval(values['xnumel'])
                width=ast.literal_eval(values['x0'].right)
                address=values['tmp1'].func.value.args[0].right
                offset=ast.literal_eval(address.left.left)
                stride=ast.literal_eval(address.right.left)
                # The isolated assignment retains the complete literal program
                # and decorators. The enclosing file is hashed separately.
                contract=check_source(ast.unparse(node),fn.name,elements=elements,
                                      width=width,stride=stride,offset=offset)
                row.update(status='SOURCE_CHECKED',contract=contract)
            except (ValueError,KeyError,AttributeError,TypeError) as exc:
                row['reason']=str(exc)
            records.append(row)
    if len({r['symbol'] for r in records})!=len(records):
        raise ValueError('Duplicate generated GELU symbol')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    sources=[dict(source=str(path.resolve()),source_sha256=sha(path),rows=scan(path.read_text()))
             for path in sorted(set(a.sources))]
    dependencies=[Path(__file__),Path(__file__).resolve().parents[1]/'src/kernel_analyzer/gelu_product_source.py',
                  Path(__file__).resolve().parents[1]/'src/kernel_analyzer/gelu_product_reference.py']
    report=dict(schema='gelu-product-source-scan-v1',sources=sources,
        source_sha256={str(path.resolve()):sha(path) for path in dependencies},
        scope='Source definitions only; not runtime support, operator-family count or bias evidence',
        reference_family='GELU_PRODUCT_BACKWARD',selection_uses_numerical_results=False)
    save(a.output,report)
    print(dict(definitions=sum(len(s['rows']) for s in sources),
               checked=sum(r['status']=='SOURCE_CHECKED' for s in sources for r in s['rows'])))


if __name__=='__main__': main()
