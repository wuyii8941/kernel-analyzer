#!/usr/bin/env python3
"""Observe a failed source guard without relaxing it or measuring another port."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from scripts import recover_gemma_square_sum_boundary as recovery
from scripts.run_training_numerical_v2 import save_new


def main():
    original_check=recovery.check
    output=recovery.OUT/'live_source_failure.json'
    def check(text):
        try:return original_check(text)
        except ValueError as error:
            from torch._inductor.codecache import PyCodeCache
            digest=hashlib.sha256(text.encode()).hexdigest()
            paths=[]
            for module in PyCodeCache.modules:
                path=Path(module.__file__)
                if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:paths.append(str(path))
            fn=recovery.kernel(text)
            stores=[ast.unparse(n) for n in ast.walk(fn) if isinstance(n,ast.Call)
                    and isinstance(n.func,ast.Attribute) and n.func.attr=='store'
                    and any(isinstance(a,ast.Name) and a.id=='out_ptr0' for a in ast.walk(n.args[0]))]
            save_new(output,{'status':'ACTUAL_EXECUTION_SOURCE_MISMATCH','error':str(error),
                'actual_wrapper_paths':paths,'actual_wrapper_sha256':digest,'actual_kernel_ast_sha256':hashlib.sha256(ast.dump(fn).encode()).hexdigest(),
                'actual_out_ptr0_stores':stores,'actual_function_arguments':[a.arg for a in fn.args.args],
                'original_out_ptr0_store':'tl.store(out_ptr0 + (x0), tmp4, xmask)',
                'old_source':str(recovery.OLD/'trace/model__0_forward_segment0_executed/output_code.py'),
                'replacement_used':False,'measurements_accepted':False,
                'scope':'The executed kernel with this name is not the saved implementation. This blocks original-boundary recapture in the current environment; it is not negative bias evidence.'})
            raise
    recovery.check=check
    sys.argv=['recover_gemma_square_sum_boundary.py','capture','--device','cuda:0']
    recovery.main()


if __name__=='__main__':main()
