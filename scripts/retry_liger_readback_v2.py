#!/usr/bin/env python3
"""Retry the frozen Liger case using its existing local source package only.

Do not add the other environment's whole site-packages to sys.path: that would
silently change PyTorch and Transformers. The failed first attempt is retained.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from kernel_analyzer.training_numerical_analysis import analyze_artifact
from scripts.run_training_numerical_v2 import BASE, ROOT, hashes, save_new


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cuda:1")
    p.add_argument("--attempt", choices=("liger_local_package", "liger_local_package_logical_cuda0", "liger_logical_cuda0_interruption_retry"), default="liger_local_package")
    p.add_argument("--package",type=Path,default=Path("/data1/tzh/envs/liger/lib/python3.10/site-packages/liger_kernel"))
    args=p.parse_args()
    protocol=json.loads((BASE/"protocol.json").read_text())
    if hashes()!=protocol["source_sha256"]:
        raise SystemExit("frozen capture sources changed")
    attempt=BASE/"attempts"/args.attempt
    manifest={"case_id":"liger_fused_ce_t128","reason":"FIRST_ATTEMPT_MISSING_LIGER_PACKAGE",
        "data_use":"EXISTING_CASE_RECAPTURE","torch_version":torch.__version__,
        "package_directory":str(args.package),"package_sources":{
            str(p.relative_to(args.package)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.package.rglob("*.py"))},"capture_sources":hashes()}
    if not manifest["package_sources"]:
        raise SystemExit("local package unavailable")
    save_new(attempt/"protocol.json",manifest)
    spec=importlib.util.spec_from_file_location("liger_kernel",args.package/"__init__.py",
                                               submodule_search_locations=[str(args.package)])
    module=importlib.util.module_from_spec(spec); sys.modules["liger_kernel"]=module
    spec.loader.exec_module(module)
    from scripts.run_training_bias_profile_v2_empirical import run_liger
    try:
        raw=run_liger(torch.device(args.device))
    except Exception as error:
        save_new(attempt/"failure.json",{"status":"EXECUTION_FAILED","type":type(error).__name__,"error":str(error)})
        raise
    save_new(attempt/"raw.json",raw)
    report=analyze_artifact(raw,protocol)
    report["provenance"].update({"retry_manifest":str(attempt/"protocol.json"),
                                 "raw_sha256":hashlib.sha256((attempt/"raw.json").read_bytes()).hexdigest()})
    save_new(attempt/"recomputed.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
