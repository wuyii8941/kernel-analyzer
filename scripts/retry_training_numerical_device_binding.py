#!/usr/bin/env python3
"""Retry the same frozen graph on logical cuda:0 without relaxing identity gates."""
import argparse
import json
import os
from pathlib import Path
import sys

from scripts import run_training_numerical_v2 as pipeline


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=("deepseek_norm","deepseek_attn"),required=True)
    parser.add_argument("--physical-device",choices=("0","1","2","3"),required=True)
    args=parser.parse_args()
    parent=pipeline.BASE
    attempt=parent/"attempts"/(args.case+"_logical_cuda0")
    protocol=json.loads((parent/"protocol.json").read_text())
    pipeline.save_new(attempt/"protocol.json",protocol)
    pipeline.save_new(attempt/"retry_reason.json",{
        "reason":"Frozen AOT graph uses logical cuda:0; preserve the original identity gate and remap the physical GPU instead.",
        "physical_device":args.physical_device,"parent_protocol":str(parent/"protocol.json"),
        "numerical_protocol_changed":False,"identity_checks_relaxed":False,
    })
    os.environ["CUDA_VISIBLE_DEVICES"]=args.physical_device
    pipeline.BASE=attempt
    sys.argv=[sys.argv[0],"capture","--case",args.case,"--device","cuda:0"]
    pipeline.main()


if __name__=="__main__": main()
