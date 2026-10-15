#!/usr/bin/env python3
"""Run reviewer checks without masking exit codes; explicitly not a production certification runner."""
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path);a=p.parse_args()
 commands=[[sys.executable,str(ROOT/'tools/validate_contracts.py')],[sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v']]
 if a.source_root:commands.append([sys.executable,str(ROOT/'tools/check_witnesses.py'),'--source-root',str(a.source_root)])
 for cmd in commands:
  r=subprocess.run(cmd,cwd=ROOT,check=False)
  if r.returncode:return r.returncode
 if not a.source_root:print('Original-spec witness execution NOT RUN: --source-root was not supplied.')
 print('Reviewer checks completed. Production adapters and runtime activation remain pending.')
 return 0
if __name__=='__main__':sys.exit(main())
