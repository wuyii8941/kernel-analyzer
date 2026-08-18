#!/usr/bin/env python3
"""Summarize the five-checkpoint final-RMSNorm F+B materialization campaign."""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from scripts.long_horizon_trigger import atomic_json,under_root
METRICS=("candidate_minus_eager_projection","forward_materialization_removal_projection","backward_materialization_additional_removal_projection","joint_final_norm_removal_projection","joint_residual_projection")
def ci(v,n=20000):
 r=random.Random(20260805);m=len(v);s=sorted(sum(v[r.randrange(m)] for _ in range(m))/m for _ in range(n));return [s[int(.025*n)],s[int(.975*n)]]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",type=Path,action="append",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 paths=[under_root(x,"input") for x in a.input];out=under_root(a.output,"output");ps=[json.loads(x.read_text()) for x in paths]
 if sorted(int(x["checkpoint_step"]) for x in ps)!=[64,256,1024,2048,4096] or any(len(x["rows"])!=32 for x in ps):raise ValueError("incomplete campaign")
 by={i:[] for i in range(8,40)}
 for x in ps:
  for row in x["rows"]:by[row["state_index"]].append(row)
 summary={}
 for metric in METRICS:
  vals=[sum(r[metric] for r in rows)/5 for rows in by.values()]
  summary[metric]={"state_cluster_mean":sum(vals)/32,"state_cluster_bootstrap_95":ci(vals),"positive_states":sum(x>0 for x in vals),"states":32,"per_checkpoint_mean":{str(x["checkpoint_step"]):sum(r[metric] for r in x["rows"])/32 for x in ps}}
 total=summary["candidate_minus_eager_projection"]["state_cluster_mean"]
 result={"schema":"kernel-analyzer-l23-final-norm-summary-v1","status":"COMPLETE","checkpoints":[64,256,1024,2048,4096],"states":list(range(8,40)),"metric_summary":summary,"ratios_over_original_total":{"forward_materialization":summary["forward_materialization_removal_projection"]["state_cluster_mean"]/total,"backward_additional":summary["backward_materialization_additional_removal_projection"]["state_cluster_mean"]/total,"joint":summary["joint_final_norm_removal_projection"]["state_cluster_mean"]/total,"residual":summary["joint_residual_projection"]["state_cluster_mean"]/total},"validation":{"all_eager_forward_replays_bitwise_exact":all(r["eager_forward_replay_matches"] for x in ps for r in x["rows"]),"max_eager_vjp_replay_abs":max(r["eager_vjp_replay_max_abs"] for x in ps for r in x["rows"]),"max_stage_closure_abs":max(r["stage_closure_max_abs"] for x in ps for r in x["rows"]),"max_candidate_restoration_sham_abs":max(r["candidate_restoration_sham_max_abs"] for x in ps for r in x["rows"]),"tensor_values_saved":False},"inputs":[str(x) for x in paths]}
 result["result_sha256"]=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(",",":")).encode()).hexdigest();atomic_json(out,result);print(json.dumps({"output":str(out),"ratios":result["ratios_over_original_total"]},sort_keys=True))
if __name__=="__main__":main()
