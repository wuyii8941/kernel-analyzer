#!/usr/bin/env python3
"""Validate the same artifact-to-decision path used by the v2 GPU runner."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

from kernel_analyzer.training_numerical_analysis import analyze_artifact
from kernel_analyzer.training_equivalence import population_total_energy_equivalence


def make_artifact(u,r):
    n=len(u)
    return {
        "case_id":"synthetic", "status":"COMPLETE", "contrast_id":"CONTROLLED_SYNTHETIC_EFFECT",
        "runtime_boundary":{"kind":"SYNTHETIC_NOT_A_NATURAL_TRAINING_CASE"},
        "parameter_write_protocol":{"version":"adamw-readback-v2","synthetic":True},
        "state_ids":list(range(n)),"calibration_state_ids":list(range(n//2)),
        "confirmation_state_ids":list(range(n//2,n)),
        "original_coordinate_statistics":{"PARAMETER_WRITE":[{
            "effect_energy":float(x@x),"repair_energy":float(y@y),
            "effect_repair_inner_product":float(x@y),"nonzero_effect_coordinates":int(np.count_nonzero(x)),
        } for x,y in zip(u,r)]},
        "stages":{"PARAMETER_WRITE":{"EXACT":{"profile":{"suite":{"joint_gram":{
            "effect_effect":(u@u.T).tolist(),"repair_repair":(r@r.T).tolist(),
            "effect_repair":(u@r.T).tolist(),
        }}}}}},
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--repetitions",type=int,default=1000); args=p.parse_args()
    if args.output.exists() or args.repetitions<100:
        raise SystemExit("Use a new output and at least 100 repetitions.")
    protocol={"schema":"v2-development-validation","claim_scope":"FIXED_SUITE_UPDATE",
              "primary_stage":"PARAMETER_WRITE","fixed_suite_margins":{"full_update_rms":.01}}
    rng=np.random.default_rng(20260906); errors=0; records=[]
    for shape in ("gaussian","heavy_tail","correlated","energy_heterogeneity","orthogonal_drift"):
        for scale in (.0095,.01,.0105):
            counts={}; mismatches=0
            for _ in range(args.repetitions):
                r=rng.normal(size=(32,16)); r/=np.linalg.norm(r,axis=1)[:,None]
                u=rng.normal(size=r.shape)
                if shape=="heavy_tail": u=rng.standard_t(3,size=r.shape)
                if shape=="correlated": u[:]=u[0]
                if shape=="energy_heterogeneity": r*=np.geomspace(1e-3,1e3,32)[:,None]
                if shape=="orthogonal_drift":
                    u[:]=0; u[:16,0]=1e-6; u[16:,1]=1
                u*=scale*math.sqrt(float(np.sum(r*r)/np.sum(u*u)))
                raw=make_artifact(u,r)
                report=analyze_artifact(raw,protocol)
                # Truth is this exact confirmation suite, NOT the whole-bank scale.
                q=float(np.sum(u[16:]**2)/np.sum(r[16:]**2))
                decision=report["equivalence_decision"]
                wrong=(decision=="EQUIVALENT" and q>.01**2*(1+1e-12)) or (decision=="NON_EQUIVALENT" and q<.01**2*(1-1e-12))
                mismatches+=bool(wrong); counts[decision]=counts.get(decision,0)+1
            errors+=mismatches
            records.append({"scenario":shape,"whole_bank_scale":scale,"counts":counts,
                            "false_fixed_suite_decisions":mismatches,"repetitions":args.repetitions})
    population=[]
    for n in (16,64):
        for dist in ("gamma","heavy_tail"):
            passes=0
            for _ in range(args.repetitions):
                # Independent unit energies at the population equivalence boundary.
                x=(rng.gamma(2,.5,n) if dist=="gamma" else rng.standard_t(3,n)**2/3)*.01**2
                out=population_total_energy_equivalence(x,np.ones(n),rms_margin=.01)
                passes+=out["decision"]=="EQUIVALENT"
            rate=passes/args.repetitions
            population.append({"distribution":dist,"independent_units":n,"false_equivalence_rate":rate,
                               "monte_carlo_standard_error":math.sqrt(rate*(1-rate)/args.repetitions),
                               "scope":"ASYMPTOTIC_DIAGNOSTIC_NOT_A_FIXED_SUITE_GUARANTEE"})
    payload={"status":"PASS_FIXED_SUITE_PATH" if errors==0 else "FAIL",
             "production_function":"kernel_analyzer.training_numerical_analysis.analyze_artifact",
             "fixed_suite_results":records,"population_boundary_diagnostics":population,
             "population_certificate_enabled":False,
             "limitations":"Finite-suite checks do not validate arbitrary population inference; heavy-tailed unit energy may violate the required variance assumptions."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f: json.dump(payload,f,indent=2,allow_nan=False); f.write("\n")
    print(json.dumps({"status":payload["status"],"fixed_suite_errors":errors,"population":population},indent=2))
    if errors: raise SystemExit(1)


if __name__=="__main__": main()
