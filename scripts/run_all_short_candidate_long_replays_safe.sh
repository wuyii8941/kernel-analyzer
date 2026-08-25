#!/usr/bin/env bash
set -u

# Long replay queue for every candidate in the frozen short-screen map that
# has an exact confirmation and trajectory bank.  It deliberately waits for a
# dedicated GPU and writes failures as unresolved.  The existing 12-row queue
# owns its original output paths, so those IDs are skipped here to avoid a
# duplicate experiment.

ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
MANIFEST="$ROOT/results/property/declared_persistent_4096/all_short_candidate_long_replay_manifest.json"
LOG_ROOT="$ROOT/results/property/declared_persistent_4096/all_candidates/logs"
OUT_ROOT="$ROOT/results/property/declared_persistent_4096/all_candidates"
FAIL_ROOT="$ROOT/results/property/declared_persistent_4096/all_candidates/unresolved"
CACHE_ROOT="/data1/tzh/cache/bias_long_all_candidates"
GPU="${1:?usage: $0 GPU SHARD TOTAL_SHARDS}"
SHARD="${2:?usage: $0 GPU SHARD TOTAL_SHARDS}"
TOTAL="${3:?usage: $0 GPU SHARD TOTAL_SHARDS}"
mkdir -p "$LOG_ROOT" "$OUT_ROOT" "$FAIL_ROOT" "$CACHE_ROOT"

wait_for_gpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v g="$GPU" '$1+0==g {gsub(/ /,"",$2); print $2}')
    if [[ -n "$free" && "$free" -ge 12000 ]]; then return 0; fi
    echo "[$(date -Is)] waiting gpu=$GPU free_memory=${free:-unknown}" >>"$LOG_ROOT/queue_gpu${GPU}.log"
    sleep 120
  done
}

python_row() {
  "$PY" - "$1" <<'PY'
import json,sys
r=json.loads(sys.argv[1])
vals=[r.get(k,"") for k in ("case_id","architecture","model_path","input_bank","release_dir","case_plan")]
print("\t".join(str(x) for x in vals))
PY
}

skip_existing() {
  case "$1" in
    multishape-backward-cell-0057|multishape-backward-cell-0103|multishape-backward-cell-0153|multishape-backward-cell-0190|multishape-backward-cell-0191|multishape-backward-cell-0450|multishape-backward-cell-0501|multishape-backward-cell-0508|multishape-backward-cell-0543|multishape-backward-cell-0654|multishape-backward-cell-0745|multishape-backward-cell-0747) return 0;;
    *) return 1;;
  esac
}

mapfile -t rows < <("$PY" - "$MANIFEST" "$SHARD" "$TOTAL" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); shard=int(sys.argv[2]); total=int(sys.argv[3])
rows=[r for r in d["rows"] if r.get("runnable")]
for i,r in enumerate(rows):
    if i % total == shard: print(json.dumps(r,sort_keys=True))
PY
)

for row in "${rows[@]}"; do
  IFS=$'\t' read -r case_id arch model input release plan <<<"$(python_row "$row")"
  if skip_existing "$case_id"; then
    echo "[$(date -Is)] skip owned-by-existing-queue $case_id" >>"$LOG_ROOT/queue_gpu${GPU}.log"
    continue
  fi
  output="$OUT_ROOT/${case_id}_4096.json"
  checkpoint="$CACHE_ROOT/${case_id}.pt"
  log="$LOG_ROOT/${case_id}.log"
  if [[ -s "$output" ]] && "$PY" - "$output" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: raise SystemExit(1)
raise SystemExit(0 if str(d.get("status","")).startswith("COMPLETE") else 1)
PY
  then
    echo "[$(date -Is)] skip complete $case_id" >>"$log"; continue
  fi
  wait_for_gpu
  echo "[$(date -Is)] START case=$case_id gpu=$GPU" >>"$log"
  cmd=("$PY" "$ROOT/scripts/run_bound_endpoint_consequence_v21.py"
    --architecture "$arch" --model "$model" --input-bank "$ROOT/$input"
    --release-dir "$ROOT/$release" --case-plan "$ROOT/$plan" --case-id "$case_id"
    --output "$output" --checkpoint "$checkpoint" --steps 4096 --keep-checkpoint
    --state-role TRAJECTORY --device cuda:0 --compact-long)
  [[ "$arch" == "phi" ]] && cmd+=(--allow-graph-breaks)
  if CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${cmd[@]}" >>"$log" 2>&1; then
    "$PY" "$ROOT/scripts/analyze_long_checkpoint_windows.py" --checkpoint "$checkpoint" \
      --output "$OUT_ROOT/${case_id}_4096_windows.json" >>"$log" 2>&1 || true
    echo "[$(date -Is)] COMPLETE case=$case_id" >>"$log"
    "$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$log" 2>&1 || true
  else
    code=$?
    "$PY" - "$case_id" "$code" "$log" "$output" >"$FAIL_ROOT/${case_id}.json" <<'PY'
import json,sys
case_id,code,log,output=sys.argv[1:]
print(json.dumps({"schema":"all-short-candidate-long-replay-failure-v1","case_id":case_id,"status":"UNRESOLVED_LONG_REPLAY_RESOURCE","exit_code":int(code),"log":log,"partial_output":output,"claim_boundary":"The candidate was scheduled for a 4096-step replay; a failed runtime is unresolved, never negative."},indent=2,sort_keys=True))
PY
    echo "[$(date -Is)] FAILED case=$case_id exit=$code" >>"$log"
    "$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$log" 2>&1 || true
  fi
done

echo "[$(date -Is)] queue complete gpu=$GPU shard=$SHARD/$TOTAL" >>"$LOG_ROOT/queue_gpu${GPU}.log"
