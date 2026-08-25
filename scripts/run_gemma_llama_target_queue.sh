#!/usr/bin/env bash
set -u

# Run the frozen Gemma/Llama target set after existing long jobs release GPU 3.
# Pilot first; only a pilot risk starts a 4096-step consequence. Failures stay
# unresolved and are never converted to negative labels.
ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
MANIFEST="$ROOT/results/property/declared_persistent_4096/operator_scan_target_manifest.json"
LOG="$ROOT/results/property/declared_persistent_4096/operator_scan_targets_queue.log"
GPU="${1:-3}"

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v g="$GPU" '$1+0==g {gsub(/ /,"",$2); print $2}')
    if [[ -n "$used" && "$used" -lt 3000 ]]; then return 0; fi
    echo "[$(date -Is)] waiting for GPU${GPU}, memory=${used:-unknown} MiB" >>"$LOG"
    sleep 120
  done
}

run_one() {
  local row_json="$1"
  local case_id arch model input consequence carrier region endpoint symbol
  case_id=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["case_id"])')
  arch=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["architecture"])')
  model=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["model_path"])')
  input=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["input_bank"])')
  consequence=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["consequence_bank"])')
  carrier=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["carrier"])')
  region=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["target_region"])')
  endpoint=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["target_endpoint"])')
  symbol=$(echo "$row_json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["target_symbol"])')
  local base="$ROOT/results/property/declared_persistent_4096/operator_scan_targets"
  local pilot="$base/${case_id}_pilot" long="$base/${case_id}"
  if [[ -f "$long/consequence.json" ]]; then echo "[$(date -Is)] skip complete $case_id" >>"$LOG"; return 0; fi
  # Reuse any completed pilot from a manual retry instead of creating a
  # duplicate process.  A failed/partial directory is left as evidence and a
  # fresh suffix is used below.
  local completed_pilot=""
  for candidate in "$base/${case_id}"_pilot*; do
    if [[ -f "$candidate/prediction.json" && -f "$candidate/short_screen.json" ]]; then
      completed_pilot="$candidate"
    fi
  done
  if [[ -n "$completed_pilot" ]]; then
    pilot="$completed_pilot"
  else
    local suffix=0
    while [[ -e "$pilot" ]]; do suffix=$((suffix + 1)); pilot="$base/${case_id}_pilot_retry${suffix}"; done
  fi
  if [[ -f "$pilot/prediction.json" && -f "$pilot/short_screen.json" ]]; then
    local decision
    decision=$("$PY" -c 'import json,sys; p=sys.argv[1]; pred=json.load(open(p+"/prediction.json")).get("source_prediction"); short=json.load(open(p+"/short_screen.json")); print("LONG" if pred=="SOURCE_PERSISTENCE_RISK" or any(x.get("status")=="RISK_CANDIDATE" for x in short.get("cases",[])) else "STOP")' "$pilot")
    if [[ "$decision" != LONG ]]; then echo "[$(date -Is)] reuse pilot no-risk $case_id" >>"$LOG"; return 0; fi
  fi
  wait_for_gpu
  echo "[$(date -Is)] pilot $case_id region=$region endpoint=$endpoint" >>"$LOG"
  if ! CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" "$ROOT/scripts/run_gemma4_v3_validation.py" --architecture "$arch" --model "$model" \
      --input-bank "$input" --consequence-bank "$consequence" --output-dir "$pilot" --steps 16 \
      --consequence-steps 16 --target-region "$region" --target-symbol "$symbol" --target-endpoint "$endpoint" --carrier "$carrier" \
      --case-id "$case_id" --learning-rate 1e-5 --device cuda:0 --null-draws 1000 \
      $([[ "$arch" == "gemma4" ]] && echo --allow-graph-breaks) >>"$LOG" 2>&1; then
    echo "[$(date -Is)] pilot unresolved $case_id" >>"$LOG"; return 0
  fi
  local decision
  decision=$("$PY" -c 'import json,sys; p=sys.argv[1]; pred=json.load(open(p+"/prediction.json")).get("source_prediction"); short=json.load(open(p+"/short_screen.json")); print("LONG" if pred=="SOURCE_PERSISTENCE_RISK" or any(x.get("status")=="RISK_CANDIDATE" for x in short.get("cases",[])) else "STOP")' "$pilot")
  if [[ "$decision" != LONG ]]; then echo "[$(date -Is)] pilot no-risk $case_id" >>"$LOG"; return 0; fi
  wait_for_gpu
  echo "[$(date -Is)] long $case_id" >>"$LOG"
  if ! CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" "$ROOT/scripts/run_gemma4_v3_validation.py" --architecture "$arch" --model "$model" \
      --input-bank "$input" --consequence-bank "$consequence" --output-dir "$long" --steps 16 \
      --consequence-steps 4096 --target-region "$region" --target-symbol "$symbol" --target-endpoint "$endpoint" --carrier "$carrier" \
      --case-id "$case_id" --learning-rate 1e-5 --device cuda:0 --null-draws 2000 \
      $([[ "$arch" == "gemma4" ]] && echo --allow-graph-breaks) >>"$LOG" 2>&1; then
    mkdir -p "$base/unresolved"
    "$PY" -c 'import json,sys; print(json.dumps({"schema":"kernel-analyzer-target-replay-failure-v1","case_id":sys.argv[1],"status":"UNRESOLVED_LONG_REPLAY_RESOURCE","log":sys.argv[2],"output_dir":sys.argv[3],"claim_boundary":"Target replay failed; no negative label is assigned."},indent=2))' "$case_id" "$LOG" "$long" >"$base/unresolved/${case_id}.json"
  fi
}

mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] queue start" >>"$LOG"
while IFS= read -r row; do run_one "$row"; done < <("$PY" -c 'import json,sys; [print(json.dumps(r,sort_keys=True)) for r in json.load(open(sys.argv[1]))["rows"]]' "$MANIFEST")
"$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$LOG" 2>&1 || true
echo "[$(date -Is)] queue complete" >>"$LOG"
