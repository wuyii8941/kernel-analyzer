#!/usr/bin/env bash
set -u

# Run every concrete endpoint from the frozen legacy coherent-F+B set.  These
# are candidates until the 4096-step replay finishes; runtime failures remain
# unresolved and are never converted into negative labels.

ROOT="/data1/tzh/kernel-analyzer"
PY="/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
MANIFEST="$ROOT/results/property/declared_persistent_4096/legacy_coherent_long_replay_manifest.json"
OUT_ROOT="$ROOT/results/property/declared_persistent_4096/legacy_coherent_candidates"
LOG_ROOT="$OUT_ROOT/logs"
FAIL_ROOT="$OUT_ROOT/unresolved"
CACHE_ROOT="/data1/tzh/cache/bias_long_legacy_coherent"
GPU="${1:?usage: $0 GPU SHARD TOTAL_SHARDS}"
SHARD="${2:?usage: $0 GPU SHARD TOTAL_SHARDS}"
TOTAL="${3:?usage: $0 GPU SHARD TOTAL_SHARDS}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$FAIL_ROOT" "$CACHE_ROOT"

wait_for_gpu() {
  local required="$1" free
  while true; do
    free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v g="$GPU" '$1+0==g {gsub(/ /,"",$2); print $2}')
    if [[ -n "$free" && "$free" -ge "$required" ]]; then return 0; fi
    echo "[$(date -Is)] waiting gpu=$GPU free_memory=${free:-unknown} required=$required" >>"$LOG_ROOT/queue_gpu${GPU}.log"
    sleep 120
  done
}

is_complete() {
  local output="$1"
  [[ -s "$output" ]] || return 1
  "$PY" - "$output" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if str(payload.get("status", "")).startswith("COMPLETE") else 1)
PY
}

mapfile -t rows < <("$PY" - "$MANIFEST" "$SHARD" "$TOTAL" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
shard, total = int(sys.argv[2]), int(sys.argv[3])
rows = [row for row in payload["rows"] if row.get("runnable")]
for index, row in enumerate(rows):
    if index % total == shard:
        print(json.dumps(row, sort_keys=True))
PY
)

for row in "${rows[@]}"; do
  values=$("$PY" - "$row" <<'PY'
import json, sys
row = json.loads(sys.argv[1])
print("\t".join(str(row.get(key, "")) for key in (
    "case_id", "architecture", "model_path", "input_bank", "release_dir", "case_plan"
)))
PY
)
  IFS=$'\t' read -r case_id arch model input release plan <<<"$values"
  output="$OUT_ROOT/${case_id}_4096.json"
  checkpoint="$CACHE_ROOT/${case_id}.pt"
  log="$LOG_ROOT/${case_id}.log"
  if is_complete "$output"; then
    echo "[$(date -Is)] skip complete $case_id" >>"$log"
    continue
  fi
  if [[ -s "$checkpoint" ]] && "$PY" "$ROOT/scripts/finalize_consequence_checkpoint_on_loss_split.py" \
      --checkpoint "$checkpoint" --output "$output" --architecture "$arch" \
      --model "$model" --release-dir "$ROOT/$release" --case-plan "$ROOT/$plan" \
      --case-id "$case_id" --planned-horizon 4096 >>"$log" 2>&1; then
    echo "[$(date -Is)] COMPLETE early-loss-split $case_id" >>"$log"
    "$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$log" 2>&1 || true
    continue
  fi
  required=18000
  [[ "$arch" == "deepseek8b" ]] && required=42000
  wait_for_gpu "$required"
  if is_complete "$output"; then
    echo "[$(date -Is)] skip completed-while-waiting $case_id" >>"$log"
    continue
  fi
  echo "[$(date -Is)] START case=$case_id gpu=$GPU" >>"$log"
  cmd=("$PY" "$ROOT/scripts/run_bound_endpoint_consequence_v21.py"
    --architecture "$arch" --model "$model" --input-bank "$ROOT/$input"
    --release-dir "$ROOT/$release" --case-plan "$ROOT/$plan" --case-id "$case_id"
    --output "$output" --checkpoint "$checkpoint" --steps 4096 --keep-checkpoint
    --state-role TRAJECTORY --device cuda:0 --compact-long --stop-on-loss-split)
  [[ -s "$checkpoint" ]] && cmd+=(--resume)
  [[ "$arch" == "phi" ]] && cmd+=(--allow-graph-breaks)
  if CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_COMPILE_THREADS=1 TORCHINDUCTOR_WORKER_START=subprocess \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${cmd[@]}" >>"$log" 2>&1; then
    if ! "$PY" - "$output" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("consequence_only_early_stop") else 1)
PY
    then
      "$PY" "$ROOT/scripts/analyze_long_checkpoint_windows.py" --checkpoint "$checkpoint" \
        --output "$OUT_ROOT/${case_id}_4096_windows.json" >>"$log" 2>&1 || true
    fi
    echo "[$(date -Is)] COMPLETE case=$case_id" >>"$log"
    "$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$log" 2>&1 || true
  else
    code=$?
    "$PY" - "$case_id" "$code" "$log" "$output" >"$FAIL_ROOT/${case_id}.json" <<'PY'
import json, sys
case_id, code, log, output = sys.argv[1:]
print(json.dumps({
    "schema": "legacy-coherent-long-replay-failure-v1",
    "case_id": case_id,
    "status": "UNRESOLVED_LONG_REPLAY_RESOURCE",
    "exit_code": int(code),
    "log": log,
    "partial_output": output,
    "claim_boundary": "The frozen bias candidate could not complete its 4096-step replay; it remains unresolved, never negative.",
}, indent=2, sort_keys=True))
PY
    echo "[$(date -Is)] FAILED case=$case_id exit=$code" >>"$log"
    "$PY" "$ROOT/scripts/build_all_bias_case_audit.py" >>"$log" 2>&1 || true
  fi
done

echo "[$(date -Is)] queue complete gpu=$GPU shard=$SHARD/$TOTAL" >>"$LOG_ROOT/queue_gpu${GPU}.log"
